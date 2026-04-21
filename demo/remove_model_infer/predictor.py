# Copyright 2026, MiLM Plus, Xiaomi Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import numpy as np
import torch
from diffusers import FlowMatchEulerDiscreteScheduler
from omegaconf import OmegaConf
from PIL import Image
from transformers import AutoTokenizer
import scipy
import cv2
from glob import glob
from threading import Lock
from typing import List
from videox_fun.dist import set_multi_gpus_devices
from videox_fun.models import AutoencoderKLWan, WanT5EncoderModel, VaceWanModel
from videox_fun.pipeline import SVORPipeline
from videox_fun.utils.fp8_optimization import (
    convert_model_weight_to_float8,
    replace_parameters_by_name,
    convert_weight_dtype_wrapper,
)
from videox_fun.utils.lora_utils import merge_lora
from videox_fun.utils.utils import filter_kwargs, postprocess_videoframe
from .data_types import VideoEditRequest


def load_patch_safetensors(path):
    list_tensors = glob(path + "/*.safetensors")
    all = {}
    for x in list_tensors:
        from safetensors.torch import load_file

        tmp = load_file(x)
        all.update(tmp)
    return all


def process_video(
    input_video_path,
    input_mask_video_path,
    video_length,
    sample_size,
    dilation=0,
):
    """Process input video and mask for editing"""

    if input_video_path is not None:
        cap = cv2.VideoCapture(input_video_path)
        frames = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(frame))
        cap.release()
        print(f"[Info] num_frame: {len(frames)}")
        resized_frames = [frame.resize([sample_size[1], sample_size[0]]) for frame in frames]

        input_video = (
            torch.stack([torch.from_numpy(np.array(frame)).permute(2, 0, 1) for frame in resized_frames])
            .permute(1, 0, 2, 3)
            .unsqueeze(0)
        )  # [1, C, T, H, W]

    else:
        input_video = torch.zeros((1, 3, video_length, sample_size[0], sample_size[1])).float()
    if input_mask_video_path is not None:
        mask_cap = cv2.VideoCapture(input_mask_video_path)
        mask_frames = []
        while mask_cap.isOpened():
            ret, frame = mask_cap.read()
            if not ret:
                break
            if len(frame.shape) == 3:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            _, mask = cv2.threshold(frame, 127, 255, cv2.THRESH_BINARY)
            if dilation > 0:
                mask_np = (mask > 0).astype(np.uint8)
                mask = (
                    scipy.ndimage.binary_dilation(
                        mask_np,
                        iterations=dilation,
                    ).astype(np.uint8)
                    * 255
                )
            mask_frames.append(mask)
        mask_cap.release()

        resized_masks = [Image.fromarray(mask).resize([sample_size[1], sample_size[0]]) for mask in mask_frames]

        input_video_mask = (
            torch.stack([torch.from_numpy(np.array(mask)) for mask in resized_masks]).unsqueeze(0).unsqueeze(0) / 255.0
        )  # [1, 1, T, H, W]

    else:
        input_video_mask = torch.ones((1, 1, video_length, sample_size[0], sample_size[1])).float()

    input_video = input_video.float().div_(127.5).sub_(1.0)

    return input_video, input_video_mask


class SVORpredictor:
    def __init__(self, args):
        weight_dtype = torch.bfloat16 if args.weight_dtype == "bfloat16" else torch.float16

        device = set_multi_gpus_devices(args.ulysses_degree, args.ring_degree)
        config = OmegaConf.load(args.config_path)

        transformer = VaceWanModel.from_pretrained(
            os.path.join(
                args.model_name, config["transformer_additional_kwargs"].get("transformer_subpath", "transformer")
            ),
            transformer_additional_kwargs=OmegaConf.to_container(config["transformer_additional_kwargs"]),
            low_cpu_mem_usage=True,
            torch_dtype=weight_dtype,
        )
        # Get Vae
        vae = AutoencoderKLWan.from_pretrained(
            os.path.join(args.model_name, config["vae_kwargs"].get("vae_subpath", "vae")),
            additional_kwargs=OmegaConf.to_container(config["vae_kwargs"]),
        ).to(weight_dtype)
        # Get Tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            os.path.join(args.model_name, config["text_encoder_kwargs"].get("tokenizer_subpath", "tokenizer")),
        )
        # Get Text encoder
        text_encoder = WanT5EncoderModel.from_pretrained(
            os.path.join(args.model_name, config["text_encoder_kwargs"].get("text_encoder_subpath", "text_encoder")),
            additional_kwargs=OmegaConf.to_container(config["text_encoder_kwargs"]),
        ).to(weight_dtype)
        text_encoder = text_encoder.eval()

        # Get Scheduler
        Choosen_Scheduler = {
            "Flow": FlowMatchEulerDiscreteScheduler,
        }["Flow"]
        scheduler = Choosen_Scheduler(
            **filter_kwargs(Choosen_Scheduler, OmegaConf.to_container(config["scheduler_kwargs"]))
        )

        # Get Pipeline
        pipeline = SVORPipeline(
            transformer=transformer,
            vae=vae,
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            scheduler=scheduler,
        )

        if args.ulysses_degree > 1 or args.ring_degree > 1:
            transformer.enable_multi_gpus_inference()

        if args.gpu_memory_mode == "sequential_cpu_offload":
            replace_parameters_by_name(
                transformer,
                [
                    "modulation",
                ],
                device=device,
            )
            transformer.freqs = transformer.freqs.to(device=device)
            pipeline.enable_sequential_cpu_offload(device=device)
        elif args.gpu_memory_mode == "model_cpu_offload_and_qfloat8":
            convert_model_weight_to_float8(
                transformer,
                exclude_module_name=[
                    "modulation",
                ],
            )
            convert_weight_dtype_wrapper(transformer, weight_dtype)
            pipeline.enable_model_cpu_offload(device=device)
        elif args.gpu_memory_mode == "model_cpu_offload":
            pipeline.enable_model_cpu_offload(device=device)
        else:
            pipeline.to(device=device)

        if args.lora_path is not None:
            if len(args.lora_weight) != len(args.lora_path):
                args.lora_weight = [args.lora_weight[0]] * len(args.lora_path)
            for lora_path, lora_weight in zip(args.lora_path, args.lora_weight):
                print(f"[INFO] Loading LoRA: {lora_path}, weight: {lora_weight}")
                pipeline = merge_lora(pipeline, lora_path, lora_weight)

        self.pipeline = pipeline
        self.global_prompt = "Remove the target and fill the content appropriately"
        self.inference_lock = Lock()
        self.args = args

    def _process_single_task(
        self,
        input_video_path,
        input_mask_video_path,
        original_frame_count=None,
    ):
        # Get video resolution
        cap = cv2.VideoCapture(input_video_path)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        print(f"[Info] num_frames: {original_frame_count}")
        # Calculate aspect ratio preserving size
        aspect_ratio = height / width
        max_area = int(self.args.sample_size.split(",")[0]) * int(self.args.sample_size.split(",")[1])
        new_height = round(np.sqrt(max_area * aspect_ratio))
        new_height = (new_height + 16 - 1) // 16 * 16
        new_width = round(np.sqrt(max_area / aspect_ratio))
        new_width = (new_width + 16 - 1) // 16 * 16
        sample_size = [new_height, new_width]

        generator = torch.Generator(device=self.pipeline.device).manual_seed(self.args.seed)

        with torch.no_grad():
            video_length = (
                int(
                    (self.args.video_length - 1)
                    // self.pipeline.vae.config.temporal_compression_ratio
                    * self.pipeline.vae.config.temporal_compression_ratio
                )
                + 1
                if self.args.video_length != 1
                else 1
            )

            # Process video and mask
            input_video, input_video_mask = process_video(
                input_video_path,
                input_mask_video_path,
                video_length=video_length,
                sample_size=sample_size,
                dilation=self.args.dilation,
            )

            sample = self.pipeline(
                self.args.prompt,
                negative_prompt=self.args.negative_prompt,
                height=sample_size[0],
                width=sample_size[1],
                generator=generator,
                guidance_scale=self.args.guidance_scale,
                num_inference_steps=self.args.num_inference_steps,
                video=input_video,
                mask_video=input_video_mask,
                context_scale=self.args.context_scale,
            ).videos

            outputs = postprocess_videoframe(sample)
            current_count = len(outputs)
            if current_count > original_frame_count:
                outputs = outputs[:original_frame_count]

            return outputs

    def predict(self, request: "VideoEditRequest") -> List[Image.Image]:
        with self.inference_lock:
            return self._process_single_task(
                input_video_path=request.input_video_path,
                input_mask_video_path=request.input_mask_video_path,
                original_frame_count=request.original_frame_count,
            )
