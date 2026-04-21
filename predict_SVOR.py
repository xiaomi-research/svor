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
import argparse
import numpy as np
import torch
from diffusers import FlowMatchEulerDiscreteScheduler
from omegaconf import OmegaConf
from PIL import Image
from transformers import AutoTokenizer
import scipy
import cv2
from glob import glob
import torch.distributed as dist
from videox_fun.dist import set_multi_gpus_devices
from videox_fun.models import AutoencoderKLWan, WanT5EncoderModel, VaceWanModel
from videox_fun.pipeline import SVORPipeline
from videox_fun.utils.fp8_optimization import (
    convert_model_weight_to_float8,
    replace_parameters_by_name,
    convert_weight_dtype_wrapper,
)
from videox_fun.utils.lora_utils import merge_lora
from videox_fun.utils.utils import save_videos_grid, filter_kwargs


def load_patch_safetensors(path):
    list_tensors = glob(path + "/*.safetensors")
    all = {}
    for x in list_tensors:
        from safetensors.torch import load_file

        tmp = load_file(x)
        all.update(tmp)
    return all


def parse_args():
    parser = argparse.ArgumentParser(description="WanFun Video Editing Script")

    # GPU and memory configuration
    parser.add_argument(
        "--gpu_memory_mode",
        type=str,
        default="model_full_load",
        choices=["model_full_load", "model_cpu_offload", "model_cpu_offload_and_qfloat8", "sequential_cpu_offload"],
        help="GPU memory optimization mode",
    )
    parser.add_argument("--ulysses_degree", type=int, default=1, help="Ulysses degree for multi-GPU configuration")
    parser.add_argument("--ring_degree", type=int, default=1, help="Ring degree for multi-GPU configuration")

    # Model paths
    parser.add_argument(
        "--config_path", type=str, default="config/wan2.1/wan_civitai.yaml", help="Path to model configuration file"
    )
    parser.add_argument("--model_name", type=str, default="models/Wan2.1-VACE-1.3B", help="Path to pretrained model")

    # Generation parameters
    parser.add_argument("--sample_size", type=str, default="720,1280", help="Output size as 'height,width'")
    parser.add_argument("--video_length", type=int, default=81, help="Length of generated video in frames")
    parser.add_argument("--fps", type=int, default=16, help="Frames per second for output video")
    parser.add_argument(
        "--weight_dtype",
        type=str,
        default="bfloat16",
        choices=["float16", "bfloat16"],
        help="Data type for model weights",
    )

    # Prompt and generation settings
    parser.add_argument(
        "--prompt",
        type=str,
        default="Remove the target and fill the content appropriately",
        help="Text prompt for generation",
    )
    parser.add_argument(
        "--negative_prompt",
        type=str,
        default="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
        help="Negative text prompt",
    )
    parser.add_argument("--guidance_scale", type=float, default=6.0, help="Guidance scale for generation")
    parser.add_argument("--seed", type=int, default=43, help="Random seed for reproducibility")
    parser.add_argument("--context_scale", type=float, default=1.0, help="Context scale for vace control")
    parser.add_argument("--dilation", type=int, default=6, help="Dilation for inp mask (only for inpaint mode)")

    # Parameters for SVOR
    parser.add_argument(
        "--lora_path",
        type=str,
        default=["models/remove_model_stage1.safetensors", "models/remove_model_stage2.safetensors"],
        nargs="+",
        help="Optional path to LoRA checkpoint",
    )
    parser.add_argument(
        "--lora_weight", type=float, default=[1.0, 1.0], nargs="+", help="Weight for LoRA model if used"
    )
    parser.add_argument("--input_video", type=str, default=None, required=True, help="Path to input video for editing")
    parser.add_argument(
        "--input_mask_video", type=str, default=None, required=True, help="Path to mask video for editing"
    )
    parser.add_argument("--num_inference_steps", type=int, default=20, help="Number of inference steps")
    parser.add_argument("--save_dir", type=str, default="samples/SVOR", help="Directory to save generated videos")

    return parser.parse_args()


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

        frames = frames[:video_length]
        if len(frames) < video_length:
            frames += [frames[-1]] * (video_length - len(frames))

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
                mask = scipy.ndimage.binary_dilation(mask_np, iterations=dilation).astype(np.uint8) * 255
            mask_frames.append(mask)
        mask_cap.release()

        mask_frames = mask_frames[:video_length]
        if len(mask_frames) < video_length:
            mask_frames += [mask_frames[-1]] * (video_length - len(mask_frames))

        resized_masks = [Image.fromarray(mask).resize([sample_size[1], sample_size[0]]) for mask in mask_frames]

        input_video_mask = (
            torch.stack([torch.from_numpy(np.array(mask)) for mask in resized_masks]).unsqueeze(0).unsqueeze(0) / 255.0
        )  # [1, 1, T, H, W]

    else:
        input_video_mask = torch.ones((1, 1, video_length, sample_size[0], sample_size[1])).float()

    if input_video_path is not None and input_video is not None:
        input_video = input_video * (torch.tile(input_video_mask, [1, 3, 1, 1, 1]) < 0.5) + (128.0) * (
            torch.tile(input_video_mask, [1, 3, 1, 1, 1]) >= 0.5
        )
    input_video = input_video.div_(127.5).sub_(1.0)

    return input_video, input_video_mask


def process_single_task(
    pipeline,
    args,
    input_video_path,
    input_mask_video_path,
    prompt,
):
    """Process a single video editing task"""
    if input_video_path is not None:
        # Get video resolution
        cap = cv2.VideoCapture(input_video_path)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        # Calculate aspect ratio preserving size
        aspect_ratio = height / width
        max_area = int(args.sample_size.split(",")[0]) * int(args.sample_size.split(",")[1])
        new_height = round(np.sqrt(max_area * aspect_ratio))
        new_height = (new_height + 16 - 1) // 16 * 16
        new_width = round(np.sqrt(max_area / aspect_ratio))
        new_width = (new_width + 16 - 1) // 16 * 16
        sample_size = [new_height, new_width]
    else:
        sample_size = [int(args.sample_size.split(",")[0]), int(args.sample_size.split(",")[1])]

    generator = torch.Generator(device=pipeline.device).manual_seed(args.seed)

    with torch.no_grad():
        video_length = (
            int(
                (args.video_length - 1)
                // pipeline.vae.config.temporal_compression_ratio
                * pipeline.vae.config.temporal_compression_ratio
            )
            + 1
            if args.video_length != 1
            else 1
        )

        # Process video and mask
        (
            input_video,
            input_video_mask,
        ) = process_video(
            input_video_path,
            input_mask_video_path,
            video_length=video_length,
            sample_size=sample_size,
            dilation=args.dilation,
        )

        # Generate edited video
        sample = pipeline(
            prompt,
            negative_prompt=args.negative_prompt,
            height=sample_size[0],
            width=sample_size[1],
            generator=generator,
            guidance_scale=args.guidance_scale,
            num_inference_steps=args.num_inference_steps,
            video=input_video,
            mask_video=input_video_mask,
            context_scale=args.context_scale,
        ).videos

    return sample, video_length


def save_results(sample, args, video_length, fps, task_name=None):
    """Save the generated results"""
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir, exist_ok=True)

    prefix = task_name
    if video_length == 1:
        video_path = os.path.join(args.save_dir, prefix + ".png")
        image = sample[0, :, 0]
        image = image.transpose(0, 1).transpose(1, 2)
        image = (image * 255).numpy().astype(np.uint8)
        image = Image.fromarray(image)
        image.save(video_path)
    else:
        video_path = os.path.join(args.save_dir, prefix + ".mp4")
        save_videos_grid(sample, video_path, fps=fps)


def main():
    args = parse_args()

    # Validate arguments
    if args.input_video is None and args.input_mask_video is None:
        raise ValueError("Must provide either --input_video and --input_mask_video")

    # Convert weight dtype
    weight_dtype = torch.bfloat16 if args.weight_dtype == "bfloat16" else torch.float16
    device = set_multi_gpus_devices(args.ulysses_degree, args.ring_degree)
    config = OmegaConf.load(args.config_path)

    # Initialize transformer
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

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir, exist_ok=True)

    # Single task processing
    sample, video_length = process_single_task(pipeline, args, args.input_video, args.input_mask_video, args.prompt)

    video_basename = os.path.splitext(os.path.basename(args.input_video))[0]
    if not dist.is_initialized() or dist.get_rank() == 0:
        save_results(sample, args, video_length, args.fps, task_name=video_basename)


if __name__ == "__main__":
    main()
