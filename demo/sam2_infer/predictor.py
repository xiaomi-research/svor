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
import logging
from threading import Lock
import uuid
from typing import Any, Dict, Generator, List
import contextlib

import numpy as np
import torch
import sys
import os
from sam2.build_sam import build_sam2_video_predictor
from .data_types import (
    AddMaskRequest,
    AddPointsRequest,
    CancelPorpagateResponse,
    CancelPropagateInVideoRequest,
    ClearPointsInFrameRequest,
    ClearPointsInVideoRequest,
    ClearPointsInVideoResponse,
    CloseSessionRequest,
    CloseSessionResponse,
    Mask,
    PropagateDataResponse,
    PropagateDataValue,
    PropagateInVideoRequest,
    RemoveObjectRequest,
    RemoveObjectResponse,
    StartSessionRequest,
    StartSessionResponse,
)
from pycocotools.mask import decode as decode_masks, encode as encode_masks

current_file_path = os.path.abspath(__file__)
project_roots = [os.path.dirname(current_file_path)]
for project_root in project_roots:
    sys.path.insert(0, project_root) if project_root not in sys.path else None


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Sam2_Predictor:
    def __init__(self, sam2_model_path, config, device="cuda"):

        device = torch.device(device)
        logger.info(f"using device: {device}")

        self.device = device
        self.predictor = build_sam2_video_predictor(config, sam2_model_path, device=device)

        self.inference_lock = Lock()
        self.session_states: Dict[str, Any] = {}
        self.score_thresh = 0

    def autocast_context(self):
        if self.device.type == "cuda":
            return torch.autocast("cuda", dtype=torch.bfloat16)
        else:
            return contextlib.nullcontext()

    def start_session(self, request: StartSessionRequest) -> StartSessionResponse:
        with self.autocast_context(), self.inference_lock:
            session_id = str(uuid.uuid4())
            # for MPS devices, we offload the video frames to CPU by default to avoid
            # memory fragmentation in MPS (which sometimes crashes the entire process)
            offload_video_to_cpu = self.device.type == "mps"
            inference_state = self.predictor.init_state(
                request.path,
                offload_video_to_cpu=offload_video_to_cpu,
            )
            self.session_states[session_id] = {
                "canceled": False,
                "state": inference_state,
            }
            return StartSessionResponse(session_id=session_id)

    def close_session(self, request: CloseSessionRequest) -> CloseSessionResponse:
        is_successful = self.__clear_session_state(request.session_id)
        return CloseSessionResponse(success=is_successful)

    def add_points(self, request: AddPointsRequest, test: str = "") -> PropagateDataResponse:
        with self.autocast_context(), self.inference_lock:
            session = self.__get_session(request.session_id)
            inference_state = session["state"]

            frame_idx = request.frame_index
            obj_id = request.object_id
            points = request.points
            labels = request.labels
            clear_old_points = request.clear_old_points

            logger.info(
                f"add points on frame {frame_idx} in session {request.session_id}: {obj_id}, {points}, {labels}"
            )
            # add new prompts and instantly get the output on the same frame
            frame_idx, object_ids, masks = self.predictor.add_new_points_or_box(
                inference_state=inference_state,
                frame_idx=frame_idx,
                obj_id=obj_id,
                points=points,
                labels=labels,
                clear_old_points=clear_old_points,
                normalize_coords=False,
            )

            masks_binary = (masks > self.score_thresh)[:, 0].cpu().numpy()

            rle_mask_list = self.__get_rle_mask_list(object_ids=object_ids, masks=masks_binary)

            return PropagateDataResponse(
                frame_index=frame_idx,
                results=rle_mask_list,
            )

    def add_mask(self, request: AddMaskRequest) -> PropagateDataResponse:
        """
        Add new points on a specific video frame.
        - mask is a numpy array of shape [H_im, W_im] (containing 1 for foreground and 0 for background).
        Note: providing an input mask would overwrite any previous input points on this frame.
        """
        with self.autocast_context(), self.inference_lock:
            session_id = request.session_id
            frame_idx = request.frame_index
            obj_id = request.object_id
            rle_mask = {
                "counts": request.mask.counts,
                "size": request.mask.size,
            }

            mask = decode_masks(rle_mask)

            logger.info(f"add mask on frame {frame_idx} in session {session_id}: {obj_id}, {mask.shape}")
            session = self.__get_session(session_id)
            inference_state = session["state"]

            frame_idx, obj_ids, video_res_masks = self.predictor.add_new_mask(
                inference_state=inference_state,
                frame_idx=frame_idx,
                obj_id=obj_id,
                mask=torch.tensor(mask > 0),
            )
            masks_binary = (video_res_masks > self.score_thresh)[:, 0].cpu().numpy()

            rle_mask_list = self.__get_rle_mask_list(object_ids=obj_ids, masks=masks_binary)

            return PropagateDataResponse(
                frame_index=frame_idx,
                results=rle_mask_list,
            )

    def clear_points_in_frame(self, request: ClearPointsInFrameRequest) -> PropagateDataResponse:
        """
        Remove all input points in a specific frame.
        """
        with self.autocast_context(), self.inference_lock:
            session_id = request.session_id
            frame_idx = request.frame_index
            obj_id = request.object_id

            logger.info(f"clear inputs on frame {frame_idx} in session {session_id}: {obj_id=}")
            session = self.__get_session(session_id)
            inference_state = session["state"]
            frame_idx, obj_ids, video_res_masks = self.predictor.clear_all_prompts_in_frame(
                inference_state, frame_idx, obj_id
            )
            masks_binary = (video_res_masks > self.score_thresh)[:, 0].cpu().numpy()

            rle_mask_list = self.__get_rle_mask_list(object_ids=obj_ids, masks=masks_binary)

            return PropagateDataResponse(
                frame_index=frame_idx,
                results=rle_mask_list,
            )

    def clear_points_in_video(self, request: ClearPointsInVideoRequest) -> ClearPointsInVideoResponse:
        """
        Remove all input points in all frames throughout the video.
        """
        with self.autocast_context(), self.inference_lock:
            session_id = request.session_id
            logger.info(f"clear all inputs across the video in session {session_id}")
            session = self.__get_session(session_id)
            inference_state = session["state"]
            self.predictor.reset_state(inference_state)
            return ClearPointsInVideoResponse(success=True)

    def remove_object(self, request: RemoveObjectRequest) -> RemoveObjectResponse:
        """
        Remove an object id from the tracking state.
        """
        with self.autocast_context(), self.inference_lock:
            session_id = request.session_id
            obj_id = request.object_id
            logger.info(f"remove object in session {session_id}: {obj_id=}")
            session = self.__get_session(session_id)
            inference_state = session["state"]
            new_obj_ids, updated_frames = self.predictor.remove_object(inference_state, obj_id)

            results = []
            for frame_index, video_res_masks in updated_frames:
                masks = (video_res_masks > self.score_thresh)[:, 0].cpu().numpy()
                rle_mask_list = self.__get_rle_mask_list(object_ids=new_obj_ids, masks=masks)
                results.append(
                    PropagateDataResponse(
                        frame_index=frame_index,
                        results=rle_mask_list,
                    )
                )

            return RemoveObjectResponse(results=results)

    def propagate_in_video(self, request: PropagateInVideoRequest) -> Generator[PropagateDataResponse, None, None]:
        session_id = request.session_id
        start_frame_idx = request.start_frame_index
        propagation_direction = "both"
        max_frame_num_to_track = None

        """
        Propagate existing input points in all frames to track the object across video.
        """

        # Note that as this method is a generator, we also need to use autocast_context
        # in caller to this method to ensure that it's called under the correct context
        # (we've added `autocast_context` to `gen_track_with_mask_stream` in app.py).
        with self.autocast_context(), self.inference_lock:
            logger.info(
                f"propagate in video in session {session_id}: "
                f"{propagation_direction=}, {start_frame_idx=}, {max_frame_num_to_track=}"
            )

            try:
                session = self.__get_session(session_id)
                session["canceled"] = False

                inference_state = session["state"]
                if propagation_direction not in ["both", "forward", "backward"]:
                    raise ValueError(f"invalid propagation direction: {propagation_direction}")

                # First doing the forward propagation
                if propagation_direction in ["both", "forward"]:
                    for outputs in self.predictor.propagate_in_video(
                        inference_state=inference_state,
                        start_frame_idx=start_frame_idx,
                        max_frame_num_to_track=max_frame_num_to_track,
                        reverse=False,
                    ):
                        if session["canceled"]:
                            return None

                        frame_idx, obj_ids, video_res_masks = outputs
                        masks_binary = (video_res_masks > self.score_thresh)[:, 0].cpu().numpy()

                        rle_mask_list = self.__get_rle_mask_list(object_ids=obj_ids, masks=masks_binary)

                        yield PropagateDataResponse(
                            frame_index=frame_idx,
                            results=rle_mask_list,
                        )

                # Then doing the backward propagation (reverse in time)
                if propagation_direction in ["both", "backward"]:
                    for outputs in self.predictor.propagate_in_video(
                        inference_state=inference_state,
                        start_frame_idx=start_frame_idx,
                        max_frame_num_to_track=max_frame_num_to_track,
                        reverse=True,
                    ):
                        if session["canceled"]:
                            return None

                        frame_idx, obj_ids, video_res_masks = outputs
                        masks_binary = (video_res_masks > self.score_thresh)[:, 0].cpu().numpy()

                        rle_mask_list = self.__get_rle_mask_list(object_ids=obj_ids, masks=masks_binary)

                        yield PropagateDataResponse(
                            frame_index=frame_idx,
                            results=rle_mask_list,
                        )
            finally:
                # Log upon completion (so that e.g. we can see if two propagations happen in parallel).
                # Using `finally` here to log even when the tracking is aborted with GeneratorExit.
                logger.info(f"propagation ended in session {session_id}; {self.__get_session_stats()}")

    def cancel_propagate_in_video(self, request: CancelPropagateInVideoRequest) -> CancelPorpagateResponse:
        session = self.__get_session(request.session_id)
        session["canceled"] = True
        return CancelPorpagateResponse(success=True)

    def __get_rle_mask_list(self, object_ids: List[int], masks: np.ndarray) -> List[PropagateDataValue]:
        """
        Return a list of data values, i.e. list of object/mask combos.
        """
        return [
            self.__get_mask_for_object(object_id=object_id, mask=mask) for object_id, mask in zip(object_ids, masks)
        ]

    def __get_mask_for_object(self, object_id: int, mask: np.ndarray) -> PropagateDataValue:
        """
        Create a data value for an object/mask combo.
        """
        mask_rle = encode_masks(np.array(mask, dtype=np.uint8, order="F"))
        mask_rle["counts"] = mask_rle["counts"].decode()
        return PropagateDataValue(
            object_id=object_id,
            mask=Mask(
                size=mask_rle["size"],
                counts=mask_rle["counts"],
            ),
        )

    def __get_session(self, session_id: str):
        session = self.session_states.get(session_id, None)
        if session is None:
            raise RuntimeError(f"Cannot find session {session_id}; it might have expired")
        return session

    def __get_session_stats(self):
        """Get a statistics string for live sessions and their GPU usage."""
        # print both the session ids and their video frame numbers
        live_session_strs = [
            f"'{session_id}' ({session['state']['num_frames']} frames, {len(session['state']['obj_ids'])} objects)"
            for session_id, session in self.session_states.items()
        ]
        session_stats_str = (
            "Test String Here - -"
            f"live sessions: [{', '.join(live_session_strs)}], GPU memory: "
            f"{torch.cuda.memory_allocated() // 1024**2} MiB used and "
            f"{torch.cuda.memory_reserved() // 1024**2} MiB reserved"
            f" (max over time: {torch.cuda.max_memory_allocated() // 1024**2} MiB used "
            f"and {torch.cuda.max_memory_reserved() // 1024**2} MiB reserved)"
        )
        return session_stats_str

    def __clear_session_state(self, session_id: str) -> bool:
        session = self.session_states.pop(session_id, None)
        if session is None:
            logger.warning(
                f"cannot close session {session_id} as it does not exist (it might have expired); "
                f"{self.__get_session_stats()}"
            )
            return False
        else:
            logger.info(f"removed session {session_id}; {self.__get_session_stats()}")
            return True


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--sam2_model_path", type=str, required=True, help="Path to the SAM2 model file.")
    parser.add_argument("--config", type=str, required=True, help="Path to the configuration file.")
    parser.add_argument("test_order_file", type=str, help="Path to the test order file.")
    parser.add_argument("output_path", type=str, help="Path to save the output results.")
    args = parser.parse_args()

    predictor = Sam2_Predictor(sam2_model_path=args.sam2_model_path, config=args.config, device="cuda")

    import json

    with open(args.test_order_file, "r") as f:
        test_orders = json.load(f)
    session_id = None
    for order in test_orders:
        if order["action"] == "start_session":
            response = predictor.start_session(StartSessionRequest(**order["params"]))
            print(f"Started session: {response.session_id}")
        elif order["action"] == "add_points":
            response = predictor.add_points(AddPointsRequest(**order["params"]))
            print(f"Added points, got masks for frame {response.frame_index}")
        elif order["action"] == "close_session":
            response = predictor.close_session(CloseSessionRequest(**order["params"]))
            print(f"Closed session, success: {response.success}")
