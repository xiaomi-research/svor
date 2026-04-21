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
import cv2
import uuid
import shutil
import argparse
import numpy as np
import gradio as gr
from typing import List, Dict, Tuple
from threading import Lock
import time

try:
    from .sam2_infer.predictor import Sam2_Predictor
    from .sam2_infer.data_types import (
        AddPointsRequest,
        PropagateInVideoRequest,
        CloseSessionRequest,
        StartSessionRequest,
    )
except ImportError as e:
    print(f"Warning: Could not import Sam2_Predictor: {e}")
    Sam2_Predictor = None


try:
    from .remove_model_infer.predictor import SVORpredictor
    from .remove_model_infer.data_types import VideoEditRequest
except ImportError as e:
    print(f"Warning: Could not import SVORpredictor: {e}")
    SVORpredictor = None


# Utility functions
def preprocess_video(
    input_path: str, output_path: str, target_size: List[int] = [1280, 720], target_fps: int = 16, max_frames: int = 81
) -> str:
    """Preprocess video: adjust resolution, frame rate, and maximum frame count"""
    cap = cv2.VideoCapture(input_path)

    orig_fps = cap.get(cv2.CAP_PROP_FPS)
    orig_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    aspect_ratio = height / width
    max_area = target_size[0] * target_size[1]
    new_height = round(np.sqrt(max_area * aspect_ratio))
    new_height = (new_height + 16 - 1) // 16 * 16
    new_width = round(np.sqrt(max_area / aspect_ratio))
    new_width = (new_width + 16 - 1) // 16 * 16

    total_frames_needed = min(max_frames, int(orig_frame_count * target_fps / orig_fps))
    if total_frames_needed <= 0:
        total_frames_needed = 1
    frame_interval = max(1, orig_frame_count // total_frames_needed) if orig_frame_count > 0 else 1

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, target_fps, (new_width, new_height))

    frame_count = 0
    processed_frames = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_interval == 0:
            resized_frame = cv2.resize(frame, (new_width, new_height))
            out.write(resized_frame)
            processed_frames += 1

            if processed_frames >= max_frames:
                break

        frame_count += 1

    cap.release()
    out.release()

    return output_path


def extract_frames(video_path: str, output_dir: str) -> List[str]:
    """
    Extract frames from video
    """
    cap = cv2.VideoCapture(video_path)
    frames = []
    frame_count = 0

    os.makedirs(output_dir, exist_ok=True)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_path = os.path.join(output_dir, f"frame_{frame_count:04d}.jpg")
        cv2.imwrite(frame_path, frame)
        frames.append(frame_path)
        frame_count += 1

    cap.release()
    return frames


def create_mask_visualization(mask_data: Dict, frame_shape: Tuple[int, int]) -> np.ndarray:
    """
    Create visual mask from RLE mask data, supporting Mask objects and dict formats
    """
    try:
        from pycocotools.mask import decode as decode_rle

        # Decode RLE mask, supporting multiple formats
        mask_rle = None

        # Case 1: mask_data contains "mask" field (mask object or dict in PropagateDataValue)
        if isinstance(mask_data, dict) and "mask" in mask_data:
            mask_obj = mask_data["mask"]
            # If mask_obj is an object, convert to dict
            if hasattr(mask_obj, "__dict__"):
                mask_obj = mask_obj.__dict__
            if isinstance(mask_obj, dict):
                mask_rle = {"size": mask_obj.get("size"), "counts": mask_obj.get("counts")}

        # Case 2: Direct RLE dict or Mask object
        if mask_rle is None:
            if hasattr(mask_data, "__dict__"):
                # mask_data is a Mask object
                mask_data = mask_data.__dict__

            if isinstance(mask_data, dict):
                mask_rle = {"size": mask_data.get("size"), "counts": mask_data.get("counts")}

        # Validate mask_rle is valid
        if mask_rle is None or mask_rle.get("size") is None or mask_rle.get("counts") is None:
            print("Error: Invalid mask data structure")
            return np.zeros((*frame_shape[:2], 3), dtype=np.uint8)

        # counts may need to be converted to bytes
        if isinstance(mask_rle.get("counts"), str):
            try:
                mask_rle["counts"] = mask_rle["counts"].encode("utf-8")
            except Exception:
                pass

        mask = decode_rle(mask_rle)

        # Create colored mask visualization
        color_mask = np.zeros((*frame_shape[:2], 3), dtype=np.uint8)
        color_mask[mask == 1] = [255, 0, 0]  # Red mask

        return color_mask
    except Exception as e:
        print(f"Error creating mask visualization: {e}")
        return np.zeros((*frame_shape[:2], 3), dtype=np.uint8)


def overlay_masks_on_frame(frame_path: str, masks_data: List[Dict]) -> np.ndarray:
    """
    Overlay masks on frame
    """
    frame = cv2.imread(frame_path)
    if frame is None:
        return np.zeros((480, 640, 3), dtype=np.uint8)  # Return default-sized black image

    # Ensure frame is 3-channel BGR image
    if len(frame.shape) == 2:
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    elif frame.shape[2] == 4:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    elif frame.shape[2] != 3:
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

    frame_with_masks = frame.copy()

    # Create different colored masks for each object
    colors = [
        (0, 0, 255),  # Red (BGR)
        (0, 255, 0),  # Green
        (255, 0, 0),  # Blue
        (0, 255, 255),  # Yellow
        (255, 0, 255),  # Purple
        (255, 255, 0),  # Cyan
    ]

    for idx, mask_data in enumerate(masks_data):
        try:
            mask_vis = create_mask_visualization(mask_data, frame.shape)

            # Apply mask to frame
            mask_bool = mask_vis[:, :, 0] > 0
            if np.any(mask_bool):
                color = colors[idx % len(colors)]

                # Create colored mask layer
                colored_mask = np.zeros_like(frame, dtype=np.uint8)
                colored_mask[:] = color
                colored_mask = cv2.bitwise_and(colored_mask, colored_mask, mask=mask_vis[:, :, 0])

                # Overlay mask on original image (alpha 0.5)
                mask_alpha = 0.5
                frame_with_masks = cv2.addWeighted(frame_with_masks, 1.0, colored_mask, mask_alpha, 0)

                # Add border
                contours, _ = cv2.findContours(mask_vis[:, :, 0], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(frame_with_masks, contours, -1, color, 2)

        except Exception as e:
            print(f"Error overlaying mask: {e}")

    return frame_with_masks


# Gradio application class
class VideoEditApp:
    def __init__(
        self,
        sam2_model_path: str = None,
        sam2_config: str = None,
        remove_model_args: argparse.Namespace = None,
        device: str = "cuda",
        session_timeout: int = 1800,
    ):
        """
        Initialize application, directly loading SAM2 and removal model predictors

        Args:
            sam2_model_path: SAM2 model path
            sam2_config: SAM2 configuration file path
            remove_model_args: Removal model parameter object
            device: Computing device ('cuda' or 'cpu')
            session_timeout: Session timeout in seconds, default 1800 seconds (30 minutes)
        """
        self.device = device
        self.sam2_predictor = None
        self.remove_predictor = None
        self.inference_lock = Lock()
        self.sessions = {}  # Store session information
        self.session_timestamps = {}  # Store creation time for each session
        self.session_timeout = session_timeout  # Session timeout

        # Initialize SAM2 predictor
        if Sam2_Predictor is not None and sam2_model_path and sam2_config:
            try:
                self.sam2_predictor = Sam2_Predictor(sam2_model_path=sam2_model_path, config=sam2_config, device=device)
                print("[Info] SAM2 Predictor initialized successfully")
            except Exception as e:
                print(f"[ERROR] SAM2 Predictor initialization failed: {e}")
                raise Exception(e)

        # Initialize removal model predictor
        if SVORpredictor is not None and remove_model_args:
            try:
                self.remove_predictor = SVORpredictor(remove_model_args)
                print("[Info] Removal model Predictor initialized successfully")
            except Exception as e:
                print(f"[ERROR] Removal model Predictor initialization failed: {e}")
                raise Exception(e)

    def upload_video(self, video_file):
        """
        Process uploaded video, initialize local SAM2 session
        """
        if video_file is None:
            return None, [], "Please upload video file", {}

        if self.sam2_predictor is None:
            return None, [], "SAM2 Predictor not initialized", {}

        # Create temporary working directory
        session_id = str(uuid.uuid4())
        work_dir = f"./temp/{session_id}"
        os.makedirs(work_dir, exist_ok=True)

        # Save uploaded video
        input_video_path = os.path.join(work_dir, "input.mp4")

        # Correctly handle Gradio uploaded files
        if hasattr(video_file, "name"):
            # If it's a TemporaryFileWrapper object, use its name attribute
            shutil.copy(video_file.name, input_video_path)
        else:
            # Otherwise use the path directly
            shutil.copy(video_file, input_video_path)

        # Preprocess video
        processed_video_path = os.path.join(work_dir, "processed.mp4")
        preprocess_video(input_video_path, processed_video_path)

        # Extract frames
        frames_dir = os.path.join(work_dir, "frames")
        frame_paths = extract_frames(processed_video_path, frames_dir)

        # Initialize local SAM2 session
        try:
            with self.inference_lock:
                # Directly call SAM2 predictor's start_session method
                request = StartSessionRequest(type="start_session", path=processed_video_path)
                response = self.sam2_predictor.start_session(request)
                sam2_session_id = response.session_id

            self.sessions[session_id] = {
                "work_dir": work_dir,
                "video_path": processed_video_path,
                "frames": frame_paths,
                "sam2_session_id": sam2_session_id,
                "annotations": {},  # Store annotation information
                "propagated_masks": None,  # Store propagated masks
                "selected_points": {},  # Store user-selected points
                "object_ids": set(),
                "object_masks": {},
                "clear_next_annotation": False,
            }
            # Record session creation time
            self.session_timestamps[session_id] = time.time()

            return (
                frame_paths[0] if frame_paths else None,
                frame_paths,
                f"Video processing completed, {len(frame_paths)} frames",
                {"session_id": session_id},
            )
        except Exception as e:
            # Clean up created directory
            if os.path.exists(work_dir):
                shutil.rmtree(work_dir)
            return None, [], f"Error processing video: {str(e)}", {}

    def update_frame_display(self, session_state, frame_index, selected_objects=None):
        """
        Update displayed frame, support visualizing masks by selected object ids

        Args:
            session_state: Session state dictionary
            frame_index: Current frame index
            selected_objects: List of checked object IDs (from CheckboxGroup)

        Returns:
            Current frame image or masked overlay image (numpy array)
        """
        if not session_state or "session_id" not in session_state:
            return None

        session_id = session_state["session_id"]
        if session_id not in self.sessions:
            return None

        session = self.sessions[session_id]
        if frame_index < 0 or frame_index >= len(session["frames"]):
            return None

        frame_path = session["frames"][frame_index]
        frame_img = cv2.imread(frame_path)
        if frame_img is None:
            return None

        # Ensure image is BGR format
        if len(frame_img.shape) == 2:
            frame_img = cv2.cvtColor(frame_img, cv2.COLOR_GRAY2BGR)

        # If no propagation results, return original frame
        if not session.get("propagated_masks"):
            try:
                return cv2.cvtColor(frame_img, cv2.COLOR_BGR2RGB)
            except Exception:
                return frame_img

        # If there are selected objects, overlay their masks
        if selected_objects and isinstance(selected_objects, list) and len(selected_objects) > 0:
            # selected_objects is a string list (from CheckboxGroup)
            try:
                selected_ids = [int(x) for x in selected_objects]
            except (ValueError, TypeError):
                selected_ids = []

            if selected_ids:
                # Get all mask data for current frame
                masks_for_frame = session["propagated_masks"].get(frame_index, [])

                # Filter masks for selected objects
                filtered_masks = []
                for mask_data in masks_for_frame:
                    if isinstance(mask_data, dict) and "object_id" in mask_data:
                        if int(mask_data["object_id"]) in selected_ids:
                            filtered_masks.append(mask_data)

                # If there are filtered masks, overlay them
                if filtered_masks:
                    img_with_masks = overlay_masks_on_frame(frame_path, filtered_masks)
                    try:
                        return cv2.cvtColor(img_with_masks, cv2.COLOR_BGR2RGB)
                    except Exception:
                        return img_with_masks

        # If no objects selected, return original frame
        try:
            return cv2.cvtColor(frame_img, cv2.COLOR_BGR2RGB)
        except Exception:
            return frame_img

    def handle_point_selection(self, session_state, frame_index, img, current_label, evt: gr.SelectData):
        """
        Handle user click selection on image
        """
        # New version: evt may not have index (no click), and we get label from current_label via gr.Radio
        # This function's signature will be updated by the caller to pass current_label
        # Keep compatibility here
        if not session_state or "session_id" not in session_state:
            return img, "Please upload video first", session_state

        session_id = session_state["session_id"]
        if session_id not in self.sessions:
            return img, "Session does not exist", session_state

        session = self.sessions[session_id]

        # Get click coordinates
        if evt is None or evt.index is None:
            return img, "No click coordinates detected", session_state
        x, y = evt.index
        # Normalize
        x = x / img.shape[1]
        y = y / img.shape[0]

        # Initialize frame's point list
        if frame_index not in session["selected_points"]:
            session["selected_points"][frame_index] = []

        # If point is close to an existing point, delete it (as toggle)
        removed = False
        for i, p in enumerate(list(session["selected_points"][frame_index])):
            px, py = p["point"]
            if abs(px - x) * img.shape[1] <= 8 and abs(py - y) * img.shape[0] <= 8:
                session["selected_points"][frame_index].pop(i)
                removed = True
                break

        if not removed:
            # current_label comes from Radio choices like "Foreground(1)" or "Background(0)"
            label = 1
            try:
                if current_label is not None and "0" in str(current_label):
                    label = 0
            except Exception:
                pass

            session["selected_points"][frame_index].append({"point": [x, y], "label": label})

        # Draw points on image (enlarged size)
        if img is not None:
            # Ensure image is correct format
            if isinstance(img, np.ndarray):
                img_copy = img.copy()
            else:
                # If it's a PIL image, convert to numpy array
                img_copy = np.array(img)
                # If RGB format, convert to BGR
                if len(img_copy.shape) == 3 and img_copy.shape[2] == 3:
                    img_copy = cv2.cvtColor(img_copy, cv2.COLOR_RGB2BGR)

            # Draw all selected points
            for point_data in session["selected_points"][frame_index]:
                px, py = point_data["point"]
                px = int(px * img_copy.shape[1])
                py = int(py * img_copy.shape[0])
                lbl = point_data.get("label", 1)
                # Foreground points are red, background points are blue
                color = (0, 0, 255) if lbl == 1 else (255, 0, 0)
                # Adaptive radius
                radius = int(min(img_copy.shape[0], img_copy.shape[1]) * 0.02)
                cv2.circle(img_copy, (int(px), int(py)), radius, color, -1)
                cv2.circle(img_copy, (int(px), int(py)), radius, (0, 0, 0), 1)

            # Return updated image, whether adding or deleting point
            if removed:
                return img_copy, f"Deleted point ({x}, {y}) from frame {frame_index}", session_state
            else:
                return img_copy, f"Added point ({x}, {y}) to frame {frame_index}", session_state

        return img, "No click coordinates detected", session_state

    def annotate_frame(self, session_state, frame_index, object_id, clear_previous=False):
        """
        Add annotation points to selected frame using local SAM2 predictor
        """
        if not session_state or "session_id" not in session_state:
            return None, "Please upload video first", {}, gr.update(choices=[])

        session_id = session_state["session_id"]
        if session_id not in self.sessions:
            return None, "Session does not exist", {}, gr.update(choices=[])

        if self.sam2_predictor is None:
            return None, "SAM2 Predictor not initialized", session_state, gr.update(choices=[])

        session = self.sessions[session_id]
        if frame_index >= len(session["frames"]):
            return None, "Frame index out of range", session_state, gr.update(choices=[])

        frame_path = session["frames"][frame_index]

        # Check if there are selected points
        if frame_index not in session["selected_points"] or not session["selected_points"][frame_index]:
            return None, "Please click on image to add points", session_state, gr.update(choices=[])

        # Prepare point data
        points_list = []
        labels_list = []

        # Extract data from selected points
        for point_data in session["selected_points"][frame_index]:
            points_list.append(point_data["point"])
            labels_list.append(point_data["label"])

        if not points_list:
            return None, "Please click on image to add points", session_state, gr.update(choices=[])

        # Use local SAM2 predictor
        try:
            with self.inference_lock:
                request = AddPointsRequest(
                    type="add_points",
                    session_id=session["sam2_session_id"],
                    frame_index=frame_index,
                    object_id=int(object_id) if object_id is not None else 1,
                    points=[[float(p[0]), float(p[1])] for p in points_list],
                    labels=[int(x) for x in labels_list],
                    clear_old_points=bool(session.get("clear_next_annotation", False)) or bool(clear_previous),
                )

                result = self.sam2_predictor.add_points(request)
                result_dict = (
                    result.to_dict()
                    if hasattr(result, "to_dict")
                    else {"frame_index": result.frame_index, "results": result.results}
                )

            # Validate return data structure: PropagateDataResponse
            if not isinstance(result_dict, dict):
                return (
                    None,
                    f"API returned wrong data type: expected dict, got {type(result_dict)}",
                    session_state,
                    gr.update(choices=[]),
                )

            if "results" not in result_dict or not isinstance(result_dict.get("results"), list):
                return (
                    None,
                    "API returned missing 'results' field or wrong data format",
                    session_state,
                    gr.update(choices=[]),
                )

            # Save annotation information
            if frame_index not in session["annotations"]:
                session["annotations"][frame_index] = []

            session["annotations"][frame_index].append(
                {"points": points_list, "labels": labels_list, "result": result_dict}
            )

            # Process and save returned masks to session (classified by object_id)
            if result_dict.get("results"):
                # Ensure session has object_masks and object_ids structure
                if "object_masks" not in session:
                    session["object_masks"] = {}
                if "object_ids" not in session:
                    session["object_ids"] = set()

                # Convert results to standard dict format
                normalized_results = []
                for pdv in result_dict["results"]:
                    # Convert PropagateDataValue object to dict
                    if not isinstance(pdv, dict):
                        if hasattr(pdv, "__dict__"):
                            pdv = pdv.__dict__.copy()
                        else:
                            print(f"Warning: PropagateDataValue type error, expected dict, got {type(pdv)}")
                            continue
                    else:
                        pdv = pdv.copy()

                    oid = pdv.get("object_id")
                    mask_data = pdv.get("mask")

                    # Validate object_id type
                    if oid is None:
                        print("Warning: Missing object_id, skipping this mask")
                        continue

                    if not isinstance(oid, int):
                        try:
                            oid = int(oid)
                        except (ValueError, TypeError):
                            print(f"Warning: object_id type cannot be converted to int: {oid}")
                            continue

                    # Handle Mask object or dict
                    if hasattr(mask_data, "__dict__"):
                        # mask_data is a Mask object, convert to dict
                        mask_data = mask_data.__dict__.copy()
                        pdv["mask"] = mask_data
                    elif not isinstance(mask_data, dict):
                        print(f"Warning: mask type error, got {type(mask_data)}")
                        continue

                    if "size" not in mask_data or "counts" not in mask_data:
                        print(f"Warning: mask missing required fields, skipping object {oid}")
                        continue

                    # Save mask
                    try:
                        session["object_ids"].add(oid)
                        session["object_masks"].setdefault(oid, {})[frame_index] = mask_data
                    except Exception as e:
                        print(f"Error: Failed to save mask - {e}")

                    normalized_results.append(pdv)

                mask_overlay = overlay_masks_on_frame(frame_path, normalized_results)
            else:
                # If no mask results returned, return original frame
                mask_overlay = cv2.imread(frame_path)
                if mask_overlay is None:
                    mask_overlay = np.zeros((480, 640, 3), dtype=np.uint8)

            # Generate object_choices for CheckboxGroup
            object_choices = (
                [str(x) for x in sorted(list(session.get("object_ids", [])))] if session.get("object_ids") else []
            )

            # Before sending to Gradio, convert BGR to RGB (Gradio displays as RGB)
            try:
                mask_overlay_rgb = cv2.cvtColor(mask_overlay, cv2.COLOR_BGR2RGB)
            except Exception:
                mask_overlay_rgb = mask_overlay

            # Reset clear_next_annotation flag after successful annotation
            session["clear_next_annotation"] = False

            return (
                mask_overlay_rgb,
                f"Annotation completed, frame {frame_index}, detected objects: {object_choices}",
                session_state,
                gr.update(choices=object_choices),
            )
        except Exception as e:
            return None, f"Error during annotation: {str(e)}", session_state, gr.update(choices=[])

    def propagate_masks(self, session_state):
        """
        Propagate masks to entire video using local SAM2 predictor
        """
        if not session_state or "session_id" not in session_state:
            return None, "Please upload video first", {}, None, []

        session_id = session_state["session_id"]
        if session_id not in self.sessions:
            return None, "Session does not exist", {}, None, []

        if self.sam2_predictor is None:
            return None, "SAM2 Predictor not initialized", session_state, None, []

        session = self.sessions[session_id]

        try:
            # Use local SAM2 predictor for propagation
            with self.inference_lock:
                request = PropagateInVideoRequest(
                    type="propagate", session_id=session["sam2_session_id"], start_frame_index=0
                )

                # Call propagate_in_video method, return generator
                propagated_masks = {}
                error_count = 0
                frame_count = 0

                for response in self.sam2_predictor.propagate_in_video(request):
                    frame_count += 1
                    try:
                        # response is PropagateDataResponse object
                        frame_idx = response.frame_index
                        results = response.results

                        if frame_idx is None or not isinstance(frame_idx, int):
                            print(f"Warning (frame {frame_count}): frame_index invalid")
                            error_count += 1
                            continue

                        if not isinstance(results, list):
                            print(f"Warning (frame {frame_count}): results is not a list")
                            error_count += 1
                            continue

                        # Validate each PropagateDataValue
                        validated_results = []
                        for pdv in results:
                            # Convert object to dict
                            if not isinstance(pdv, dict):
                                if hasattr(pdv, "__dict__"):
                                    pdv_dict = pdv.__dict__.copy()
                                else:
                                    print(f"Warning (frame {frame_idx}): PropagateDataValue type error")
                                    error_count += 1
                                    continue
                            else:
                                pdv_dict = pdv.copy()

                            oid = pdv_dict.get("object_id")
                            mask_data = pdv_dict.get("mask")

                            # Validate object_id
                            if oid is None or not isinstance(oid, int):
                                print(f"Warning (frame {frame_idx}): object_id invalid: {oid}")
                                error_count += 1
                                continue

                            # Handle Mask object or dict
                            if hasattr(mask_data, "__dict__"):
                                # mask_data is a Mask object, convert to dict
                                mask_data = mask_data.__dict__.copy()
                                pdv_dict["mask"] = mask_data
                            elif not isinstance(mask_data, dict):
                                print(
                                    f"Warning (frame {frame_idx}, object {oid}): mask type error, got {type(mask_data)}"
                                )
                                error_count += 1
                                continue

                            if "size" not in mask_data or "counts" not in mask_data:
                                print(f"Warning (frame {frame_idx}, object {oid}): mask missing required fields")
                                error_count += 1
                                continue

                            # Validate counts is string (RLE encoded)
                            counts = mask_data.get("counts")
                            if not isinstance(counts, str):
                                print(
                                    f"Warning (frame {frame_idx}, object {oid}): counts should be string, got {type(counts)}"
                                )
                                error_count += 1
                                continue

                            validated_results.append(pdv_dict)

                        # Save validated results
                        propagated_masks[frame_idx] = validated_results

                    except Exception as e:
                        print(f"Error (frame {frame_count}): Failed to process response - {e}")
                        error_count += 1
                        continue

            # Save propagation results
            session["propagated_masks"] = propagated_masks

            # Build mask dictionary stored by object_id for later visualization and removal by object
            session.setdefault("object_masks", {})
            session.setdefault("object_ids", set())

            for fidx, plist in propagated_masks.items():
                for pdv in plist:
                    try:
                        oid = int(pdv.get("object_id", -1)) if isinstance(pdv, dict) else pdv.object_id
                        mask_val = pdv.get("mask") if isinstance(pdv, dict) else pdv.mask
                        if oid >= 0:
                            session["object_ids"].add(oid)
                            session["object_masks"].setdefault(oid, {})[fidx] = mask_val
                    except Exception as e:
                        print(f"Failed to build object_masks: {e}")

            # Generate object_choices for CheckboxGroup
            object_choices = [str(x) for x in sorted(list(session.get("object_ids", [])))]

            # Create first frame visualization with mask
            if propagated_masks:
                first_frame_idx = sorted(propagated_masks.keys())[0]
                first_frame_path = session["frames"][first_frame_idx]
                mask_overlay = overlay_masks_on_frame(first_frame_path, propagated_masks[first_frame_idx])

                # Generate overlay video for complete visualization and download
                try:
                    overlay_video_path = self.create_overlay_video(session)
                except Exception as e:
                    overlay_video_path = None
                    print(f"Failed to create overlay video: {e}")

                # Convert to RGB for Gradio display
                try:
                    mask_overlay_rgb = cv2.cvtColor(mask_overlay, cv2.COLOR_BGR2RGB)
                except Exception:
                    mask_overlay_rgb = mask_overlay

                summary_msg = f"Mask propagation completed, processed {len(propagated_masks)} frames, detected {len(object_choices)} objects"
                if error_count > 0:
                    summary_msg += f" ({error_count} warnings)"

                return (
                    mask_overlay_rgb,
                    summary_msg,
                    session_state,
                    overlay_video_path,
                    gr.update(choices=object_choices),
                )

            return None, "No propagation results received", session_state, None, gr.update(choices=[])

        except Exception as e:
            return None, f"Error propagating masks: {str(e)}", session_state, None, gr.update(choices=[])

    def create_mask_video(self, session, selected_object_ids: List[int] = None):
        """
        Create mask video based on propagated masks. Generate masks only for objects in selected_object_ids (if None, include all objects).
        """
        # Create mask video file path
        mask_video_path = os.path.join(session["work_dir"], "mask_video.mp4")

        # Get video information
        cap = cv2.VideoCapture(session["video_path"])
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(mask_video_path, fourcc, fps, (width, height), isColor=False)

        # Create mask for each frame
        for i in range(len(session["frames"])):
            # Create blank mask
            mask_frame = np.zeros((height, width), dtype=np.uint8)

            # If frame has propagated masks, draw mask
            if session["propagated_masks"] and i in session["propagated_masks"]:
                masks_for_frame = session["propagated_masks"][i]
                # masks_for_frame expected to be a list, each item is a dict with object_id and mask
                for mask_data in masks_for_frame:
                    try:
                        from pycocotools.mask import decode as decode_rle

                        # Ensure mask_data is a dict
                        if not isinstance(mask_data, dict):
                            if hasattr(mask_data, "__dict__"):
                                mask_data = mask_data.__dict__
                            else:
                                continue

                        # Support encapsulation containing object_id
                        obj_id = mask_data.get("object_id")

                        # Filter non-selected objects (if selected_object_ids is provided)
                        if selected_object_ids is not None and obj_id is not None and obj_id not in selected_object_ids:
                            continue

                        # Extract RLE data
                        mask_obj = mask_data.get("mask")

                        # If mask is an object, convert to dict
                        if hasattr(mask_obj, "__dict__"):
                            mask_obj = mask_obj.__dict__

                        if isinstance(mask_obj, dict):
                            rle = {"size": mask_obj.get("size"), "counts": mask_obj.get("counts")}
                        else:
                            # Not a valid mask object, skip
                            continue

                        # counts may be string, pycocotools needs bytes (for compressed RLE)
                        if isinstance(rle.get("counts"), str):
                            try:
                                rle["counts"] = rle["counts"].encode("utf-8")
                            except Exception:
                                pass

                        # Skip if counts is empty or size is missing
                        if rle.get("counts") is None or rle.get("size") is None:
                            continue

                        mask = decode_rle(rle)
                        mask_frame = np.maximum(mask_frame, mask.astype(np.uint8) * 255)
                    except Exception as e:
                        print(f"Error decoding mask for frame {i}: {e}")

            out.write(mask_frame)

        out.release()
        return mask_video_path

    def create_overlay_video(self, session, alpha: float = 0.6):
        """
        Create visualization video with colored semi-transparent masks on original video frames based on propagated masks, return video path
        """
        if not session.get("propagated_masks"):
            return None

        overlay_video_path = os.path.join(session["work_dir"], "overlay_mask_video.mp4")

        # Get video information
        cap = cv2.VideoCapture(session["video_path"])
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(overlay_video_path, fourcc, fps, (width, height), isColor=True)

        for i, frame_path in enumerate(session["frames"]):
            # Read original frame
            frame = cv2.imread(frame_path)
            if frame is None:
                frame = np.zeros((height, width, 3), dtype=np.uint8)

            masks_for_frame = session.get("propagated_masks", {}).get(i, [])
            if masks_for_frame:
                try:
                    overlay = overlay_masks_on_frame(frame_path, masks_for_frame)
                    # overlay_masks_on_frame returns BGR image, same size as original frame
                    blended = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
                except Exception:
                    blended = frame
            else:
                blended = frame

            out.write(blended)

        out.release()
        return overlay_video_path

    def clear_current_frame_points(self, session_state, frame_index, selected_objects=None):
        """
        Clear selected points for current frame and set flag for backend to clear old points on next annotation.
        """
        if not session_state or "session_id" not in session_state:
            return session_state, "Please upload video first", None
        sid = session_state["session_id"]
        if sid not in self.sessions:
            return session_state, "Session does not exist", None
        session = self.sessions[sid]
        try:
            session.setdefault("selected_points", {})[frame_index] = []
            session["clear_next_annotation"] = True
            # Update frame_display to show frame with cleared points
            frame_display_img = self.update_frame_display(session_state, frame_index, selected_objects)
            return (
                session_state,
                f"Cleared points for frame {frame_index}, will clear old points on next annotation",
                frame_display_img,
            )
        except Exception as e:
            return session_state, f"Error clearing points: {e}", None

    def close_session(self, session_state):
        """
        Close local SAM2 session and release local cache files and session objects.
        """
        if not session_state or "session_id" not in session_state:
            return session_state, "No active session"
        sid = session_state["session_id"]
        if sid not in self.sessions:
            return {}, "Session no longer exists"

        session = self.sessions[sid]

        # First try to notify local SAM2 predictor to close session
        if self.sam2_predictor is not None:
            try:
                with self.inference_lock:
                    request = CloseSessionRequest(type="close_session", session_id=session.get("sam2_session_id"))
                    self.sam2_predictor.close_session(request)
            except Exception as e:
                print(f"Error closing SAM2 session (non-fatal): {e}")

        # Delete local temporary directory
        try:
            work_dir = session.get("work_dir")
            if work_dir and os.path.exists(work_dir):
                shutil.rmtree(work_dir)
        except Exception as e:
            print(f"failed to remove work_dir: {e}")

        # Remove from session dictionary
        try:
            del self.sessions[sid]
            # Also remove timestamp record
            if sid in self.session_timestamps:
                del self.session_timestamps[sid]
        except Exception:
            pass

        return {}, "Session closed and resources released"

    def cleanup_expired_sessions(self):
        """
        Clean up all expired sessions
        Returns number of cleaned sessions
        """
        current_time = time.time()
        expired_session_ids = []

        # Find all expired sessions
        for sid, create_time in list(self.session_timestamps.items()):
            if current_time - create_time > self.session_timeout:
                expired_session_ids.append(sid)

        # Clean up expired sessions
        cleanup_count = 0
        for sid in expired_session_ids:
            try:
                if sid in self.sessions:
                    session = self.sessions[sid]

                    # Close SAM2 session
                    if self.sam2_predictor is not None:
                        try:
                            with self.inference_lock:
                                request = CloseSessionRequest(
                                    type="close_session", session_id=session.get("sam2_session_id")
                                )
                                self.sam2_predictor.close_session(request)
                        except Exception as e:
                            print(f"[Cleanup] Error closing SAM2 session {sid}: {e}")

                    # Delete temporary directory
                    try:
                        work_dir = session.get("work_dir")
                        if work_dir and os.path.exists(work_dir):
                            shutil.rmtree(work_dir)
                    except Exception as e:
                        print(f"[Cleanup] Error deleting directory {sid}: {e}")

                    # Remove from session dictionary
                    del self.sessions[sid]
                    del self.session_timestamps[sid]
                    cleanup_count += 1
                    print(
                        f"[Cleanup] Session {sid} has timed out and been cleaned (created at {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(create_time))})"
                    )
            except Exception as e:
                print(f"[Cleanup Error] Error cleaning session {sid}: {e}")

        return cleanup_count

    def _add_object_to_session(self, session_state, oid: int):
        """Add an object id to the session and return updated session_state and choices list."""
        if not session_state or "session_id" not in session_state:
            return session_state, []
        sid = session_state["session_id"]
        if sid not in self.sessions:
            return session_state, []
        session = self.sessions[sid]
        if "object_ids" not in session:
            session["object_ids"] = set()
        try:
            session["object_ids"].add(int(oid))
        except Exception:
            pass
        choices = [str(x) for x in sorted(list(session.get("object_ids", [])))]
        return session_state, choices

    def remove_objects(self, session_state, selected_objects=None):
        """
        Preview removal video, display input video and mask
        Parameters:
            session_state: Session state
            selected_objects: List of checked object IDs (from CheckboxGroup)
        Returns: (input video, mask video, status info, session_state)
        """
        if not session_state or "session_id" not in session_state:
            return None, None, "Please upload video first", {}

        session_id = session_state["session_id"]
        if session_id not in self.sessions:
            return None, None, "Session does not exist", {}

        session = self.sessions[session_id]

        # Check if there are propagated masks
        if not session.get("propagated_masks"):
            return None, None, "Please propagate masks first", session_state

        try:
            # Convert selected objects to integer list
            selected_object_ids = None
            if selected_objects and isinstance(selected_objects, list) and len(selected_objects) > 0:
                try:
                    selected_object_ids = [int(x) for x in selected_objects]
                except (ValueError, TypeError):
                    selected_object_ids = None

            # If no objects selected, prompt user
            if selected_object_ids is None or len(selected_object_ids) == 0:
                return None, None, "Please check objects in display list to preview/remove", session_state

            # Create mask video, only containing selected objects
            mask_video_path = self.create_mask_video(session, selected_object_ids)
            input_video_path = session["video_path"]

            # Save selected object IDs to session for execute_removal use
            session["selected_object_ids_for_removal"] = selected_object_ids

            # Return preview mode results (only show input video and mask)
            return (
                input_video_path,
                mask_video_path,
                "Preview ready, mask for selected objects generated. Click 'Start removal' to begin processing",
                session_state,
            )

        except Exception as e:
            return None, None, f"Error during preview: {str(e)}", session_state

    def execute_removal(self, session_state):
        """
        Execute actual object removal operation
        Parameters:
            session_state: Session state
        Returns: (removal result video, input video, mask video, status info, session_state)
        """
        if not session_state or "session_id" not in session_state:
            return None, None, None, "Please upload video first", {}

        session_id = session_state["session_id"]
        if session_id not in self.sessions:
            return None, None, None, "Session does not exist", {}

        if self.remove_predictor is None:
            return None, None, None, "Removal model Predictor not initialized", session_state

        session = self.sessions[session_id]

        # Check if there are pre-saved object IDs
        selected_object_ids = session.get("selected_object_ids_for_removal")
        if not selected_object_ids:
            return None, None, None, "Please preview objects before executing removal", session_state

        try:
            # Create mask video, only containing selected objects
            mask_video_path = self.create_mask_video(session, selected_object_ids)
            input_video_path = session["video_path"]

            # Get original video frame count
            original_frame_count = len(session["frames"])

            # Use local removal model predictor
            with self.inference_lock:
                request = VideoEditRequest(
                    input_video_path=input_video_path,
                    input_mask_video_path=mask_video_path,
                    original_frame_count=original_frame_count,
                )

                result_frames = self.remove_predictor.predict(request)

            if result_frames and len(result_frames) > 0:
                # Generate output video file
                out_video_path = os.path.join(session["work_dir"], "removed_result.mp4")

                # Get dimensions of first frame
                first_frame = np.array(result_frames[0])
                h, w = first_frame.shape[:2]

                # Convert to BGR format for cv2
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                out = cv2.VideoWriter(out_video_path, fourcc, 16.0, (w, h))

                for frm in result_frames:
                    frame_array = np.array(frm)
                    # If RGB format, convert to BGR
                    if len(frame_array.shape) == 3 and frame_array.shape[2] == 3:
                        frame_array = cv2.cvtColor(frame_array, cv2.COLOR_RGB2BGR)
                    out.write(frame_array)
                out.release()

                return (
                    out_video_path,
                    input_video_path,
                    mask_video_path,
                    f"Object removal completed, {len(result_frames)} frames",
                    session_state,
                )

            return (
                None,
                input_video_path,
                mask_video_path,
                "Removal succeeded but no frame data returned",
                session_state,
            )
        except Exception as e:
            return None, None, None, f"Error removing objects: {str(e)}", session_state


# Create Gradio interface
def create_ui(sam2_model_path: str = None, sam2_config: str = None, remove_model_args: argparse.Namespace = None):
    app = VideoEditApp(
        sam2_model_path=sam2_model_path, sam2_config=sam2_config, remove_model_args=remove_model_args, device="cuda"
    )

    with gr.Blocks(title="SVOR Video Removal Experience") as demo:
        session_state = gr.State({})

        gr.Markdown("# SVOR Video Removal Experience")
        gr.Markdown("Upload video, annotate objects on frames, propagate masks, confirm effect, then remove objects")

        with gr.Tab("1. Upload and Preprocess"):
            with gr.Row():
                with gr.Column():
                    video_input = gr.File(label="Upload video", file_types=[".mp4", ".avi", ".mov"])
                    upload_btn = gr.Button("Process video")

                with gr.Column():
                    status_output = gr.Textbox(label="Status", interactive=False)

            # Display first frame
            first_frame = gr.Image(label="First frame", interactive=False)

        with gr.Tab("2. Annotate and Propagate"):
            # Top: Status information
            with gr.Row():
                annotation_status = gr.Textbox(label="Status", interactive=False)

            # Current frame display (full width, increased height)
            with gr.Row():
                frame_display = gr.Image(
                    label="Current frame (click to add points)", type="numpy", height=400, interactive=True
                )

            # Middle: Control panel (left) and results (right)
            with gr.Row():
                with gr.Column(scale=1):
                    frame_slider = gr.Slider(minimum=0, maximum=100, step=1, label="Select frame")
                    with gr.Row():
                        current_label = gr.Radio(
                            choices=["Foreground(1)", "Background(0)"],
                            value="Foreground(1)",
                            label="Current point label",
                            info="Label used when clicking",
                        )
                        clear_points_btn = gr.Button("Clear points for current frame")
                    with gr.Row():
                        object_id_input = gr.Number(label="Object ID (for this annotation)", value=1, precision=0)
                        add_object_btn = gr.Button("Add object ID")

                    # Object selector for visualizing multiple object masks
                    object_selector = gr.CheckboxGroup(
                        choices=[], label="Display object list (check to visualize and remove)", interactive=True
                    )

                    with gr.Row():
                        annotate_btn = gr.Button("Add annotation")
                        propagate_btn = gr.Button("Propagate masks")

                with gr.Column(scale=1):
                    annotated_frame = gr.Image(label="Annotation/Propagation result", interactive=False, type="numpy")
                    overlay_video = gr.Video(label="Mask overlay video (downloadable)", interactive=False)

        with gr.Tab("3. Remove Objects"):
            # First row: Status information
            with gr.Row():
                removal_status = gr.Textbox(label="Removal status", interactive=False)

            # Second row: Input video and mask video
            with gr.Row():
                input_video_display = gr.Video(label="Input video", interactive=False, scale=1)
                mask_video_display = gr.Video(label="Mask video", interactive=False, scale=1)

            # Third row: Removal result
            with gr.Row():
                removal_video = gr.Video(label="Removal result video", interactive=False, scale=1)

            # Fourth row: Buttons
            with gr.Row():
                preview_btn = gr.Button("Preview video", size="lg")
                remove_btn = gr.Button("Start removal", size="lg")

        # Event handling
        def upload_and_update_slider(video_file):
            first_frame, frame_paths, status, session_state = app.upload_video(video_file)
            max_frames = len(frame_paths) - 1 if frame_paths else 0
            # Also update frame_display to show first frame
            frame_display_img = None
            if session_state and "session_id" in session_state:
                frame_display_img = app.update_frame_display(session_state, 0, [])
            return (
                first_frame,
                gr.update(maximum=max_frames if max_frames > 0 else 100),
                status,
                session_state,
                frame_display_img,
            )

        upload_btn.click(
            fn=upload_and_update_slider,
            inputs=[video_input],
            outputs=[first_frame, frame_slider, status_output, session_state, frame_display],
        )

        # When sliding frame or changing object_selector, update display (consider currently selected objects)
        # frame_slider changes update frame_display
        frame_slider.change(
            fn=app.update_frame_display, inputs=[session_state, frame_slider, object_selector], outputs=[frame_display]
        )

        # object_selector changes also update frame_display (current frame visualization)
        object_selector.change(
            fn=app.update_frame_display, inputs=[session_state, frame_slider, object_selector], outputs=[frame_display]
        )

        # Handle image click event
        frame_display.select(
            fn=app.handle_point_selection,
            inputs=[session_state, frame_slider, frame_display, current_label],
            outputs=[frame_display, annotation_status, session_state],
        )

        # Add annotation button click event - update annotated_frame and object_selector
        annotate_btn.click(
            fn=app.annotate_frame,
            inputs=[session_state, frame_slider, object_id_input, gr.State(False)],
            outputs=[annotated_frame, annotation_status, session_state, object_selector],
        )

        # Add object ID button click event - update object_selector choices
        def update_object_selector_choices(sess, oid):
            """Helper to update CheckboxGroup choices via gr.update()"""
            sess_updated, choices = app._add_object_to_session(sess, int(oid))
            return sess_updated, gr.update(choices=choices)

        add_object_btn.click(
            fn=update_object_selector_choices,
            inputs=[session_state, object_id_input],
            outputs=[session_state, object_selector],
        )

        # Propagate button click event - return annotated_frame and object_selector
        propagate_btn.click(
            fn=app.propagate_masks,
            inputs=[session_state],
            outputs=[annotated_frame, annotation_status, session_state, overlay_video, object_selector],
        )

        # Clear current frame points button
        clear_points_btn.click(
            fn=app.clear_current_frame_points,
            inputs=[session_state, frame_slider, object_selector],
            outputs=[session_state, annotation_status, frame_display],
        )

        # Preview video button click event
        preview_btn.click(
            fn=app.remove_objects,
            inputs=[session_state, object_selector],
            outputs=[input_video_display, mask_video_display, removal_status, session_state],
        )

        # Start removal button click event
        remove_btn.click(
            fn=app.execute_removal,
            inputs=[session_state],
            outputs=[removal_video, input_video_display, mask_video_display, removal_status, session_state],
        )

        # Release SAM2 resources when video_input changes
        def on_video_input_change(video_file, session_state):
            """When new video is uploaded, release old session resources and clean up all expired sessions"""
            # First clean up all expired sessions
            cleanup_count = app.cleanup_expired_sessions()
            cleanup_msg = (
                f"Cleaned up {cleanup_count} expired sessions"
                if cleanup_count > 0
                else "No expired sessions to clean up"
            )

            # Then close current active session
            if session_state and "session_id" in session_state:
                app.close_session(session_state)
                return session_state, f"New video uploaded, old resources released. {cleanup_msg}"

            return session_state, cleanup_msg

        video_input.change(
            fn=on_video_input_change, inputs=[video_input, session_state], outputs=[session_state, removal_status]
        )

    return demo


def parse_args():
    parser = argparse.ArgumentParser(description="SVOR Video Removal Experience")

    # SAM2 related parameters
    parser.add_argument("--sam2_model_path", type=str, default="./models/sam2.1_hiera_large.pt", help="SAM2 model path")
    parser.add_argument(
        "--sam2_config",
        type=str,
        default="configs/sam2.1/sam2.1_hiera_l",
        help="SAM2 configuration file or config name",
    )

    # Removal model related parameters
    parser.add_argument("--model_name", type=str, default="./models/Wan2.1-VACE-1.3B", help="Removal model path")
    parser.add_argument(
        "--config_path",
        type=str,
        default="./config/wan2.1/wan_civitai.yaml",
        help="Removal model configuration file path",
    )
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
    # GPU and memory related parameters
    parser.add_argument(
        "--gpu_memory_mode",
        type=str,
        default="sequential_cpu_offload",
        choices=["model_full_load", "model_cpu_offload", "model_cpu_offload_and_qfloat8", "sequential_cpu_offload"],
        help="GPU memory optimization mode",
    )
    parser.add_argument("--ulysses_degree", type=int, default=1, help="Ulysses degree")
    parser.add_argument("--ring_degree", type=int, default=1, help="Ring degree")
    parser.add_argument(
        "--weight_dtype", type=str, default="bfloat16", choices=["bfloat16", "float16"], help="Weight data type"
    )

    # Inference parameters
    parser.add_argument("--sample_size", type=str, default="720,480", help="Sample size")
    parser.add_argument("--video_length", type=int, default=81, help="Video length")
    parser.add_argument("--num_inference_steps", type=int, default=20, help="Number of inference steps")
    parser.add_argument("--guidance_scale", type=float, default=6.0, help="Guidance scale")
    parser.add_argument("--context_scale", type=float, default=1.0, help="Context scale")
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
        help="Negative prompt",
    )
    parser.add_argument("--seed", type=int, default=43, help="Random seed")
    parser.add_argument("--dilation", type=int, default=0, help="Mask dilation")

    # Gradio related parameters
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Gradio service listening address")
    parser.add_argument("--port", type=int, default=7861, help="Gradio service port")
    parser.add_argument("--device", type=str, default="cuda", help="Computing device")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Create removal model parameter object
    remove_model_args = argparse.Namespace(
        model_name=args.model_name,
        config_path=args.config_path,
        lora_path=args.lora_path,
        lora_weight=args.lora_weight,
        gpu_memory_mode=args.gpu_memory_mode,
        ulysses_degree=args.ulysses_degree,
        ring_degree=args.ring_degree,
        weight_dtype=args.weight_dtype,
        sample_size=args.sample_size,
        video_length=args.video_length,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        context_scale=args.context_scale,
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        seed=args.seed,
        dilation=args.dilation,
    )

    demo = create_ui(
        sam2_model_path=args.sam2_model_path, sam2_config=args.sam2_config, remove_model_args=remove_model_args
    )
    demo.launch(server_name=args.host, server_port=args.port, share=False)
