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
from typing import List, Optional
from dataclasses import dataclass


@dataclass
class VideoEditRequest:
    input_video_path: str
    input_mask_video_path: Optional[str] = None
    ref_images: Optional[List[str]] = None
    prompt: Optional[str] = None
    mask_idx: Optional[int] = None
    task_name: Optional[str] = None
    original_frame_count: Optional[int] = None
