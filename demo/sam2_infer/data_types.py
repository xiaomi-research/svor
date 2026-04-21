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
from dataclasses import dataclass
from typing import Dict, List, Optional, Union

from dataclasses_json import dataclass_json
from torch import Tensor


@dataclass_json
@dataclass
class Mask:
    size: List[int]
    counts: str


@dataclass_json
@dataclass
class BaseRequest:
    type: str


@dataclass_json
@dataclass
class StartSessionRequest(BaseRequest):
    type: str
    path: str
    session_id: Optional[str] = None


@dataclass_json
@dataclass
class SaveSessionRequest(BaseRequest):
    type: str
    session_id: str


@dataclass_json
@dataclass
class LoadSessionRequest(BaseRequest):
    type: str
    session_id: str


@dataclass_json
@dataclass
class RenewSessionRequest(BaseRequest):
    type: str
    session_id: str


@dataclass_json
@dataclass
class CloseSessionRequest(BaseRequest):
    type: str
    session_id: str


@dataclass_json
@dataclass
class AddPointsRequest(BaseRequest):
    type: str
    session_id: str
    frame_index: int
    clear_old_points: bool
    object_id: int
    labels: List[int]
    points: List[List[float]]


@dataclass_json
@dataclass
class AddMaskRequest(BaseRequest):
    type: str
    session_id: str
    frame_index: int
    object_id: int
    mask: Mask


@dataclass_json
@dataclass
class ClearPointsInFrameRequest(BaseRequest):
    type: str
    session_id: str
    frame_index: int
    object_id: int


@dataclass_json
@dataclass
class ClearPointsInVideoRequest(BaseRequest):
    type: str
    session_id: str


@dataclass_json
@dataclass
class RemoveObjectRequest(BaseRequest):
    type: str
    session_id: str
    object_id: int


@dataclass_json
@dataclass
class PropagateInVideoRequest(BaseRequest):
    type: str
    session_id: str
    start_frame_index: int


@dataclass_json
@dataclass
class CancelPropagateInVideoRequest(BaseRequest):
    type: str
    session_id: str


@dataclass_json
@dataclass
class StartSessionResponse:
    session_id: str


@dataclass_json
@dataclass
class SaveSessionResponse:
    session_id: str


@dataclass_json
@dataclass
class LoadSessionResponse:
    session_id: str


@dataclass_json
@dataclass
class RenewSessionResponse:
    session_id: str


@dataclass_json
@dataclass
class CloseSessionResponse:
    success: bool


@dataclass_json
@dataclass
class ClearPointsInVideoResponse:
    success: bool


@dataclass_json
@dataclass
class PropagateDataValue:
    object_id: int
    mask: Mask


@dataclass_json
@dataclass
class PropagateDataResponse:
    frame_index: int
    results: List[PropagateDataValue]


@dataclass_json
@dataclass
class RemoveObjectResponse:
    results: List[PropagateDataResponse]


@dataclass_json
@dataclass
class CancelPorpagateResponse:
    success: bool


@dataclass_json
@dataclass
class InferenceSession:
    start_time: float
    last_use_time: float
    session_id: str
    state: Dict[str, Dict[str, Union[Tensor, Dict[int, Tensor]]]]
