from pydantic import BaseModel
from typing import List


class DetectionResult(BaseModel):
    bbox: List[int]
    distance_m: float


class InferenceResponse(BaseModel):
    results: List[DetectionResult]
