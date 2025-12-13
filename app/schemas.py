from typing import List, Literal
from pydantic import BaseModel, Field, field_validator


class TrafficPayload(BaseModel):
    server_id: str
    recent_traffic: List[float] = Field(min_length=5, max_length=5)
    
    @field_validator('recent_traffic')
    @classmethod
    def validate_traffic_values(cls, v: List[float]) -> List[float]:
        if any(val < 0 for val in v):
            raise ValueError('Traffic values must be non-negative')
        return v


class PredictionResult(BaseModel):
    server_id: str
    predicted_load: float
    action_suggested: Literal["SCALE_UP", "SCALE_DOWN", "MAINTAIN"]
    model_version: str