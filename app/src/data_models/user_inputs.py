from pydantic import BaseModel
from typing import Optional


class PredictionInput(BaseModel):
    img_base64: str
    model_name: str
    min_score: Optional[float] = 0.2
    filter_predictions: Optional[list[str]] = None
