from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

class PredictionRequest(BaseModel):
    text: str = Field(..., title="Text to classify", min_length=1)
    task: str = Field(..., title="Task type", description="fake_news | toxic | sentiment")

class ExplanationFeature(BaseModel):
    word: str
    weight: float

class PredictionResponse(BaseModel):
    prediction: str
    confidence: float
    explanation: List[ExplanationFeature]
    html: str
