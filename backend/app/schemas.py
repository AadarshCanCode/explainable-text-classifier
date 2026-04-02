from pydantic import BaseModel, Field
from typing import List

class PredictionRequest(BaseModel):
    text: str = Field(..., title="Text to classify", min_length=1)
    task: str = Field(..., title="Task type", description="fake_news | toxic | sentiment")

class ExplanationFeature(BaseModel):
    word: str
    weight: float


class ClassProbability(BaseModel):
    label: str
    probability: float


class PredictionResponse(BaseModel):
    prediction: str
    confidence: float
    explanation: List[ExplanationFeature]
    probabilities: List[ClassProbability]
    html: str
