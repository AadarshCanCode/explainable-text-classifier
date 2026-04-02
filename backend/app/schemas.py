from pydantic import BaseModel, Field
from typing import List, Optional


class PredictionRequest(BaseModel):
    text: str = Field(..., title="Text to classify", min_length=1)
    task: str = Field(..., title="Task type", description="fake_news | toxic | sentiment")
    model_name: Optional[str] = Field(
        default=None,
        title="Model name",
        description="Optional model id from /benchmarks. If omitted, the best model for the task is used.",
    )


class ComparePredictionRequest(BaseModel):
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
    model_name: str
    html: str


class ModelMetric(BaseModel):
    model_name: str
    model_label: str
    accuracy: float
    precision_weighted: float
    recall_weighted: float
    f1_weighted: float
    train_time_seconds: float


class TaskBenchmark(BaseModel):
    best_model: Optional[str]
    models: List[ModelMetric]


class ComparePredictionRow(BaseModel):
    model_name: str
    model_label: str
    prediction: str
    confidence: float


class ComparePredictionResponse(BaseModel):
    task: str
    selected_by_metric: Optional[str]
    predictions: List[ComparePredictionRow]
