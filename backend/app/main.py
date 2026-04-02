from contextlib import asynccontextmanager
from typing import Dict

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.explain import explain_prediction
from app.models import model_manager
from app.schemas import (
    ComparePredictionRequest,
    ComparePredictionResponse,
    PredictionRequest,
    PredictionResponse,
    TaskBenchmark,
)

TASK_CATALOG = {
    "fake_news": {
        "label": "Fake News Detection",
        "description": "Classify whether a news-like article appears fake or real.",
        "sample_text": "Reuters reported that the central bank raised interest rates after inflation cooled.",
    },
    "toxic": {
        "label": "Toxic Comment Detection",
        "description": "Detect harmful or abusive user comments.",
        "sample_text": "You are an idiot and nobody wants your opinion.",
    },
    "sentiment": {
        "label": "Sentiment Analysis",
        "description": "Classify text sentiment into negative, neutral, or positive.",
        "sample_text": "The product is okay overall, but I expected better battery life.",
    },
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    model_manager.train_models()
    yield


app = FastAPI(title=settings.PROJECT_NAME, lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_origin_regex=settings.CORS_ORIGIN_REGEX,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _validate_task(task: str) -> None:
    valid_tasks = list(TASK_CATALOG.keys())
    if task not in valid_tasks:
        raise HTTPException(status_code=400, detail=f"Invalid task. Must be one of {valid_tasks}")


@app.get("/")
def read_root():
    return {"message": "Welcome to Explainable Text Classification API"}


@app.get("/health")
def health():
    tasks = list(TASK_CATALOG.keys())
    loaded = [task for task in tasks if model_manager.get_model(task) is not None]
    return {
        "status": "ok" if len(loaded) == len(tasks) else "degraded",
        "loaded_models": loaded,
        "total_models": len(tasks),
    }


@app.get("/tasks")
def tasks():
    return TASK_CATALOG


@app.get("/model-info")
def model_info():
    return {
        task: {
            "classes": model_manager.get_class_names(task),
            "source": model_manager.get_model_source(task),
            "cached": model_manager.get_model(task) is not None,
            "best_model": model_manager.get_best_model_name(task),
            "available_models": model_manager.get_available_models(task),
        }
        for task in TASK_CATALOG
    }


@app.get("/benchmarks", response_model=Dict[str, TaskBenchmark])
def benchmarks():
    return model_manager.get_all_benchmarks()


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    task = request.task
    _validate_task(task)

    model_name = request.model_name if request.model_name else "best"
    model = model_manager.get_model(task, model_name)
    class_names = model_manager.get_class_names(task, model_name)

    if model is None:
        available = [item["model_name"] for item in model_manager.get_available_models(task)]
        raise HTTPException(
            status_code=400,
            detail=f"Unknown model '{model_name}' for task '{task}'. Available: {available}",
        )

    if class_names is None:
        raise HTTPException(status_code=500, detail="Class names not initialized")

    try:
        explanation_result = explain_prediction(
            text=request.text,
            predict_proba_fn=model.predict_proba,
            class_names=class_names,
        )
        explanation_result["model_name"] = (
            model_manager.get_best_model_name(task) if model_name == "best" else model_name
        )
        return explanation_result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/compare", response_model=ComparePredictionResponse)
def compare_predictions(request: ComparePredictionRequest):
    task = request.task
    _validate_task(task)

    models = model_manager.get_models_for_task(task)
    if not models:
        raise HTTPException(status_code=500, detail="Task models are not initialized")

    rows = []
    for model_name, model in models.items():
        class_names = model_manager.get_class_names(task, model_name)
        if class_names is None:
            continue

        probs = model.predict_proba([request.text])[0]
        idx = int(np.argmax(probs))
        rows.append(
            {
                "model_name": model_name,
                "model_label": model_manager.get_model_label(model_name),
                "prediction": class_names[idx],
                "confidence": float(probs[idx]),
            }
        )

    rows.sort(key=lambda row: row["confidence"], reverse=True)

    return {
        "task": task,
        "selected_by_metric": model_manager.get_best_model_name(task),
        "predictions": rows,
    }
