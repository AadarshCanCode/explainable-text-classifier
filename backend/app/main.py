from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from app.schemas import PredictionRequest, PredictionResponse
from app.models import model_manager
from app.explain import explain_prediction
from app.config import settings
from contextlib import asynccontextmanager

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
    # Train models on startup
    model_manager.train_models()
    yield
    # Clean up (if any)

app = FastAPI(title=settings.PROJECT_NAME, lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_origin_regex=settings.CORS_ORIGIN_REGEX,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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
        }
        for task in TASK_CATALOG
    }

@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    task = request.task

    # Validate task
    valid_tasks = list(TASK_CATALOG.keys())
    if task not in valid_tasks:
        raise HTTPException(status_code=400, detail=f"Invalid task. Must be one of {valid_tasks}")

    model = model_manager.get_model(task)
    class_names = model_manager.get_class_names(task)

    if model is None or class_names is None:
        raise HTTPException(status_code=500, detail="Model not initialized")

    try:
        # Extract predict_proba function to pass to LIME
        predict_proba_fn = model.predict_proba

        # Calculate explanation
        explanation_result = explain_prediction(
            text=request.text,
            predict_proba_fn=predict_proba_fn,
            class_names=class_names
        )
        return explanation_result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
