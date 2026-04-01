from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from app.schemas import PredictionRequest, PredictionResponse
from app.models import model_manager
from app.explain import explain_prediction
from app.config import settings
from contextlib import asynccontextmanager

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
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"message": "Welcome to Explainable Text Classification API"}

@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    task = request.task
    
    # Validate task
    valid_tasks = ["fake_news", "toxic", "sentiment"]
    if task not in valid_tasks:
        raise HTTPException(status_code=400, detail=f"Invalid task. Must be one of {valid_tasks}")
        
    model = model_manager.get_model(task)
    class_names = model_manager.get_class_names(task)
    
    if not model:
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
