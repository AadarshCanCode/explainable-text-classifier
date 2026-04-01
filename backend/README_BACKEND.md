# Explainable Text Classifier - Backend

This directory contains the Python FastAPI backend.

## Structure
- `app/main.py`: The FastAPI application, API routes, and CORS settings.
- `app/models.py`: Handles initializing the scikit-learn machine learning pipelines using synthetic datasets during server startup.
- `app/explain.py`: Integrates `lime.lime_text.LimeTextExplainer` to interpret the models' predictions.
- `app/schemas.py`: Pydantic models mapping incoming requests and outgoing responses.
- `app/config.py`: Configuration and environment settings.
- `app/utils.py`: Text preprocessing and normalisation functions.

## How to Run
Ensure you are using Python 3.10+.
```bash
python -m venv venv
# Linux/Mac
source venv/bin/activate
# Windows
venv\Scripts\activate

pip install -r requirements.txt
uvicorn app.main:app --reload
```
API Documentation is automatically available at `http://localhost:8000/docs`.
