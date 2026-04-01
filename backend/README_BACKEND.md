# Explainable Text Classifier - Backend

This directory contains the Python FastAPI backend.

## Structure
- `app/main.py`: The FastAPI application, API routes, and CORS settings.
- `app/models.py`: Downloads real-world datasets, trains explainable scikit-learn pipelines, and caches trained models.
- `app/explain.py`: Integrates `lime.lime_text.LimeTextExplainer` to interpret the models' predictions.
- `app/schemas.py`: Pydantic models mapping incoming requests and outgoing responses.
- `app/config.py`: Configuration and environment settings.
- `app/utils.py`: Text preprocessing and normalisation functions.
- `trained_models/`: Auto-generated cached `.pkl` models (created after first successful training run).

## Real Datasets Used
- `fake_news`: Hugging Face dataset `mrm8488/fake-news`
- `toxic`: Hugging Face dataset `tasksource/jigsaw_toxicity`
- `sentiment`: Hugging Face dataset `tweet_eval` (configuration: `sentiment`)

## Open-Source Model Used
All tasks use:
- `TfidfVectorizer` (word/phrase features)
- `LogisticRegression` classifier

This combination is open-source, downloadable (saved locally as `.pkl`), fast enough for local use, and works very well with LIME explanations because token-level feature weights are directly interpretable.

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
