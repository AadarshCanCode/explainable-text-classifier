# Explainable AI Text Classifier

A comprehensive local development project showcasing text classification explained using LIME (Local Interpretable Model-agnostic Explanations).

## Project Overview

This application provides a web interface to classify text into different categories (Fake News, Toxic Comment, or Sentiment) and visualizes **why** the underlying machine learning model made its decision. By leveraging LIME, the system highlights the exact words that positively or negatively influenced the prediction, giving developers and users transparency into the model's behavior.

### 3 Core Use Cases Supported:
1. **Fake News Detection** (Output: Fake / Real)
2. **Toxic Comment Detection** (Output: Toxic / Non-Toxic)
3. **Sentiment Analysis** (Output: Positive / Neutral / Negative)

## Architecture Explanation

The project is built entirely without Docker, utilizing a modern, modular Monorepo structure containing separate frontend and backend applications.

- **Backend (`/backend`)**: A robust Python (FastAPI) server handling RESTful requests. 
  - Uses `scikit-learn` to train specific classifiers on real-world Hugging Face datasets at startup (then caches models locally). 
  - Incorporates the `lime` package to compute locally interpretable weights for the text features.
  - Returns structured prediction data and raw HTML for the LIME explanation visualizer.
  
- **Frontend (`/frontend`)**: A React Single Page Application (SPA), bootstrapped with Vite.
  - Styled with **Tailwind CSS** for a clean, responsive, and aesthetically pleasing interface.
  - State management uses core React Hooks, while network requests are handled safely via `axios`.
  - Predictions are visualized using `Chart.js` via `react-chartjs-2`, rendering clear feature-weight bar charts alongside the native LIME HTML output.

## Setup Instructions

### Backend Setup (FastAPI & scikit-learn)

1. Navigate to the backend directory:
   ```bash
   cd backend
   ```
2. Create and activate a Python virtual environment:
   ```bash
   python -m venv venv
   # On macOS/Linux:
   source venv/bin/activate
   # On Windows:
   venv\Scripts\activate
   ```
3. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Start the Uvicorn web server:
   ```bash
   uvicorn app.main:app --reload
   ```
   *The backend will be available at `http://localhost:8000`.*

### Frontend Setup (React & Vite)

1. Open a new terminal and navigate to the frontend directory:
   ```bash
   cd frontend
   ```
2. Install the necessary packages:
   ```bash
   npm install
   ```
3. Start the Vite development server:
   ```bash
   npm run dev
   ```
   *The frontend will be available at `http://localhost:5173`.*

## API Documentation

### POST `/predict`

Analyzes text to return predictions, confidence, and LIME explanations.

**Request Body (JSON):**
```json
{
  "text": "This is a great and wonderful experience!",
  "task": "sentiment"  // One of: "fake_news", "toxic", "sentiment"
}
```

**Response (JSON):**
```json
{
  "prediction": "Positive",
  "confidence": 0.894,
  "explanation": [
      {"word": "wonderful", "weight": 0.35},
      {"word": "great", "weight": 0.21}
  ],
  "html": "<div>...LIME generated explanation...</div>"
}
```

## Example `curl` Request

```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{"text": "breaking news official shock claims", "task": "fake_news"}'
```

## Expected Output

![Expected UI Output Screenshot](placeholder.png)
*(Run both backend and frontend servers locally and point your browser to `localhost:5173` to view the beautiful dashboard)*


# Currently Writing a research paper on this