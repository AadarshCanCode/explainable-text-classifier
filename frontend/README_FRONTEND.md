# Explainable Text Classifier - Frontend

This repository houses the React frontend app built with Vite. Requirements were met by implementing responsive UI styling with Tailwind CSS and network requests via Axios.

## Core Features

- **ExplainableClassifier Component**: Offers the ability to drop down and select models (`Fake News`, `Toxic Comments`, `Sentiment Analysis`), input text, and dynamically render outputs.
- **Chart.js Visualizations**: `react-chartjs-2` maps out standard bar charts for the numerical interpretation of feature weights.
- **LIME HTML Injection**: Safely renders LIME `as_html()` directly beneath the primary summary results for deeper insights into language processing.

## Running Locally

To run the frontend alone, make sure the API logic exists on localhost:8000.

```bash
npm install
npm run dev
```
Navigate to `http://localhost:5173`.
