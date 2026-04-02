from lime.lime_text import LimeTextExplainer
from typing import Dict, Any, Callable
import numpy as np

def explain_prediction(
    text: str, 
    predict_proba_fn: Callable, 
    class_names: list[str]
) -> Dict[str, Any]:
    """
    Generate explanation using LIME for text classification.
    """
    # Initialize the specific text explainer
    explainer = LimeTextExplainer(class_names=class_names)
    
    # Generate explanation
    exp = explainer.explain_instance(
        text, 
        predict_proba_fn, 
        num_features=10,
        top_labels=1 # We want explanation for the highest probability predicted class
    )
    
    # Get the predicted label index and name
    predicted_idx = exp.available_labels()[0]
    predicted_label = class_names[predicted_idx]
    
    # Get confidence score
    probs = predict_proba_fn([text])[0]
    confidence_score = float(probs[predicted_idx])
    
    # Get top features (word, weight) for the predicted class
    feature_weights = exp.as_list(label=predicted_idx)
    top_features = [{"word": word, "weight": weight} for word, weight in feature_weights]
    
    # Generate static HTML suitable for embedding
    # Note: LIME's as_html() by default might include scripts that are stripped or blocked in React.
    # We use show_in_notebook=False to get the raw HTML string.
    explanation_html = exp.as_html(predict_proba=False)
    
    return {
        "prediction": predicted_label,
        "confidence": confidence_score,
        "explanation": top_features,
        "html": explanation_html
    }
