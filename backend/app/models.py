import os
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline, make_pipeline
from typing import Dict, Any, Tuple

# Simple synthetic datasets to train on startup
fake_news_data = (
    ["breaking news official shock claims", "shocking official news just in", "aliens found in the official shocking report", "completely official and shocking", "exclusive breaking report shock"],
    ["Fake", "Fake", "Fake", "Fake", "Fake"]
)
real_news_data = (
    ["the government passed a new bill", "economic growth is steady", "local elections are coming up", "healthcare policies updated", "climate conference yields positive results"],
    ["Real", "Real", "Real", "Real", "Real"]
)

toxic_data = (
    ["you are stupid and I hate you", "this is garbage", "shut up you idiot", "you are the worst person", "absolute trash"],
    ["Toxic", "Toxic", "Toxic", "Toxic", "Toxic"]
)
non_toxic_data = (
    ["thank you for the help", "have a great day", "this is interesting", "I appreciate your response", "good job"],
    ["Non-Toxic", "Non-Toxic", "Non-Toxic", "Non-Toxic", "Non-Toxic"]
)

sentiment_pos = (
    ["I love this so much", "this is amazing and wonderful", "great job excellent", "fantastic outstanding experience", "really positive awesome"],
    ["Positive", "Positive", "Positive", "Positive", "Positive"]
)
sentiment_neu = (
    ["this is okay", "it was average", "nothing special but fine", "neutral response just regular", "it is what it is"],
    ["Neutral", "Neutral", "Neutral", "Neutral", "Neutral"]
)
sentiment_neg = (
    ["this is terrible", "hate this awful horrible", "worst experience ever", "doing this makes me sad and angry", "absolutely disgusting"],
    ["Negative", "Negative", "Negative", "Negative", "Negative"]
)

class ModelManager:
    def __init__(self):
        self.models: Dict[str, Pipeline] = {}
        self.class_names: Dict[str, list[str]] = {}

    def train_models(self):
        # 1. Fake News (Logistic Regression)
        X_fake = fake_news_data[0] + real_news_data[0]
        y_fake = fake_news_data[1] + real_news_data[1]
        fake_pipeline = make_pipeline(
            TfidfVectorizer(),
            LogisticRegression(max_iter=1000)
        )
        fake_pipeline.fit(X_fake, y_fake)
        self.models['fake_news'] = fake_pipeline
        self.class_names['fake_news'] = fake_pipeline.classes_.tolist()

        # 2. Toxic Comment (Linear SVM with probabilities)
        print("Loading Jigsaw Toxic Comment Challenge data from HuggingFace...")
        try:
            from datasets import load_dataset
            # Load dataset and extract samples
            ds = load_dataset("tasksource/jigsaw_toxicity", split="train")
            ds = ds.shuffle(seed=42)
            
            # Filter to get toxic and non-toxic examples
            toxic_examples = ds.filter(lambda x: x['toxic'] == 1).select(range(1000))
            non_toxic_examples = ds.filter(lambda x: x['toxic'] == 0).select(range(1000))
            
            X_toxic = list(toxic_examples['comment_text']) + list(non_toxic_examples['comment_text'])
            y_toxic = ["Toxic"] * 1000 + ["Non-Toxic"] * 1000
        except Exception as e:
            print(f"Failed to load Jigsaw dataset: {e}. Falling back to synthetic.")
            X_toxic = toxic_data[0] + non_toxic_data[0]
            y_toxic = toxic_data[1] + non_toxic_data[1]

        toxic_pipeline = make_pipeline(
            TfidfVectorizer(max_features=10000), # Limit features for faster train
            SVC(kernel='linear', probability=True, max_iter=2000)
        )
        toxic_pipeline.fit(X_toxic, y_toxic)
        self.models['toxic'] = toxic_pipeline
        self.class_names['toxic'] = toxic_pipeline.classes_.tolist()
        print("Toxic Comment model trained successfully!")

        # 3. Sentiment Analysis (Logistic Regression)
        X_sent = sentiment_pos[0] + sentiment_neu[0] + sentiment_neg[0]
        y_sent = sentiment_pos[1] + sentiment_neu[1] + sentiment_neg[1]
        sent_pipeline = make_pipeline(
            TfidfVectorizer(),
            LogisticRegression(max_iter=1000)
        )
        sent_pipeline.fit(X_sent, y_sent)
        self.models['sentiment'] = sent_pipeline
        self.class_names['sentiment'] = sent_pipeline.classes_.tolist()
        print("Models trained successfully!")

    def get_model(self, task: str) -> Pipeline:
        return self.models.get(task)

    def get_class_names(self, task: str) -> list[str]:
        return self.class_names.get(task)

model_manager = ModelManager()
