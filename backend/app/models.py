import pickle
from pathlib import Path
from typing import Dict, List, Tuple
import warnings

from datasets import load_dataset
from sklearn.exceptions import InconsistentVersionWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn import __version__ as sklearn_version


TaskData = Tuple[List[str], List[str]]


def _clean_text(value: object) -> str:
    if value is None:
        return ""
    return str(value).strip()


class ModelManager:
    def __init__(self):
        self.models: Dict[str, Pipeline] = {}
        self.class_names: Dict[str, list[str]] = {}
        self.model_sources: Dict[str, str] = {}
        self.model_dir = Path(__file__).resolve().parent.parent / "trained_models"
        self.model_dir.mkdir(parents=True, exist_ok=True)

    def _model_path(self, task: str) -> Path:
        return self.model_dir / f"{task}_tfidf_logreg.pkl"

    def _save_model(self, task: str, pipeline: Pipeline, source: str) -> None:
        payload = {
            "pipeline": pipeline,
            "class_names": pipeline.classes_.tolist(),
            "source": source,
            "sklearn_version": sklearn_version,
        }
        with self._model_path(task).open("wb") as f:
            pickle.dump(payload, f)

    def _load_saved_model(self, task: str) -> bool:
        path = self._model_path(task)
        if not path.exists():
            return False
        with path.open("rb") as f:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", InconsistentVersionWarning)
                payload = pickle.load(f)

        cached_version = payload.get("sklearn_version")
        if cached_version != sklearn_version:
            print(
                f"[{task}] Cached model built with scikit-learn {cached_version}; "
                f"current version is {sklearn_version}. Retraining..."
            )
            return False

        self.models[task] = payload["pipeline"]
        self.class_names[task] = payload["class_names"]
        self.model_sources[task] = payload.get("source", "Unknown cached source")
        return True

    def _build_pipeline(self) -> Pipeline:
        # Logistic Regression + TF-IDF is strong, open-source, and highly explainable with LIME.
        return make_pipeline(
            TfidfVectorizer(
                max_features=40000,
                ngram_range=(1, 2),
                min_df=3,
                sublinear_tf=True,
                strip_accents="unicode",
            ),
            LogisticRegression(
                max_iter=2000,
                solver="saga",
                n_jobs=-1,
                random_state=42,
            ),
        )

    def _load_fake_news_dataset(self, max_rows: int = 20000) -> TaskData:
        ds = load_dataset("mrm8488/fake-news", split=f"train[:{max_rows}]").shuffle(seed=42)
        text = [_clean_text(t) for t in ds["text"]]
        # Dataset label semantics: 0 = Reuters-style real news, 1 = fake/manipulated news.
        labels = ["Real" if int(v) == 0 else "Fake" for v in ds["label"]]
        return text, labels

    def _load_toxic_dataset(self, per_class: int = 6000) -> TaskData:
        ds = load_dataset("tasksource/jigsaw_toxicity", split="train").shuffle(seed=42)
        toxic = ds.filter(lambda row: int(row["toxic"]) == 1).select(range(per_class))
        non_toxic = ds.filter(lambda row: int(row["toxic"]) == 0).select(range(per_class))

        text = [_clean_text(t) for t in toxic["comment_text"]] + [_clean_text(t) for t in non_toxic["comment_text"]]
        labels = ["Toxic"] * len(toxic) + ["Non-Toxic"] * len(non_toxic)
        return text, labels

    def _load_sentiment_dataset(self, per_class: int = 5000) -> TaskData:
        ds = load_dataset("tweet_eval", "sentiment", split="train").shuffle(seed=42)
        negative = ds.filter(lambda row: int(row["label"]) == 0).select(range(per_class))
        neutral = ds.filter(lambda row: int(row["label"]) == 1).select(range(per_class))
        positive = ds.filter(lambda row: int(row["label"]) == 2).select(range(per_class))

        text = [_clean_text(t) for t in negative["text"]] + [_clean_text(t) for t in neutral["text"]] + [
            _clean_text(t) for t in positive["text"]
        ]
        labels = ["Negative"] * len(negative) + ["Neutral"] * len(neutral) + ["Positive"] * len(positive)
        return text, labels

    def _train_task(self, task: str, data_loader, source: str) -> None:
        if self._load_saved_model(task):
            print(f"[{task}] Loaded cached model from {self._model_path(task)}")
            return

        print(f"[{task}] Downloading and preparing dataset...")
        X, y = data_loader()
        pipeline = self._build_pipeline()
        print(f"[{task}] Training model on {len(X)} samples...")
        pipeline.fit(X, y)

        self.models[task] = pipeline
        self.class_names[task] = pipeline.classes_.tolist()
        self.model_sources[task] = source
        self._save_model(task, pipeline, source)
        print(f"[{task}] Training complete. Saved to {self._model_path(task)}")

    def train_models(self) -> None:
        self._train_task(
            task="fake_news",
            data_loader=lambda: self._load_fake_news_dataset(max_rows=20000),
            source="HuggingFace: mrm8488/fake-news",
        )
        self._train_task(
            task="toxic",
            data_loader=lambda: self._load_toxic_dataset(per_class=6000),
            source="HuggingFace: tasksource/jigsaw_toxicity",
        )
        self._train_task(
            task="sentiment",
            data_loader=lambda: self._load_sentiment_dataset(per_class=5000),
            source="HuggingFace: tweet_eval (sentiment)",
        )
        print("All models are ready.")

    def get_model(self, task: str) -> Pipeline:
        return self.models.get(task)

    def get_class_names(self, task: str) -> list[str]:
        return self.class_names.get(task)

    def get_model_source(self, task: str) -> str:
        return self.model_sources.get(task, "Unknown")


model_manager = ModelManager()
