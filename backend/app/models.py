import pickle
import time
import warnings
from pathlib import Path
from typing import Callable, Dict, List, Tuple

from datasets import load_dataset
from sklearn import __version__ as sklearn_version
from sklearn.exceptions import InconsistentVersionWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import Pipeline, make_pipeline


TaskData = Tuple[List[str], List[str]]
ModelFactory = Callable[[], Pipeline]


def _clean_text(value: object) -> str:
    if value is None:
        return ""
    return str(value).strip()


class ModelManager:
    def __init__(self):
        self.model_dir = Path(__file__).resolve().parent.parent / "trained_models"
        self.model_dir.mkdir(parents=True, exist_ok=True)

        self.model_specs = self._build_model_specs()
        self.models: Dict[str, Pipeline] = {}
        self.task_models: Dict[str, Dict[str, Pipeline]] = {}
        self.class_names: Dict[str, Dict[str, list[str]]] = {}
        self.best_models: Dict[str, str] = {}
        self.model_sources: Dict[str, str] = {}
        self.metrics: Dict[str, Dict[str, dict]] = {}

    def _build_model_specs(self) -> Dict[str, Dict[str, object]]:
        return {
            "logreg": {
                "label": "Logistic Regression",
                "factory": lambda: make_pipeline(
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
                ),
            },
            "sgd_log": {
                "label": "SGD Classifier (Log Loss)",
                "factory": lambda: make_pipeline(
                    TfidfVectorizer(
                        max_features=50000,
                        ngram_range=(1, 2),
                        min_df=2,
                        sublinear_tf=True,
                        strip_accents="unicode",
                    ),
                    SGDClassifier(
                        loss="log_loss",
                        alpha=1e-5,
                        max_iter=3000,
                        tol=1e-3,
                        random_state=42,
                    ),
                ),
            },
            "complement_nb": {
                "label": "Complement Naive Bayes",
                "factory": lambda: make_pipeline(
                    TfidfVectorizer(
                        max_features=80000,
                        ngram_range=(1, 2),
                        min_df=2,
                        strip_accents="unicode",
                    ),
                    ComplementNB(alpha=0.3),
                ),
            },
        }

    def _model_path(self, task: str) -> Path:
        return self.model_dir / f"{task}_modelpack.pkl"

    def _save_task_pack(self, task: str) -> None:
        payload = {
            "all_models": self.task_models[task],
            "class_names": self.class_names[task],
            "best_model": self.best_models[task],
            "source": self.model_sources[task],
            "metrics": self.metrics[task],
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

        required_fields = {"all_models", "class_names", "best_model", "source", "metrics", "sklearn_version"}
        if not required_fields.issubset(set(payload.keys())):
            return False

        cached_version = payload.get("sklearn_version")
        if cached_version != sklearn_version:
            print(
                f"[{task}] Cached models built with scikit-learn {cached_version}; "
                f"current version is {sklearn_version}. Retraining..."
            )
            return False

        all_models = payload["all_models"]
        best_model = payload["best_model"]
        if not isinstance(all_models, dict) or best_model not in all_models:
            return False

        self.task_models[task] = all_models
        self.best_models[task] = best_model
        self.models[task] = all_models[best_model]
        self.class_names[task] = payload["class_names"]
        self.model_sources[task] = payload["source"]
        self.metrics[task] = payload["metrics"]
        return True

    def _load_fake_news_dataset(self, max_rows: int = 20000) -> TaskData:
        ds = load_dataset("mrm8488/fake-news", split=f"train[:{max_rows}]").shuffle(seed=42)
        text = [_clean_text(t) for t in ds["text"]]
        labels = ["Real" if int(v) == 0 else "Fake" for v in ds["label"]]
        return text, labels

    def _load_toxic_dataset(self, per_class: int = 6000) -> TaskData:
        ds = load_dataset("tasksource/jigsaw_toxicity", split="train").shuffle(seed=42)
        toxic = ds.filter(lambda row: int(row["toxic"]) == 1)
        non_toxic = ds.filter(lambda row: int(row["toxic"]) == 0)

        toxic_count = min(per_class, len(toxic))
        non_toxic_count = min(per_class, len(non_toxic))

        toxic = toxic.select(range(toxic_count))
        non_toxic = non_toxic.select(range(non_toxic_count))

        text = [_clean_text(t) for t in toxic["comment_text"]] + [_clean_text(t) for t in non_toxic["comment_text"]]
        labels = ["Toxic"] * len(toxic) + ["Non-Toxic"] * len(non_toxic)
        return text, labels

    def _load_sentiment_dataset(self, per_class: int = 5000) -> TaskData:
        ds = load_dataset("tweet_eval", "sentiment", split="train").shuffle(seed=42)
        negative = ds.filter(lambda row: int(row["label"]) == 0)
        neutral = ds.filter(lambda row: int(row["label"]) == 1)
        positive = ds.filter(lambda row: int(row["label"]) == 2)

        negative = negative.select(range(min(per_class, len(negative))))
        neutral = neutral.select(range(min(per_class, len(neutral))))
        positive = positive.select(range(min(per_class, len(positive))))

        text = [_clean_text(t) for t in negative["text"]] + [_clean_text(t) for t in neutral["text"]] + [
            _clean_text(t) for t in positive["text"]
        ]
        labels = ["Negative"] * len(negative) + ["Neutral"] * len(neutral) + ["Positive"] * len(positive)
        return text, labels

    def _compute_metrics(self, y_true: List[str], y_pred: List[str], train_time_seconds: float) -> dict:
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true,
            y_pred,
            average="weighted",
            zero_division=0,
        )
        return {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision_weighted": float(precision),
            "recall_weighted": float(recall),
            "f1_weighted": float(f1),
            "train_time_seconds": float(train_time_seconds),
        }

    def _train_task(self, task: str, data_loader, source: str) -> None:
        if self._load_saved_model(task):
            print(f"[{task}] Loaded cached model pack from {self._model_path(task)}")
            return

        print(f"[{task}] Downloading and preparing dataset...")
        X, y = data_loader()
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42,
            stratify=y,
        )

        print(f"[{task}] Training {len(self.model_specs)} candidate models on {len(X_train)} train samples...")
        task_models: Dict[str, Pipeline] = {}
        task_class_names: Dict[str, list[str]] = {}
        task_metrics: Dict[str, dict] = {}

        for model_name, spec in self.model_specs.items():
            start = time.perf_counter()
            pipeline = spec["factory"]()
            pipeline.fit(X_train, y_train)
            train_time = time.perf_counter() - start

            y_pred = pipeline.predict(X_test)
            metrics = self._compute_metrics(y_test, y_pred, train_time)
            metrics["model_label"] = spec["label"]

            task_models[model_name] = pipeline
            task_class_names[model_name] = pipeline.classes_.tolist()
            task_metrics[model_name] = metrics

            print(
                f"[{task}] {model_name}: accuracy={metrics['accuracy']:.4f} "
                f"f1_weighted={metrics['f1_weighted']:.4f} train_time={metrics['train_time_seconds']:.2f}s"
            )

        ranked_models = sorted(
            task_metrics.items(),
            key=lambda item: (item[1]["f1_weighted"], item[1]["accuracy"]),
            reverse=True,
        )
        best_model_name = ranked_models[0][0]

        self.task_models[task] = task_models
        self.class_names[task] = task_class_names
        self.best_models[task] = best_model_name
        self.models[task] = task_models[best_model_name]
        self.model_sources[task] = source
        self.metrics[task] = task_metrics

        self._save_task_pack(task)
        print(f"[{task}] Best model: {best_model_name}. Saved model pack to {self._model_path(task)}")

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
        print("All model packs are ready.")

    def get_model(self, task: str, model_name: str | None = None) -> Pipeline | None:
        if model_name is None or model_name == "best":
            return self.models.get(task)
        return self.task_models.get(task, {}).get(model_name)

    def get_models_for_task(self, task: str) -> Dict[str, Pipeline]:
        return self.task_models.get(task, {})

    def get_class_names(self, task: str, model_name: str | None = None) -> list[str] | None:
        if task not in self.class_names:
            return None
        if model_name is None or model_name == "best":
            best_name = self.best_models.get(task)
            if best_name is None:
                return None
            return self.class_names[task].get(best_name)
        return self.class_names[task].get(model_name)

    def get_best_model_name(self, task: str) -> str | None:
        return self.best_models.get(task)

    def get_model_source(self, task: str) -> str:
        return self.model_sources.get(task, "Unknown")

    def get_model_label(self, model_name: str) -> str:
        spec = self.model_specs.get(model_name)
        if not spec:
            return model_name
        return str(spec["label"])

    def get_available_models(self, task: str) -> List[dict]:
        model_names = list(self.task_models.get(task, {}).keys())
        return [
            {
                "model_name": name,
                "model_label": self.get_model_label(name),
                "is_best": name == self.best_models.get(task),
            }
            for name in model_names
        ]

    def get_task_benchmark(self, task: str) -> dict:
        task_metrics = self.metrics.get(task, {})
        rows = []
        for model_name, metric in task_metrics.items():
            rows.append(
                {
                    "model_name": model_name,
                    "model_label": self.get_model_label(model_name),
                    "accuracy": metric["accuracy"],
                    "precision_weighted": metric["precision_weighted"],
                    "recall_weighted": metric["recall_weighted"],
                    "f1_weighted": metric["f1_weighted"],
                    "train_time_seconds": metric["train_time_seconds"],
                }
            )

        rows.sort(key=lambda item: (item["f1_weighted"], item["accuracy"]), reverse=True)
        return {
            "best_model": self.best_models.get(task),
            "models": rows,
        }

    def get_all_benchmarks(self) -> Dict[str, dict]:
        return {
            task: self.get_task_benchmark(task)
            for task in self.task_models.keys()
        }


model_manager = ModelManager()
