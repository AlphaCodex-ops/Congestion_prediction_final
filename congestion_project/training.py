from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_curve,
    roc_auc_score,
)
from sklearn.utils.class_weight import compute_sample_weight


class ModelTrainer:
    def __init__(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        config: dict[str, Any] | None = None,
        feature_names: list[str] | None = None,
    ) -> None:
        self.X_train = np.asarray(X_train, dtype=float)
        self.y_train = np.asarray(y_train, dtype=int)
        self.X_test = np.asarray(X_test, dtype=float)
        self.y_test = np.asarray(y_test, dtype=int)
        self.config = config or {}
        self.feature_names = feature_names or [f"feature_{i}" for i in range(self.X_train.shape[1])]
        self.model: GradientBoostingClassifier | None = None
        self.metrics: dict[str, Any] | None = None
        if self.X_train.ndim != 2 or self.X_test.ndim != 2:
            raise ValueError("Training and test features must be two-dimensional arrays.")
        if self.X_train.shape[1] != self.X_test.shape[1] or self.X_train.shape[1] != len(self.feature_names):
            raise ValueError("Training, test, and feature-name dimensions must match.")
        if len(self.X_train) != len(self.y_train) or len(self.X_test) != len(self.y_test):
            raise ValueError("Feature and label counts must match.")
        if len(np.unique(self.y_train)) < 2:
            raise ValueError("Training data must contain both congested and non-congested samples.")

    def train(self) -> GradientBoostingClassifier:
        supported = {
            "n_estimators",
            "learning_rate",
            "max_depth",
            "min_samples_split",
            "min_samples_leaf",
            "subsample",
            "random_state",
        }
        parameters = {key: value for key, value in self.config.items() if key in supported}
        self.model = GradientBoostingClassifier(**parameters)
        sample_weight = (
            compute_sample_weight(class_weight="balanced", y=self.y_train)
            if self.config.get("class_weight", "balanced") == "balanced"
            else None
        )
        self.model.fit(self.X_train, self.y_train, sample_weight=sample_weight)
        self.metrics = self.evaluate()
        return self.model

    def evaluate(self) -> dict[str, Any]:
        if self.model is None:
            raise RuntimeError("Call train() before evaluate().")
        predictions = self.model.predict(self.X_test)
        probabilities = self.model.predict_proba(self.X_test)[:, 1]
        auc = (
            float(roc_auc_score(self.y_test, probabilities))
            if len(np.unique(self.y_test)) == 2
            else None
        )
        self.metrics = {
            "accuracy": float(accuracy_score(self.y_test, predictions)),
            "precision": float(precision_score(self.y_test, predictions, zero_division=0)),
            "recall": float(recall_score(self.y_test, predictions, zero_division=0)),
            "f1": float(f1_score(self.y_test, predictions, zero_division=0)),
            "roc_auc": auc,
            "confusion_matrix": confusion_matrix(self.y_test, predictions, labels=[0, 1]),
            "feature_importance": dict(zip(self.feature_names, self.model.feature_importances_.tolist())),
            "baseline_accuracy": float(
                max(np.mean(self.y_test == 0), np.mean(self.y_test == 1))
            ),
        }
        return self.metrics

    def save_model(self, path: str | Path) -> Path:
        if self.model is None:
            raise RuntimeError("Call train() before saving the model.")
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "model": self.model,
                "feature_names": self.feature_names,
                "format_version": 1,
            },
            destination,
        )
        return destination

    def plot_results(self, output_dir: str | Path) -> list[Path]:
        if self.metrics is None:
            self.evaluate()
        try:
            import matplotlib.pyplot as plt
        except ImportError as exc:
            raise RuntimeError(
                "Plotting requires the optional matplotlib dependency. Install with: pip install -e '.[plots]'"
            ) from exc
        destination = Path(output_dir)
        destination.mkdir(parents=True, exist_ok=True)
        saved: list[Path] = []
        matrix = np.asarray(self.metrics["confusion_matrix"])
        fig, ax = plt.subplots()
        image = ax.imshow(matrix, cmap="Blues")
        ax.set(title="Congestion prediction confusion matrix", xlabel="Predicted", ylabel="Actual")
        ax.set_xticks([0, 1], labels=["Clear", "Congested"])
        ax.set_yticks([0, 1], labels=["Clear", "Congested"])
        for row in range(2):
            for column in range(2):
                ax.text(column, row, str(matrix[row, column]), ha="center", va="center")
        fig.colorbar(image, ax=ax)
        path = destination / "confusion_matrix.png"
        fig.tight_layout()
        fig.savefig(path)
        plt.close(fig)
        saved.append(path)

        if len(np.unique(self.y_test)) == 2:
            false_positive_rate, true_positive_rate, _ = roc_curve(
                self.y_test, self.model.predict_proba(self.X_test)[:, 1]
            )
            fig, ax = plt.subplots()
            ax.plot(false_positive_rate, true_positive_rate, label=f"ROC AUC = {self.metrics['roc_auc']:.3f}")
            ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Chance")
            ax.set(
                title="Congestion prediction ROC curve",
                xlabel="False positive rate",
                ylabel="True positive rate",
            )
            ax.legend(loc="lower right")
            path = destination / "roc_curve.png"
            fig.tight_layout()
            fig.savefig(path)
            plt.close(fig)
            saved.append(path)

        importances = self.metrics["feature_importance"]
        ordered = sorted(importances.items(), key=lambda item: item[1], reverse=True)
        names, values = zip(*ordered)
        fig, ax = plt.subplots(figsize=(9, max(4, len(names) * 0.25)))
        ax.barh(list(names)[::-1], list(values)[::-1])
        ax.set(title="Feature importance", xlabel="Importance")
        path = destination / "feature_importance.png"
        fig.tight_layout()
        fig.savefig(path)
        plt.close(fig)
        saved.append(path)
        return saved


class Predictor:
    def __init__(self, model_path: str | Path, scaler_path: str | Path) -> None:
        model_file = Path(model_path)
        scaler_file = Path(scaler_path)
        if not model_file.is_file():
            raise FileNotFoundError(f"Model file does not exist: {model_file}")
        if not scaler_file.is_file():
            raise FileNotFoundError(f"Scaler file does not exist: {scaler_file}")
        bundle = joblib.load(model_file)
        if not isinstance(bundle, dict) or "model" not in bundle or "feature_names" not in bundle:
            raise ValueError(f"Invalid model bundle: {model_file}")
        self.model = bundle["model"]
        self.feature_names = list(bundle["feature_names"])
        self.scaler = joblib.load(scaler_file)
        if not hasattr(self.scaler, "transform"):
            raise TypeError(f"Scaler does not support transform: {scaler_file}")
        if len(self.feature_names) != self.scaler.n_features_in_:
            raise ValueError("Model feature names and scaler dimensions do not match.")

    def get_feature_names(self) -> list[str]:
        return list(self.feature_names)

    def predict(self, feature_vector: np.ndarray | list[float]) -> float:
        values = np.asarray(feature_vector, dtype=float)
        if values.ndim != 1 or values.shape[0] != len(self.feature_names):
            raise ValueError(f"Expected one feature vector with {len(self.feature_names)} values.")
        if not np.isfinite(values).all():
            raise ValueError("Feature values must all be finite.")
        scaled = self.scaler.transform(values.reshape(1, -1))
        return float(self.model.predict_proba(scaled)[0, 1])

    def predict_batch(self, X: np.ndarray | list[list[float]]) -> np.ndarray:
        values = np.asarray(X, dtype=float)
        if values.ndim != 2 or values.shape[1] != len(self.feature_names):
            raise ValueError(f"Expected a feature matrix with {len(self.feature_names)} columns.")
        if not np.isfinite(values).all():
            raise ValueError("Feature values must all be finite.")
        return self.model.predict_proba(self.scaler.transform(values))[:, 1]
