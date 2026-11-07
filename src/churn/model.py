"""Train, compare and evaluate churn classifiers."""
from __future__ import annotations

from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import RocCurveDisplay, accuracy_score, f1_score, roc_auc_score  # noqa: E402
from sklearn.model_selection import cross_val_score, train_test_split  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402

from .features import CATEGORICAL, NUMERIC, TARGET, build_preprocessor  # noqa: E402


def candidates() -> dict:
    return {
        "logistic_regression": LogisticRegression(max_iter=1000),
        "random_forest": RandomForestClassifier(
            n_estimators=300, min_samples_leaf=5, random_state=0
        ),
        "gradient_boosting": GradientBoostingClassifier(random_state=0),
    }


def make_pipeline(estimator) -> Pipeline:
    return Pipeline([("prep", build_preprocessor()), ("model", estimator)])


def train_and_evaluate(
    df: pd.DataFrame, fig_dir: Path, model_path: Path, seed: int = 0
) -> dict:
    X = df[NUMERIC + CATEGORICAL]
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=seed
    )

    models = candidates()
    cv_scores = {}
    for name, est in models.items():
        scores = cross_val_score(
            make_pipeline(est), X_train, y_train, cv=5, scoring="roc_auc"
        )
        cv_scores[name] = round(float(scores.mean()), 4)

    best_name = max(cv_scores, key=cv_scores.get)
    best = make_pipeline(models[best_name]).fit(X_train, y_train)

    proba = best.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)
    test_metrics = {
        "roc_auc": round(float(roc_auc_score(y_test, proba)), 4),
        "accuracy": round(float(accuracy_score(y_test, pred)), 4),
        "f1": round(float(f1_score(y_test, pred)), 4),
    }

    fig_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5, 5))
    RocCurveDisplay.from_predictions(y_test, proba, name=best_name, ax=ax)
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_title("ROC curve (test set)")
    fig.tight_layout()
    fig.savefig(fig_dir / "roc_curve.png", dpi=120)
    plt.close(fig)

    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best, model_path)

    return {"cv_roc_auc": cv_scores, "best_model": best_name, "test": test_metrics}
