"""Exploratory data analysis: summary stats and figures."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402


def summarize(df: pd.DataFrame) -> dict:
    return {
        "rows": len(df),
        "churn_rate": round(float(df["churn"].mean()), 4),
        "missing_values": int(df.isna().sum().sum()),
        "churn_by_contract": df.groupby("contract")["churn"].mean().round(3).to_dict(),
    }


def _save(fig, path: Path) -> Path:
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def make_figures(df: pd.DataFrame, out_dir: Path) -> list:
    out_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")
    paths = []

    fig, ax = plt.subplots(figsize=(7, 4))
    rates = df.groupby("contract")["churn"].mean().sort_values() * 100
    rates.plot.barh(ax=ax, color="#4C72B0")
    ax.set_xlabel("Churn rate (%)")
    ax.set_ylabel("")
    ax.set_title("Churn rate by contract type")
    paths.append(_save(fig, out_dir / "churn_by_contract.png"))

    fig, ax = plt.subplots(figsize=(7, 4))
    sns.histplot(data=df, x="tenure_months", hue="churn", bins=36, multiple="stack", ax=ax)
    ax.set_title("Tenure distribution by churn")
    paths.append(_save(fig, out_dir / "tenure_by_churn.png"))

    fig, ax = plt.subplots(figsize=(6, 5))
    corr = df.select_dtypes("number").corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="vlag", center=0, ax=ax)
    ax.set_title("Numeric feature correlations")
    paths.append(_save(fig, out_dir / "correlations.png"))
    return paths
