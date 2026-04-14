from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Creates heatmap

def plot_correlation_heatmap(df: pd.DataFrame, title: str = "Correlation Heatmap"):
    numeric = df.select_dtypes(include="number")
    corr = numeric.corr()
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111)
    sns.heatmap(corr, annot=True, fmt=".2f", ax=ax)
    ax.set_title(title)
    return fig

# Creates scatter graph

def plot_scatter(df: pd.DataFrame, x: str, y: str, title: Optional[str] = None):
    fig = plt.figure(figsize=(7, 5))
    ax = fig.add_subplot(111)
    sns.scatterplot(data=df, x=x, y=y, ax=ax)
    ax.set_title(title or f"{y} vs {x}")
    return fig


def save_fig(fig, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")