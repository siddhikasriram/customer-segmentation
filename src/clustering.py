"""K-Means customer segmentation pipeline."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler

FEATURE_COLUMNS = ["Visits", "Items Bought", "Spending"]
ID_COLUMN = "Customer ID"
DEFAULT_DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "visitItemSpend.csv"

SEGMENT_BIG_SPENDER = "Semi-loyal, big spender"
SEGMENT_LOYAL = "Loyal, moderate spender"
SEGMENT_SEMI_LOYAL = "Semi-loyal, moderate spender"
SEGMENT_INFREQUENT = "Infrequent, low spender"


@dataclass
class SegmentationResult:
    labeled_df: pd.DataFrame
    scaled_centroids: pd.DataFrame
    original_centroids: pd.DataFrame
    cluster_sizes: pd.DataFrame
    n_customers: int
    scaler: MinMaxScaler
    model: KMeans
    X_scaled: np.ndarray
    labels: np.ndarray
    segment_map: dict[int, str]


def _require_columns(df: pd.DataFrame) -> None:
    missing = [col for col in [ID_COLUMN, *FEATURE_COLUMNS] if col not in df.columns]
    if missing:
        raise ValueError(f"CSV is missing required columns: {missing}")


def load_data(path: str | Path | None = None) -> pd.DataFrame:
    csv_path = Path(path) if path is not None else DEFAULT_DATA_PATH
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {csv_path}. "
            "Place visitItemSpend.csv in the data/ folder or upload a CSV in the app."
        )
    df = pd.read_csv(csv_path)
    _require_columns(df)
    return df


def scale_features(X: np.ndarray) -> tuple[np.ndarray, MinMaxScaler]:
    scaler = MinMaxScaler()
    return scaler.fit_transform(X), scaler


def fit_kmeans(
    X: np.ndarray, n_clusters: int = 4, random_state: int = 0
) -> tuple[KMeans, np.ndarray]:
    model = KMeans(
        n_clusters=n_clusters,
        init="random",
        n_init=10,
        max_iter=300,
        tol=1e-04,
        random_state=random_state,
    )
    labels = model.fit_predict(X)
    return model, labels


def assign_segment_labels(centers: np.ndarray) -> dict[int, str]:
    """Map cluster IDs to README segment names using visits and spending.

    K-Means IDs can permute across runs, so labels are assigned from centroid
    characteristics rather than hardcoded cluster numbers.
    """
    n_clusters = centers.shape[0]
    if n_clusters != 4:
        return {i: f"Cluster {i + 1}" for i in range(n_clusters)}

    visits = centers[:, 0]
    spend = centers[:, 2]
    remaining = set(range(n_clusters))
    mapping: dict[int, str] = {}

    big_spender = int(np.argmax(spend))
    mapping[big_spender] = SEGMENT_BIG_SPENDER
    remaining.remove(big_spender)

    loyal = max(remaining, key=lambda i: visits[i])
    mapping[loyal] = SEGMENT_LOYAL
    remaining.remove(loyal)

    infrequent = min(remaining, key=lambda i: (visits[i], spend[i]))
    mapping[infrequent] = SEGMENT_INFREQUENT
    remaining.remove(infrequent)

    mapping[remaining.pop()] = SEGMENT_SEMI_LOYAL
    return mapping


def cluster_summary(
    df: pd.DataFrame,
    labels: np.ndarray,
    model: KMeans,
    scaler: MinMaxScaler,
    X_scaled: np.ndarray,
) -> SegmentationResult:
    segment_map = assign_segment_labels(model.cluster_centers_)
    labeled_df = df[[ID_COLUMN, *FEATURE_COLUMNS]].copy()
    labeled_df["Cluster"] = labels
    labeled_df["Segment"] = labeled_df["Cluster"].map(segment_map)

    scaled_centroids = pd.DataFrame(model.cluster_centers_, columns=FEATURE_COLUMNS)
    scaled_centroids.insert(0, "Cluster", range(len(scaled_centroids)))
    scaled_centroids["Segment"] = scaled_centroids["Cluster"].map(segment_map)

    original_centroids = pd.DataFrame(
        scaler.inverse_transform(model.cluster_centers_), columns=FEATURE_COLUMNS
    )
    original_centroids.insert(0, "Cluster", range(len(original_centroids)))
    original_centroids["Segment"] = original_centroids["Cluster"].map(segment_map)

    counts = Counter(labels)
    cluster_sizes = pd.DataFrame(
        {
            "Cluster": list(counts.keys()),
            "Customers": list(counts.values()),
        }
    ).sort_values("Cluster", ignore_index=True)
    cluster_sizes["Segment"] = cluster_sizes["Cluster"].map(segment_map)

    return SegmentationResult(
        labeled_df=labeled_df,
        scaled_centroids=scaled_centroids,
        original_centroids=original_centroids,
        cluster_sizes=cluster_sizes,
        n_customers=len(labeled_df),
        scaler=scaler,
        model=model,
        X_scaled=X_scaled,
        labels=labels,
        segment_map=segment_map,
    )


def run_segmentation(
    path: str | Path | None = None,
    n_clusters: int = 4,
    random_state: int = 0,
    df: pd.DataFrame | None = None,
) -> SegmentationResult:
    if df is None:
        df = load_data(path)
    else:
        _require_columns(df)
    X_raw = df[FEATURE_COLUMNS].to_numpy()
    X_scaled, scaler = scale_features(X_raw)
    model, labels = fit_kmeans(X_scaled, n_clusters=n_clusters, random_state=random_state)
    return cluster_summary(df, labels, model, scaler, X_scaled)


def save_cluster_plot(result: SegmentationResult, output_path: str | Path) -> Path:
    """Save a 3D cluster scatter PNG using a non-interactive backend."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    colors = ["blue", "green", "orange", "purple", "cyan", "magenta", "brown", "olive"]

    for cluster_id in np.unique(result.labels):
        mask = result.labels == cluster_id
        points = result.X_scaled[mask]
        ax.scatter(
            points[:, 0],
            points[:, 1],
            points[:, 2],
            s=50,
            c=colors[int(cluster_id) % len(colors)],
            label=result.segment_map[int(cluster_id)],
        )

    centers = result.model.cluster_centers_
    ax.scatter(
        centers[:, 0],
        centers[:, 1],
        centers[:, 2],
        c="red",
        marker="x",
        s=200,
        label="Cluster Centers",
    )
    ax.set_xlabel(FEATURE_COLUMNS[0])
    ax.set_ylabel(FEATURE_COLUMNS[1])
    ax.set_zlabel(FEATURE_COLUMNS[2])
    ax.set_title(f"KMeans Clustering with {len(centers)} Clusters")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    return output_path
