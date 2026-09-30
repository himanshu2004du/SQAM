"""
clustering_analysis.py

Exploratory-analysis stage of the experimental framework: applies DBSCAN
and a Self-Organizing Map (SOM) to the pre-processed evolution-metric
feature space, to examine whether defective and non-defective components
form separable regions (RQ4).

Outputs:
    - cluster assignments (DBSCAN) per row, with a purity score per cluster
      (fraction of defective components) so separability can be read off
      directly
    - a trained SOM plus each row's winning-neuron coordinates, and a
      simple separation score (Silhouette-style contrast between defective
      / non-defective mean positions on the map)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd
from minisom import MiniSom
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score


@dataclass
class ClusteringResult:
    dbscan_labels: np.ndarray
    dbscan_cluster_purity: pd.DataFrame
    dbscan_silhouette: float
    som_coords: np.ndarray
    som_separation_score: float


def run_dbscan(X: pd.DataFrame, y: pd.Series, eps: float = 1.5, min_samples: int = 5) -> Tuple[np.ndarray, pd.DataFrame, float]:
    db = DBSCAN(eps=eps, min_samples=min_samples)
    labels = db.fit_predict(X.values)

    df = pd.DataFrame({"cluster": labels, "defective": y.values})
    purity = (
        df[df["cluster"] != -1]
        .groupby("cluster")["defective"]
        .agg(n="count", defect_rate="mean")
        .sort_values("defect_rate", ascending=False)
    )

    non_noise_mask = labels != -1
    if non_noise_mask.sum() > 1 and len(set(labels[non_noise_mask])) > 1:
        sil = silhouette_score(X.values[non_noise_mask], labels[non_noise_mask])
    else:
        sil = float("nan")

    return labels, purity, sil


def run_som(X: pd.DataFrame, y: pd.Series, grid_size: int = 10, num_iterations: int = 2000, seed: int = 42) -> Tuple[np.ndarray, float]:
    """Trains a SOM on the feature matrix and returns each sample's winning
    neuron coordinates plus a simple separation score: the Euclidean
    distance, in grid units, between the centroid of defective components'
    winning neurons and the centroid of non-defective ones. A larger score
    means the two classes land in more distinct regions of the map."""
    n_features = X.shape[1]
    som = MiniSom(
        x=grid_size, y=grid_size, input_len=n_features,
        sigma=1.0, learning_rate=0.5, random_seed=seed,
    )
    data = X.values
    som.random_weights_init(data)
    som.train_random(data, num_iterations)

    coords = np.array([som.winner(row) for row in data])

    defective_mask = y.values.astype(bool)
    if defective_mask.any() and (~defective_mask).any():
        centroid_defective = coords[defective_mask].mean(axis=0)
        centroid_clean = coords[~defective_mask].mean(axis=0)
        separation_score = float(np.linalg.norm(centroid_defective - centroid_clean))
    else:
        separation_score = float("nan")

    return coords, separation_score


def analyze(X: pd.DataFrame, y: pd.Series, dbscan_eps: float = 1.5, dbscan_min_samples: int = 5, som_grid_size: int = 10) -> ClusteringResult:
    if y is None:
        raise ValueError("clustering analysis needs defect labels to compute purity / separation scores")

    dbscan_labels, purity, sil = run_dbscan(X, y, eps=dbscan_eps, min_samples=dbscan_min_samples)
    som_coords, som_sep = run_som(X, y, grid_size=som_grid_size)

    return ClusteringResult(
        dbscan_labels=dbscan_labels,
        dbscan_cluster_purity=purity,
        dbscan_silhouette=sil,
        som_coords=som_coords,
        som_separation_score=som_sep,
    )


def main():
    import argparse

    parser = argparse.ArgumentParser(description="DBSCAN + SOM exploratory clustering analysis (RQ4).")
    parser.add_argument("--in", dest="in_path", required=True, help="Pre-processed dataset (output of preprocessing/preprocess.py).")
    parser.add_argument("--eps", type=float, default=1.5)
    parser.add_argument("--min-samples", type=int, default=5)
    parser.add_argument("--som-grid-size", type=int, default=10)
    parser.add_argument("--out-prefix", default="data/results/clustering")
    args = parser.parse_args()

    df = pd.read_csv(args.in_path)
    feature_cols = [c for c in df.columns if c not in ("release", "file_path", "defective")]
    X = df[feature_cols]
    y = df["defective"]

    result = analyze(X, y, dbscan_eps=args.eps, dbscan_min_samples=args.min_samples, som_grid_size=args.som_grid_size)

    out = df[["release", "file_path", "defective"]].copy()
    out["dbscan_cluster"] = result.dbscan_labels
    out["som_x"] = result.som_coords[:, 0]
    out["som_y"] = result.som_coords[:, 1]
    out.to_csv(f"{args.out_prefix}_assignments.csv", index=False)
    result.dbscan_cluster_purity.to_csv(f"{args.out_prefix}_dbscan_purity.csv")

    print(f"DBSCAN silhouette score: {result.dbscan_silhouette:.3f}")
    print(f"SOM defective/non-defective separation score: {result.som_separation_score:.3f}")
    print("\nDBSCAN cluster purity (defect rate per cluster):")
    print(result.dbscan_cluster_purity)


if __name__ == "__main__":
    main()
