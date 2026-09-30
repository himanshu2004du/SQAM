"""
preprocess.py

Implements the "Pre-processing" stage from the experimental framework:
    - remove near-constant / low-variance features
    - remove highly-correlated (redundant) features
    - impute any remaining missing values
    - scale features (z-score) before clustering / model training

Kept independent of any specific model so the same cleaned matrix feeds
both the DBSCAN/SOM clustering analysis and the Random Forest classifier.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

ID_COLUMNS = ["release", "file_path", "defective"]


@dataclass
class PreprocessingReport:
    dropped_low_variance: List[str]
    dropped_correlated: List[str]
    kept_features: List[str]


def split_features_labels(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Returns (feature_matrix, label_series, id_columns_df)."""
    feature_cols = [c for c in df.columns if c not in ID_COLUMNS]
    X = df[feature_cols].copy()
    y = df["defective"].copy() if "defective" in df.columns else None
    ids = df[[c for c in ID_COLUMNS if c in df.columns]].copy()
    return X, y, ids


def drop_low_variance(X: pd.DataFrame, threshold: float = 1e-4) -> Tuple[pd.DataFrame, List[str]]:
    variances = X.var(numeric_only=True)
    to_drop = variances[variances <= threshold].index.tolist()
    return X.drop(columns=to_drop), to_drop


def drop_highly_correlated(X: pd.DataFrame, threshold: float = 0.95) -> Tuple[pd.DataFrame, List[str]]:
    """Greedy pairwise-correlation pruning: for each pair of features with
    |corr| above `threshold`, drop the second one (arbitrary but
    deterministic given column order)."""
    corr = X.corr(numeric_only=True).abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    to_drop = [col for col in upper.columns if any(upper[col] > threshold)]
    return X.drop(columns=to_drop), to_drop


def preprocess(
    df: pd.DataFrame,
    variance_threshold: float = 1e-4,
    correlation_threshold: float = 0.95,
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, PreprocessingReport]:
    """
    Full pre-processing pipeline. Returns:
        X_scaled : DataFrame of scaled, pruned features (same row order as df)
        y         : label Series (or None if `defective` column absent)
        ids       : release/file_path/defective passthrough columns
        report    : which features were dropped and why
    """
    X, y, ids = split_features_labels(df)

    imputer = SimpleImputer(strategy="median")
    X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns, index=X.index)

    X_var, dropped_var = drop_low_variance(X_imputed, variance_threshold)
    X_corr, dropped_corr = drop_highly_correlated(X_var, correlation_threshold)

    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X_corr), columns=X_corr.columns, index=X_corr.index)

    report = PreprocessingReport(
        dropped_low_variance=dropped_var,
        dropped_correlated=dropped_corr,
        kept_features=list(X_scaled.columns),
    )
    return X_scaled, y, ids, report


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Pre-process the merged dataset (variance/correlation pruning + scaling).")
    parser.add_argument("--in", dest="in_path", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--variance-threshold", type=float, default=1e-4)
    parser.add_argument("--correlation-threshold", type=float, default=0.95)
    args = parser.parse_args()

    df = pd.read_csv(args.in_path)
    X_scaled, y, ids, report = preprocess(df, args.variance_threshold, args.correlation_threshold)

    out_df = pd.concat([ids.reset_index(drop=True), X_scaled.reset_index(drop=True)], axis=1)
    out_df.to_csv(args.out, index=False)

    print(f"Kept {len(report.kept_features)} features.")
    print(f"Dropped (low variance): {report.dropped_low_variance}")
    print(f"Dropped (correlated):   {report.dropped_correlated}")
    print(f"Wrote {len(out_df)} rows to {args.out}")


if __name__ == "__main__":
    main()
