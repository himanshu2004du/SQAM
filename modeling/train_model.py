"""
train_model.py

Model-training and validation stage of the experimental framework:
    - Random Forest defect classifier
    - Chronological expanding-window cross-validation: fold k trains on
      releases [0..k) and tests on release k, for k = 1 .. n_releases-1
    - Reports Precision, Recall, F1, AUC-ROC and AUPRC per fold, plus
      the mean +/- std across folds
    - Compares three feature-set variants (evolution-only, static-only,
      combined) to directly answer RQ1 / RQ2
    - Reports Random Forest feature importances (RQ3)

Usage (CLI):
    python -m modeling.train_model \
        --in data/processed/dataset.csv \
        --release-order v1.0 v1.1 v1.2 v1.3 \
        --out-prefix data/results/rf
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

FEATURE_SET_PREFIXES = {
    "evolution": ("MOSER_", "HASSAN_", "CHANGE_TYPE_", "ISSUE_"),
    "static": ("PMD_",),
    "coverage": ("COV_",),
}


@dataclass
class FoldResult:
    fold: int
    train_releases: List[str]
    test_release: str
    n_train: int
    n_test: int
    n_test_defective: int
    precision: float
    recall: float
    f1: float
    auc_roc: Optional[float]
    auprc: Optional[float]


def select_feature_columns(df: pd.DataFrame, feature_set: str) -> List[str]:
    all_feature_cols = [c for c in df.columns if c not in ("release", "file_path", "defective")]
    if feature_set == "combined":
        return all_feature_cols
    prefixes = FEATURE_SET_PREFIXES.get(feature_set)
    if prefixes is None:
        raise ValueError(f"Unknown feature_set '{feature_set}', expected one of "
                          f"{list(FEATURE_SET_PREFIXES) + ['combined']}")
    return [c for c in all_feature_cols if c.startswith(prefixes)]


def expanding_window_cv(
    df: pd.DataFrame,
    release_order: List[str],
    feature_set: str = "combined",
    n_estimators: int = 300,
    random_state: int = 42,
    class_weight: str = "balanced",
) -> Dict:
    """Chronological expanding-window CV: fold k trains on releases
    release_order[0:k] and tests on release_order[k]. Requires `df` to
    already contain a `defective` column (see dataset.build_dataset)."""
    feature_cols = select_feature_columns(df, feature_set)
    if not feature_cols:
        raise ValueError(f"No feature columns found for feature_set='{feature_set}'")

    available_releases = [r for r in release_order if r in set(df["release"])]
    fold_results: List[FoldResult] = []
    feature_importances = []
    trained_models = []

    for k in range(1, len(available_releases)):
        train_releases = available_releases[:k]
        test_release = available_releases[k]

        train_df = df[df["release"].isin(train_releases)]
        test_df = df[df["release"] == test_release]

        if train_df["defective"].nunique() < 2 or len(test_df) == 0:
            # Can't train (only one class) or nothing to test on -- skip fold.
            continue

        X_train, y_train = train_df[feature_cols], train_df["defective"]
        X_test, y_test = test_df[feature_cols], test_df["defective"]

        model = RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=random_state,
            class_weight=class_weight,
            n_jobs=-1,
        )
        model.fit(X_train, y_train)
        trained_models.append(model)

        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1] if len(model.classes_) > 1 else None

        auc_roc = roc_auc_score(y_test, y_proba) if y_proba is not None and y_test.nunique() > 1 else None
        auprc = average_precision_score(y_test, y_proba) if y_proba is not None and y_test.nunique() > 1 else None

        fold_results.append(FoldResult(
            fold=k,
            train_releases=train_releases,
            test_release=test_release,
            n_train=len(train_df),
            n_test=len(test_df),
            n_test_defective=int(y_test.sum()),
            precision=precision_score(y_test, y_pred, zero_division=0),
            recall=recall_score(y_test, y_pred, zero_division=0),
            f1=f1_score(y_test, y_pred, zero_division=0),
            auc_roc=auc_roc,
            auprc=auprc,
        ))
        feature_importances.append(pd.Series(model.feature_importances_, index=feature_cols))

    summary = summarize_folds(fold_results)
    mean_importance = (
        pd.concat(feature_importances, axis=1).mean(axis=1).sort_values(ascending=False)
        if feature_importances else pd.Series(dtype=float)
    )

    return {
        "feature_set": feature_set,
        "fold_results": fold_results,
        "summary": summary,
        "feature_importance": mean_importance,
        "models": trained_models,
    }


def summarize_folds(fold_results: List[FoldResult]) -> Dict[str, float]:
    if not fold_results:
        return {}
    df = pd.DataFrame([asdict(f) for f in fold_results])
    metrics = ["precision", "recall", "f1", "auc_roc", "auprc"]
    summary = {}
    for m in metrics:
        summary[f"{m}_mean"] = float(df[m].mean(skipna=True))
        summary[f"{m}_std"] = float(df[m].std(skipna=True))
    return summary


def compare_feature_sets(df: pd.DataFrame, release_order: List[str], **kwargs) -> pd.DataFrame:
    """Runs expanding-window CV for evolution-only, static-only and
    combined feature sets, to answer RQ1/RQ2 side by side."""
    rows = []
    for feature_set in ["evolution", "static", "combined"]:
        cols = select_feature_columns(df, feature_set)
        if not cols:
            continue
        result = expanding_window_cv(df, release_order, feature_set=feature_set, **kwargs)
        row = {"feature_set": feature_set, "n_features": len(cols), "n_folds": len(result["fold_results"])}
        row.update(result["summary"])
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="Train Random Forest defect model with chronological expanding-window CV.")
    parser.add_argument("--in", dest="in_path", required=True, help="Pre-processed dataset with a `defective` column.")
    parser.add_argument("--release-order", nargs="+", required=True)
    parser.add_argument("--feature-set", default="combined", choices=["evolution", "static", "coverage", "combined"])
    parser.add_argument("--compare-feature-sets", action="store_true", help="Also run evolution-only / static-only baselines for RQ1/RQ2.")
    parser.add_argument("--n-estimators", type=int, default=300)
    parser.add_argument("--out-prefix", default="data/results/rf")
    args = parser.parse_args()

    df = pd.read_csv(args.in_path)

    result = expanding_window_cv(df, args.release_order, feature_set=args.feature_set, n_estimators=args.n_estimators)

    fold_df = pd.DataFrame([asdict(f) for f in result["fold_results"]])
    fold_df.to_csv(f"{args.out_prefix}_fold_results.csv", index=False)
    result["feature_importance"].to_csv(f"{args.out_prefix}_feature_importance.csv", header=["importance"])

    print(f"\n=== Chronological expanding-window CV ({args.feature_set} features) ===")
    print(fold_df.to_string(index=False))
    print("\nMean +/- std across folds:")
    for k, v in result["summary"].items():
        print(f"  {k}: {v:.3f}")

    print("\nTop 10 most important features:")
    print(result["feature_importance"].head(10).to_string())

    if args.compare_feature_sets:
        comparison = compare_feature_sets(df, args.release_order, n_estimators=args.n_estimators)
        comparison.to_csv(f"{args.out_prefix}_feature_set_comparison.csv", index=False)
        print("\n=== Feature-set comparison (RQ1 / RQ2) ===")
        print(comparison.to_string(index=False))


if __name__ == "__main__":
    main()
