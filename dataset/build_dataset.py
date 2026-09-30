"""
build_dataset.py

Merges the outputs of the four mining scripts (evolution/MOSER+HASSAN+
CHANGE_TYPE, ISSUE, PMD, Test-Coverage) into a single wide table keyed by
(release, file_path), and attaches a binary defect label per the
problem formulation in the synopsis:

    A component is labeled DEFECTIVE at release r if it receives at least
    one bug-fixing commit between release r (exclusive) and release r+1
    (inclusive) -- i.e. a *post-release* defect, the label the model is
    meant to predict using only information available as of release r.

This is a lightweight SZZ-style scheme: it reuses the cumulative
MOSER_bug_fixes count already computed per release by the evolution miner
and labels a file defective when that count increases from one release to
the next.

Usage (CLI):
    python -m dataset.build_dataset \
        --evolution data/raw/evolution_metrics.csv \
        --issues data/raw/issue_metrics.csv \
        --pmd data/raw/pmd_metrics_*.csv \
        --coverage data/raw/coverage_metrics_*.csv \
        --release-order v1.0 v1.1 v1.2 v1.3 \
        --out data/processed/dataset.csv
"""
from __future__ import annotations

import argparse
import glob
from typing import List, Optional

import pandas as pd


def _load_and_concat(paths: List[str]) -> Optional[pd.DataFrame]:
    frames = []
    for pattern in paths:
        for path in glob.glob(pattern):
            frames.append(pd.read_csv(path))
    if not frames:
        return None
    return pd.concat(frames, ignore_index=True)


def merge_feature_groups(
    evolution_path: str,
    issues_path: Optional[str] = None,
    pmd_paths: Optional[List[str]] = None,
    coverage_paths: Optional[List[str]] = None,
) -> pd.DataFrame:
    df = pd.read_csv(evolution_path)

    if issues_path:
        issues_df = pd.read_csv(issues_path)
        df = df.merge(issues_df, on=["release", "file_path"], how="left")

    if pmd_paths:
        pmd_df = _load_and_concat(pmd_paths)
        if pmd_df is not None:
            df = df.merge(pmd_df, on=["release", "file_path"], how="left")

    if coverage_paths:
        cov_df = _load_and_concat(coverage_paths)
        if cov_df is not None:
            df = df.merge(cov_df, on=["release", "file_path"], how="left")

    # Any file missing from the ISSUE/PMD/coverage mining (e.g., no linked
    # issues, or file not present in that release's coverage report) is
    # filled with 0 rather than dropped -- absence of signal is informative.
    numeric_cols = df.select_dtypes(include="number").columns
    df[numeric_cols] = df[numeric_cols].fillna(0)
    return df


def attach_defect_labels(df: pd.DataFrame, release_order: List[str]) -> pd.DataFrame:
    """Forward-looking, chronology-respecting labeling: a file at release r
    is DEFECTIVE if its cumulative MOSER_bug_fixes count increases by the
    time of release r+1. The last release in `release_order` has no
    "future" release to look ahead to, so it is dropped (no ground truth
    available for it) -- this mirrors real deployment, where the most
    recent release is exactly the one you don't have labels for yet."""
    if "MOSER_bug_fixes" not in df.columns:
        raise ValueError("evolution metrics must include MOSER_bug_fixes to build labels")

    pivot = df.pivot_table(index="file_path", columns="release", values="MOSER_bug_fixes", aggfunc="max")
    labeled_frames = []

    for i, release in enumerate(release_order[:-1]):
        next_release = release_order[i + 1]
        if release not in pivot.columns or next_release not in pivot.columns:
            continue
        delta = (pivot[next_release].fillna(0) - pivot[release].fillna(0)).clip(lower=0)
        labels = (delta > 0).astype(int)

        release_df = df[df["release"] == release].copy()
        release_df["defective"] = release_df["file_path"].map(labels).fillna(0).astype(int)
        labeled_frames.append(release_df)

    if not labeled_frames:
        raise ValueError("No consecutive release pairs found -- check --release-order matches mined releases.")

    result = pd.concat(labeled_frames, ignore_index=True)
    return result


def build(
    evolution_path: str,
    release_order: List[str],
    issues_path: Optional[str] = None,
    pmd_paths: Optional[List[str]] = None,
    coverage_paths: Optional[List[str]] = None,
) -> pd.DataFrame:
    merged = merge_feature_groups(evolution_path, issues_path, pmd_paths, coverage_paths)
    labeled = attach_defect_labels(merged, release_order)
    return labeled


def main():
    parser = argparse.ArgumentParser(description="Merge metric groups and attach post-release defect labels.")
    parser.add_argument("--evolution", required=True)
    parser.add_argument("--issues")
    parser.add_argument("--pmd", nargs="*", default=[])
    parser.add_argument("--coverage", nargs="*", default=[])
    parser.add_argument("--release-order", nargs="+", required=True, help="Releases in chronological order.")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    df = build(args.evolution, args.release_order, args.issues, args.pmd, args.coverage)
    df.to_csv(args.out, index=False)
    n_defective = int(df["defective"].sum())
    print(f"Wrote {len(df)} rows across {df['release'].nunique()} releases "
          f"({n_defective} defective, {n_defective / len(df):.1%}) to {args.out}")


if __name__ == "__main__":
    main()
