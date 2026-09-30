"""
run_pipeline.py

End-to-end orchestration of the experimental framework, stage by stage:

    1. Mine evolution metrics (MOSER, HASSAN, CHANGE_TYPE)  [required]
    2. Mine ISSUE metrics from the GitHub API                [optional]
    3. Mine PMD static-analysis metrics per release snapshot [optional]
    4. Mine JaCoCo test-coverage metrics per release         [optional]
    5. Merge feature groups + attach post-release defect labels
    6. Pre-process (variance/correlation pruning + scaling)
    7. DBSCAN + SOM clustering analysis (RQ4)
    8. Random Forest + chronological expanding-window CV (RQ1, RQ2, RQ3, RQ5)

Steps 2-4 are optional because they depend on external tools (GitHub API
token, a local PMD install, a JaCoCo report) that a grader/reviewer may not
have configured -- the pipeline degrades gracefully to evolution-metrics-only
if they're skipped. See README.md for full setup instructions and for the
config-driven version of this script (config/pipeline_config.yaml).

Usage:
    python run_pipeline.py \
        --repo /path/to/local/clone \
        --release-tags v1.0 v1.1 v1.2 v1.3 \
        --owner apache --repo-name somelib \
        --out-dir data
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd

from dataset.build_dataset import build as build_dataset
from mining.mine_evolution_metrics import mine_repository_by_tags
from modeling.clustering_analysis import analyze as run_clustering
from modeling.train_model import compare_feature_sets, expanding_window_cv
from preprocessing.preprocess import preprocess


def main():
    parser = argparse.ArgumentParser(description="Run the full defect-prediction pipeline end to end.")
    parser.add_argument("--repo", required=True, help="Path to a local git clone.")
    parser.add_argument("--release-tags", nargs="+", required=True, help="Chronologically ordered release tags.")
    parser.add_argument("--ext", nargs="+", default=[".java"])
    parser.add_argument("--owner", help="GitHub owner/org, for ISSUE mining (optional).")
    parser.add_argument("--repo-name", help="GitHub repo name, for ISSUE mining (optional).")
    parser.add_argument("--skip-issues", action="store_true")
    parser.add_argument("--out-dir", default="data")
    args = parser.parse_args()

    raw_dir = Path(args.out_dir) / "raw"
    processed_dir = Path(args.out_dir) / "processed"
    results_dir = Path(args.out_dir) / "results"
    for d in (raw_dir, processed_dir, results_dir):
        d.mkdir(parents=True, exist_ok=True)

    print("== Stage 1/8: mining evolution metrics (MOSER / HASSAN / CHANGE_TYPE) ==")
    evolution_df = mine_repository_by_tags(args.repo, args.release_tags, file_extensions=args.ext)
    evolution_path = raw_dir / "evolution_metrics.csv"
    evolution_df.to_csv(evolution_path, index=False)
    print(f"  -> {len(evolution_df)} rows, {evolution_df['file_path'].nunique() if len(evolution_df) else 0} files")

    issues_path = None
    if args.owner and args.repo_name and not args.skip_issues and os.environ.get("GITHUB_TOKEN"):
        print("== Stage 2/8: mining ISSUE metrics from GitHub API ==")
        from mining.mine_issue_metrics import build_issue_metrics
        issues_df = build_issue_metrics(args.owner, args.repo_name, args.repo, args.release_tags)
        issues_path = raw_dir / "issue_metrics.csv"
        issues_df.to_csv(issues_path, index=False)
        print(f"  -> {len(issues_df)} rows")
    else:
        print("== Stage 2/8: skipped (no --owner/--repo-name/GITHUB_TOKEN, or --skip-issues) ==")

    print("== Stage 3-4/8: PMD / coverage mining skipped in this quick-start run ==")
    print("   (run mining.mine_pmd_metrics / mining.mine_coverage_metrics separately per release")
    print("    snapshot, then pass their CSVs to dataset.build_dataset --pmd / --coverage)")

    print("== Stage 5/8: merging feature groups + attaching defect labels ==")
    dataset_df = build_dataset(
        evolution_path=str(evolution_path),
        release_order=args.release_tags,
        issues_path=str(issues_path) if issues_path else None,
    )
    dataset_path = processed_dir / "dataset.csv"
    dataset_df.to_csv(dataset_path, index=False)
    n_defective = int(dataset_df["defective"].sum())
    print(f"  -> {len(dataset_df)} rows, {n_defective} defective ({n_defective / max(len(dataset_df), 1):.1%})")

    print("== Stage 6/8: pre-processing (variance/correlation pruning + scaling) ==")
    X_scaled, y, ids, report = preprocess(dataset_df)
    preprocessed_df = pd.concat([ids.reset_index(drop=True), X_scaled.reset_index(drop=True)], axis=1)
    preprocessed_path = processed_dir / "dataset_preprocessed.csv"
    preprocessed_df.to_csv(preprocessed_path, index=False)
    print(f"  -> kept {len(report.kept_features)} features "
          f"(dropped {len(report.dropped_low_variance)} low-variance, {len(report.dropped_correlated)} correlated)")

    print("== Stage 7/8: DBSCAN + SOM clustering analysis (RQ4) ==")
    try:
        clustering_result = run_clustering(X_scaled, y)
        print(f"  -> DBSCAN silhouette: {clustering_result.dbscan_silhouette:.3f}, "
              f"SOM separation score: {clustering_result.som_separation_score:.3f}")
    except Exception as e:
        print(f"  -> clustering skipped ({e})")

    print("== Stage 8/8: Random Forest + chronological expanding-window CV (RQ1,2,3,5) ==")
    try:
        result = expanding_window_cv(preprocessed_df, args.release_tags, feature_set="combined")
        for k, v in result["summary"].items():
            print(f"  {k}: {v:.3f}")
        result["feature_importance"].head(10).to_csv(results_dir / "top_feature_importance.csv", header=["importance"])
        comparison = compare_feature_sets(preprocessed_df, args.release_tags)
        comparison.to_csv(results_dir / "feature_set_comparison.csv", index=False)
        print("\nFeature-set comparison (RQ1/RQ2):")
        print(comparison.to_string(index=False))
    except Exception as e:
        print(f"  -> modeling skipped ({e})")

    print(f"\nDone. Outputs under: {args.out_dir}/")


if __name__ == "__main__":
    main()
