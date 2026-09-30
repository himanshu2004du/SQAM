# AI-Based Software Quality Prediction from GitHub Evolution Metrics

Defect prediction using software evolution and development-history metrics,
implementing the pipeline described in the project synopsis (Problem
Formulation, Selected Dataset, Research Questions, Experimental Framework).

**Team:** Himanshu (119) · Yannamini Madhu Hemanth (121) · Nimesh (21)

## What this does

Given a local git clone of any (Java) project with several tagged releases,
this pipeline:

1. Mines **MOSER** (revisions, authors, churn, bug fixes, refactorings, age),
   **HASSAN** (change-entropy complexity) and **CHANGE_TYPE**
   (computation/data/interface/logic/other) metrics directly from commit
   history using [PyDriller](https://pydriller.readthedocs.io/).
2. Optionally mines **ISSUE** metrics from the GitHub REST API (issue/bug
   counts and severity, linked to files via "Fixes #N" commit messages).
3. Optionally mines **PMD** static-analysis violation counts and **JaCoCo**
   test-coverage percentages per release snapshot.
4. Merges every metric group into one table and attaches a **post-release
   defect label** per file/release (SZZ-lite: a file is "defective" at
   release *r* if it receives a bug-fixing commit before release *r+1*).
5. Pre-processes the feature matrix (drops low-variance / highly-correlated
   features, imputes, scales).
6. Runs **DBSCAN** clustering and trains a **Self-Organizing Map (SOM)** to
   examine whether defective/non-defective components separate in feature
   space (RQ4).
7. Trains a **Random Forest** classifier and evaluates it with
   **chronological expanding-window cross-validation** (train on releases
   `[0..k)`, test on release `k`), reporting Precision, Recall, F1, AUC-ROC
   and AUPRC per fold, plus a feature-set comparison (evolution-only vs.
   static-only vs. combined) for RQ1/RQ2, and feature importances for RQ3.

## Project layout

```
defect-prediction/
├── mining/
│   ├── mine_evolution_metrics.py   # MOSER + HASSAN + CHANGE_TYPE (PyDriller)
│   ├── mine_issue_metrics.py       # ISSUE (GitHub REST API)
│   ├── mine_pmd_metrics.py         # PMD static-analysis violations
│   └── mine_coverage_metrics.py    # JaCoCo test-coverage parser
├── dataset/
│   └── build_dataset.py            # merge metric groups + defect labels
├── preprocessing/
│   └── preprocess.py               # variance/correlation pruning + scaling
├── modeling/
│   ├── clustering_analysis.py      # DBSCAN + SOM (RQ4)
│   └── train_model.py              # Random Forest + expanding-window CV
├── config/
│   └── pipeline_config.example.yaml
├── data/
│   ├── raw/                        # mined metrics, one CSV per group
│   ├── processed/                  # merged + labeled + pre-processed data
│   └── results/                    # fold results, feature importance, clustering
├── run_pipeline.py                 # end-to-end orchestrator (stages 1-8)
├── requirements.txt
└── tests/
    └── test_smoke.py               # runs the pipeline on a tiny synthetic repo
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Optional external tools, only needed for the corresponding metric group:

- **ISSUE metrics** — a GitHub personal access token:
  `export GITHUB_TOKEN=ghp_xxx`
- **PMD metrics** — install [PMD](https://pmd.github.io/) and either add it
  to `PATH` or pass `--pmd-bin /full/path/to/pmd`.
- **Test-Coverage metrics** — a JaCoCo XML report (`report.xml`), generated
  by `mvn jacoco:report` / `gradle jacocoTestReport` on each release
  snapshot you want coverage for.

## Quick start

```bash
git clone <the-project-you-want-to-study> /tmp/target_repo
cd defect-prediction

# 1-2, 5-8: evolution + issue metrics, merge, preprocess, cluster, model
python run_pipeline.py \
    --repo /tmp/target_repo \
    --release-tags v1.0.0 v1.1.0 v1.2.0 v1.3.0 \
    --owner <github-owner> --repo-name <github-repo> \
    --out-dir data
```

This writes intermediate CSVs to `data/raw/`, the merged & labeled dataset
to `data/processed/`, and fold-by-fold model results + feature importances
to `data/results/`.

### Adding PMD / coverage metrics (per-release snapshots)

PMD and JaCoCo both need a **checked-out snapshot** of the source tree at
each release, so they run outside the main history-walking loop:

```bash
for tag in v1.0.0 v1.1.0 v1.2.0 v1.3.0; do
    git -C /tmp/target_repo checkout -q "$tag"
    python -m mining.mine_pmd_metrics \
        --source-dir /tmp/target_repo/src/main/java \
        --release "$tag" \
        --out "data/raw/pmd_metrics_${tag}.csv"
done
git -C /tmp/target_repo checkout -q main   # restore the working tree

python -m dataset.build_dataset \
    --evolution data/raw/evolution_metrics.csv \
    --issues data/raw/issue_metrics.csv \
    --pmd "data/raw/pmd_metrics_*.csv" \
    --coverage "data/raw/coverage_metrics_*.csv" \
    --release-order v1.0.0 v1.1.0 v1.2.0 v1.3.0 \
    --out data/processed/dataset.csv
```

### Running stages individually

```bash
python -m preprocessing.preprocess --in data/processed/dataset.csv --out data/processed/dataset_preprocessed.csv
python -m modeling.clustering_analysis --in data/processed/dataset_preprocessed.csv
python -m modeling.train_model --in data/processed/dataset_preprocessed.csv \
    --release-order v1.0.0 v1.1.0 v1.2.0 v1.3.0 --compare-feature-sets
```

## Testing

A smoke test builds a small synthetic git repository with a handful of
tagged commits (including deliberate "fix:" bug commits) and runs the full
pipeline against it, asserting every stage produces non-empty, sane output:

```bash
pytest tests/test_smoke.py -v
```

## Mapping back to the synopsis

| Synopsis section | Where it lives in this repo |
|---|---|
| Problem Formulation | `dataset/build_dataset.py` (label definition), docstrings throughout |
| Selected Dataset | `mining/*.py` (what's mined), `README.md` Quick Start |
| RQ1 / RQ2 | `modeling/train_model.py::compare_feature_sets` |
| RQ3 | `modeling/train_model.py` feature-importance output |
| RQ4 | `modeling/clustering_analysis.py` |
| RQ5 | `modeling/train_model.py::expanding_window_cv` |
| Experimental Framework stages 1-9 | `run_pipeline.py` (one print block per stage) |

## Notes and limitations

- `CHANGE_TYPE` classification uses lightweight regex heuristics over diff
  text rather than a full AST-diff classifier — documented as a pragmatic
  simplification in `mining/mine_evolution_metrics.py`.
- Defect labeling is SZZ-*lite*: it uses commit-message keyword matching
  (`fix`, `bug`, `defect`, ...) rather than full SZZ blame-based bug
  introduction tracing. Swapping in a proper SZZ implementation (e.g.
  [PyDriller's `RepositoryMining` + a dedicated SZZ library](https://github.com/apache/pydriller))
  is a natural next step.
- On very small datasets (a handful of files/releases, as in the smoke
  test) metrics like AUC/AUPRC and cross-fold std can be `NaN` or trivial —
  this is expected; the pipeline is meant to be pointed at a real
  multi-release open-source project for meaningful results.
