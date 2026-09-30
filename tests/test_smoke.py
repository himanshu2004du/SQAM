"""
test_smoke.py

End-to-end smoke test: builds a tiny synthetic git repository with a
handful of tagged releases (including deliberate bug-fix commits so
labels are non-trivial), then runs mining -> dataset build -> preprocess
-> clustering -> modeling and asserts each stage produces sane,
non-empty output. This is not meant to validate predictive *accuracy*
(the synthetic repo is far too small for that) -- it validates that the
pipeline runs end to end without errors and that shapes/labels line up.
"""
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dataset.build_dataset import build as build_dataset
from mining.mine_evolution_metrics import mine_repository_by_tags
from modeling.clustering_analysis import analyze as run_clustering
from modeling.train_model import expanding_window_cv
from preprocessing.preprocess import preprocess


def _run(cmd, cwd):
    subprocess.run(cmd, cwd=cwd, check=True, capture_output=True)


@pytest.fixture(scope="module")
def synthetic_repo(tmp_path_factory):
    repo_dir = tmp_path_factory.mktemp("demo_repo")

    _run(["git", "init", "-q"], cwd=repo_dir)
    _run(["git", "config", "user.email", "a@example.com"], cwd=repo_dir)
    _run(["git", "config", "user.name", "Author A"], cwd=repo_dir)

    src = repo_dir / "src"
    src.mkdir()

    (src / "Foo.java").write_text(
        "public class Foo {\n    public int compute(int x) {\n        return x * 2;\n    }\n}\n"
    )
    _run(["git", "add", "-A"], cwd=repo_dir)
    _run(["git", "commit", "-q", "-m", "Initial commit: add Foo"], cwd=repo_dir)
    _run(["git", "tag", "v1.0"], cwd=repo_dir)

    (src / "Bar.java").write_text(
        "public class Bar {\n    private int value;\n    public int getValue() { return value; }\n}\n"
    )
    _run(["git", "add", "-A"], cwd=repo_dir)
    _run(["git", "commit", "-q", "-m", "Add Bar data class"], cwd=repo_dir)

    (src / "Foo.java").write_text(
        "public class Foo {\n    public int compute(int x) {\n"
        "        if (x > 0) { return x * 2; } else { return 0; }\n    }\n}\n"
    )
    _run(["git", "add", "-A"], cwd=repo_dir)
    _run(["git", "commit", "-q", "-m", "fix: handle negative input bug in Foo"], cwd=repo_dir)
    _run(["git", "tag", "v1.1"], cwd=repo_dir)

    with open(src / "Bar.java", "a") as f:
        f.write("// interface tweak\n")
    _run(["git", "add", "-A"], cwd=repo_dir)
    _run(["git", "commit", "-q", "-m", "refactor Bar interface"], cwd=repo_dir)
    _run(["git", "tag", "v1.2"], cwd=repo_dir)

    (src / "Bar.java").write_text(
        "public class Bar {\n    private int value;\n    public interface Listener {}\n"
        "    public int getValue() { return value; }\n}\n// interface tweak\n"
    )
    _run(["git", "add", "-A"], cwd=repo_dir)
    _run(["git", "commit", "-q", "-m", "fix crash when Bar listener is null - defect fix"], cwd=repo_dir)
    _run(["git", "tag", "v1.3"], cwd=repo_dir)

    return repo_dir


RELEASES = ["v1.0", "v1.1", "v1.2", "v1.3"]


def test_evolution_mining_produces_expected_files(synthetic_repo):
    df = mine_repository_by_tags(str(synthetic_repo), RELEASES, file_extensions=[".java"])
    assert not df.empty
    assert set(df["file_path"]) == {"src/Foo.java", "src/Bar.java"}
    assert set(df["release"]) <= set(RELEASES)
    # MOSER_bug_fixes should be > 0 somewhere, since we made "fix:" commits.
    assert df["MOSER_bug_fixes"].sum() > 0


def test_dataset_build_attaches_sane_labels(synthetic_repo, tmp_path):
    evolution_df = mine_repository_by_tags(str(synthetic_repo), RELEASES, file_extensions=[".java"])
    evolution_path = tmp_path / "evolution.csv"
    evolution_df.to_csv(evolution_path, index=False)

    dataset_df = build_dataset(str(evolution_path), RELEASES)
    assert "defective" in dataset_df.columns
    assert dataset_df["defective"].isin([0, 1]).all()
    # The last release must be excluded (no future release to derive a label from).
    assert "v1.3" not in set(dataset_df["release"])
    # At least one row should be labeled defective, since real bug-fix commits exist.
    assert dataset_df["defective"].sum() >= 1


def test_preprocess_drops_degenerate_features(synthetic_repo, tmp_path):
    evolution_df = mine_repository_by_tags(str(synthetic_repo), RELEASES, file_extensions=[".java"])
    evolution_path = tmp_path / "evolution.csv"
    evolution_df.to_csv(evolution_path, index=False)
    dataset_df = build_dataset(str(evolution_path), RELEASES)

    X_scaled, y, ids, report = preprocess(dataset_df)
    assert len(X_scaled) == len(dataset_df)
    assert y.isin([0, 1]).all()
    # Scaling should produce (approximately) zero-mean columns.
    if len(X_scaled) > 1:
        assert X_scaled.mean().abs().max() < 1e-6 or X_scaled.shape[1] == 0


def test_clustering_and_modeling_run_without_error(synthetic_repo, tmp_path):
    evolution_df = mine_repository_by_tags(str(synthetic_repo), RELEASES, file_extensions=[".java"])
    evolution_path = tmp_path / "evolution.csv"
    evolution_df.to_csv(evolution_path, index=False)
    dataset_df = build_dataset(str(evolution_path), RELEASES)

    X_scaled, y, ids, _ = preprocess(dataset_df)
    import pandas as pd
    preprocessed_df = pd.concat([ids.reset_index(drop=True), X_scaled.reset_index(drop=True)], axis=1)

    # Clustering should not throw, even on a tiny dataset.
    result = run_clustering(X_scaled, y, dbscan_eps=3, dbscan_min_samples=2, som_grid_size=3)
    assert result.som_coords.shape[0] == len(X_scaled)

    # Modeling should not throw; on this tiny dataset there may be 0-1 usable folds.
    cv_result = expanding_window_cv(preprocessed_df, RELEASES, feature_set="combined", n_estimators=50)
    assert "summary" in cv_result
