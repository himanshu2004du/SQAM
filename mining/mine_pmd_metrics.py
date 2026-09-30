"""
mine_pmd_metrics.py

Wraps the PMD static-analysis tool (https://pmd.github.io/) to produce the
PMD metric group: rule-violation counts per file, broken down by severity
(priority 1 = highest .. 5 = lowest in PMD's scale).

Requires the `pmd` CLI to be installed and on PATH (or pass --pmd-bin).
PMD is NOT installed by this repo automatically -- see README for setup.

Usage (CLI):
    python -m mining.mine_pmd_metrics \
        --source-dir /path/to/checked-out/release/src \
        --release v1.2 \
        --out data/raw/pmd_metrics_v1.2.csv
"""
from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import pandas as pd


def run_pmd(source_dir: str, pmd_bin: str = "pmd", ruleset: str = "category/java/bestpractices.xml,category/java/errorprone.xml,category/java/design.xml") -> dict:
    """Runs PMD with JSON output ('format json') and returns the parsed
    report. Falls back to a clear error message if the `pmd` binary is
    missing, since PMD is an external Java tool this project does not
    vendor."""
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
        out_path = tmp.name
    cmd = [pmd_bin, "check", "-d", source_dir, "-R", ruleset, "-f", "json", "-r", out_path, "--no-fail-on-violation"]
    try:
        subprocess.run(cmd, check=True, capture_output=True)
    except FileNotFoundError as e:
        raise RuntimeError(
            "PMD executable not found. Install PMD (https://pmd.github.io/) and either add it to "
            "PATH or pass --pmd-bin /full/path/to/pmd."
        ) from e
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"PMD failed: {e.stderr.decode(errors='ignore')}") from e

    with open(out_path) as f:
        report = json.load(f)
    Path(out_path).unlink(missing_ok=True)
    return report


def summarize_pmd_report(report: dict, source_dir: str) -> pd.DataFrame:
    """Aggregates PMD's per-violation JSON report into per-file counts,
    total and by priority (1=highest severity .. 5=lowest)."""
    file_stats: Dict[str, dict] = defaultdict(lambda: {
        "PMD_violations_total": 0,
        "PMD_priority_1": 0, "PMD_priority_2": 0, "PMD_priority_3": 0,
        "PMD_priority_4": 0, "PMD_priority_5": 0,
    })

    for file_entry in report.get("files", []):
        rel_path = str(Path(file_entry["filename"]).resolve().relative_to(Path(source_dir).resolve()))
        for violation in file_entry.get("violations", []):
            stats = file_stats[rel_path]
            stats["PMD_violations_total"] += 1
            prio = violation.get("priority", 3)
            key = f"PMD_priority_{prio}"
            if key in stats:
                stats[key] += 1

    rows = [{"file_path": path, **stats} for path, stats in file_stats.items()]
    return pd.DataFrame(rows)


def mine_pmd_for_release(source_dir: str, release: str, pmd_bin: str = "pmd") -> pd.DataFrame:
    report = run_pmd(source_dir, pmd_bin=pmd_bin)
    df = summarize_pmd_report(report, source_dir)
    df.insert(0, "release", release)
    return df


def main():
    parser = argparse.ArgumentParser(description="Mine PMD static-analysis violation metrics for one release snapshot.")
    parser.add_argument("--source-dir", required=True, help="Checked-out source tree for this release (e.g. `git checkout v1.2` first).")
    parser.add_argument("--release", required=True)
    parser.add_argument("--pmd-bin", default="pmd")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    df = mine_pmd_for_release(args.source_dir, args.release, pmd_bin=args.pmd_bin)
    df.to_csv(args.out, index=False)
    print(f"Wrote {len(df)} rows to {args.out}")


if __name__ == "__main__":
    main()
