"""
mine_evolution_metrics.py

Mines MOSER, HASSAN and CHANGE_TYPE metric groups for every source file that
exists at a given release (git tag / commit), by walking the commit history
of a local git repository with PyDriller.

MOSER group   : revisions, n_authors, churn (added/deleted), bug_fixes,
                refactorings, age_days
HASSAN group  : change_entropy (Shannon entropy of a file's contribution to
                each commit's total churn -- a simplified version of
                Hassan's (2009) complexity-of-changes metric)
CHANGE_TYPE   : counts of commits touching the file classified as
                computation / data / interface / logic / other, using a
                lightweight keyword + diff heuristic (a pragmatic stand-in
                for a full AST-diff classifier).

Usage (CLI):
    python -m mining.mine_evolution_metrics \
        --repo /path/to/local/repo \
        --release-tags v1.0 v1.1 v1.2 \
        --out data/raw/evolution_metrics.csv
"""
from __future__ import annotations

import argparse
import math
import re
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional

import pandas as pd
from pydriller import Repository

BUGFIX_PATTERN = re.compile(
    r"\b(fix|fixes|fixed|bug|defect|error|issue|patch|crash|fail(s|ure)?)\b",
    re.IGNORECASE,
)
REFACTOR_PATTERN = re.compile(
    r"\b(refactor|rename|cleanup|clean up|restructure|reorganize|simplify)\b",
    re.IGNORECASE,
)

# Very lightweight change-type keyword heuristics, applied to changed line
# content. This is intentionally simple (regex over diff text) -- it stands
# in for a full AST-level change classifier such as the one used in the
# literature (Hindle et al., Hassan et al.).
CHANGE_TYPE_PATTERNS = {
    "interface": re.compile(r"\b(public|private|protected)\s+\w+\s+\w+\s*\(|interface\s+\w+|extends|implements"),
    "data": re.compile(r"\b(int|float|double|String|boolean|long|List|Map|Set)\s+\w+\s*[=;]|class\s+\w+"),
    "logic": re.compile(r"\b(if|else|for|while|switch|case|try|catch)\b"),
    "computation": re.compile(r"[+\-*/%]=?\s|\bMath\.\w+|\breturn\b"),
}


@dataclass
class FileEvolutionState:
    file_path: str
    revisions: int = 0
    authors: set = field(default_factory=set)
    lines_added: int = 0
    lines_deleted: int = 0
    bug_fixes: int = 0
    refactorings: int = 0
    first_seen: Optional[datetime] = None
    last_seen: Optional[datetime] = None
    per_commit_churn: List[int] = field(default_factory=list)
    change_type_counts: Dict[str, int] = field(
        default_factory=lambda: {"computation": 0, "data": 0, "interface": 0, "logic": 0, "other": 0}
    )

    def as_row(self, release: str, as_of: datetime) -> dict:
        age_days = (as_of - self.first_seen).days if self.first_seen else 0
        entropy = _shannon_entropy(self.per_commit_churn)
        row = {
            "release": release,
            "file_path": self.file_path,
            "MOSER_revisions": self.revisions,
            "MOSER_n_authors": len(self.authors),
            "MOSER_lines_added": self.lines_added,
            "MOSER_lines_deleted": self.lines_deleted,
            "MOSER_churn": self.lines_added + self.lines_deleted,
            "MOSER_bug_fixes": self.bug_fixes,
            "MOSER_refactorings": self.refactorings,
            "MOSER_age_days": age_days,
            "HASSAN_change_entropy": entropy,
        }
        for k, v in self.change_type_counts.items():
            row[f"CHANGE_TYPE_{k}"] = v
        return row


def _shannon_entropy(values: List[int]) -> float:
    """Shannon entropy of a file's per-commit churn distribution -- a
    simplified proxy for Hassan's change-complexity metric: files whose
    change effort is spread unevenly / unpredictably across commits get a
    higher score."""
    total = sum(values)
    if total == 0 or len(values) <= 1:
        return 0.0
    entropy = 0.0
    for v in values:
        if v <= 0:
            continue
        p = v / total
        entropy -= p * math.log2(p)
    return entropy


def _classify_change_type(diff_text: str) -> str:
    """Classify a single file-modification's diff text into one of the
    CHANGE_TYPE buckets. Falls back to 'other' when nothing matches."""
    if not diff_text:
        return "other"
    for change_type, pattern in CHANGE_TYPE_PATTERNS.items():
        if pattern.search(diff_text):
            return change_type
    return "other"


def mine_repository(
    repo_path: str,
    release_tags: List[str],
    file_extensions: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Walk the full commit history once, and emit one FileEvolutionState
    snapshot per (release, file) pair, where a "release" boundary is a git
    tag name in `release_tags` (interpreted in the order given -- each
    release's metrics are computed from all commits up to and including
    that tag).

    Returns a long-format DataFrame with one row per (release, file).
    """
    file_extensions = file_extensions or [".java"]
    tag_set = set(release_tags)
    state: Dict[str, FileEvolutionState] = {}
    rows: List[dict] = []
    remaining_tags = list(release_tags)

    for commit in Repository(repo_path, order="reverse", only_no_merge=True).traverse_commits():
        is_bugfix = bool(BUGFIX_PATTERN.search(commit.msg or ""))
        is_refactor = bool(REFACTOR_PATTERN.search(commit.msg or ""))

        for mod in commit.modified_files:
            path = mod.new_path or mod.old_path
            if path is None:
                continue
            if file_extensions and not any(path.endswith(ext) for ext in file_extensions):
                continue

            fs = state.setdefault(path, FileEvolutionState(file_path=path))
            fs.revisions += 1
            fs.authors.add(commit.author.email or commit.author.name)
            fs.lines_added += mod.added_lines or 0
            fs.lines_deleted += mod.deleted_lines or 0
            fs.per_commit_churn.append((mod.added_lines or 0) + (mod.deleted_lines or 0))
            if is_bugfix:
                fs.bug_fixes += 1
            if is_refactor:
                fs.refactorings += 1
            if fs.first_seen is None:
                fs.first_seen = commit.committer_date
            fs.last_seen = commit.committer_date

            ctype = _classify_change_type(mod.diff or "")
            fs.change_type_counts[ctype] += 1

        # If this commit is tagged as one of the requested releases, snapshot
        # the current state of every known file into `rows`.
        matched_tag = _commit_matches_tag(commit, tag_set)
        if matched_tag and matched_tag in remaining_tags:
            for fs in state.values():
                rows.append(fs.as_row(release=matched_tag, as_of=commit.committer_date))
            remaining_tags.remove(matched_tag)

    return pd.DataFrame(rows)


def _commit_matches_tag(commit, tag_set) -> Optional[str]:
    """PyDriller doesn't map commits -> tags directly in this streaming mode;
    this helper checks the commit's `hash` against tags resolved up-front by
    the caller via `resolve_tag_commits`. Kept as a small seam so the mining
    loop stays readable; see `resolve_tag_commits` below."""
    tag = getattr(commit, "_resolved_tag", None)
    return tag if tag in tag_set else None


def resolve_tag_commits(repo_path: str, release_tags: List[str]) -> Dict[str, str]:
    """Resolve each release tag to the commit hash it points at, using git
    directly (PyDriller doesn't expose tag->commit resolution)."""
    import subprocess

    tag_to_commit = {}
    for tag in release_tags:
        try:
            out = subprocess.check_output(
                ["git", "-C", repo_path, "rev-list", "-n", "1", tag],
                stderr=subprocess.DEVNULL,
            )
            tag_to_commit[tag] = out.decode().strip()
        except subprocess.CalledProcessError:
            print(f"[warn] could not resolve tag '{tag}' in {repo_path}, skipping")
    return tag_to_commit


def mine_repository_by_tags(repo_path: str, release_tags: List[str], file_extensions=None) -> pd.DataFrame:
    """Public entry point: resolves tags to commit hashes first, then mines."""
    tag_to_commit = resolve_tag_commits(repo_path, release_tags)
    commit_to_tag = {v: k for k, v in tag_to_commit.items()}

    file_extensions = file_extensions or [".java"]
    state: Dict[str, FileEvolutionState] = {}
    rows: List[dict] = []

    for commit in Repository(repo_path, only_no_merge=True).traverse_commits():
        is_bugfix = bool(BUGFIX_PATTERN.search(commit.msg or ""))
        is_refactor = bool(REFACTOR_PATTERN.search(commit.msg or ""))

        for mod in commit.modified_files:
            path = mod.new_path or mod.old_path
            if path is None:
                continue
            if file_extensions and not any(path.endswith(ext) for ext in file_extensions):
                continue

            fs = state.setdefault(path, FileEvolutionState(file_path=path))
            fs.revisions += 1
            fs.authors.add(commit.author.email or commit.author.name)
            fs.lines_added += mod.added_lines or 0
            fs.lines_deleted += mod.deleted_lines or 0
            fs.per_commit_churn.append((mod.added_lines or 0) + (mod.deleted_lines or 0))
            if is_bugfix:
                fs.bug_fixes += 1
            if is_refactor:
                fs.refactorings += 1
            if fs.first_seen is None:
                fs.first_seen = commit.committer_date
            fs.last_seen = commit.committer_date

            ctype = _classify_change_type(mod.diff or "")
            fs.change_type_counts[ctype] += 1

        if commit.hash in commit_to_tag:
            release = commit_to_tag[commit.hash]
            for fs in state.values():
                rows.append(fs.as_row(release=release, as_of=commit.committer_date))

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="Mine MOSER/HASSAN/CHANGE_TYPE evolution metrics from a git repo.")
    parser.add_argument("--repo", required=True, help="Path to a local git repository (already cloned).")
    parser.add_argument("--release-tags", nargs="+", required=True, help="Git tag names, in chronological order.")
    parser.add_argument("--ext", nargs="+", default=[".java"], help="File extensions to include.")
    parser.add_argument("--out", required=True, help="Output CSV path.")
    args = parser.parse_args()

    df = mine_repository_by_tags(args.repo, args.release_tags, file_extensions=args.ext)
    df.to_csv(args.out, index=False)
    print(f"Wrote {len(df)} rows ({df['file_path'].nunique()} files, {df['release'].nunique()} releases) to {args.out}")


if __name__ == "__main__":
    main()
