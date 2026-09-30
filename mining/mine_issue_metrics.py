"""
mine_issue_metrics.py

Mines the ISSUE metric group by querying the GitHub REST API for closed
issues, and linking each issue to the files touched by its closing /
referencing commit (via the commit message's "Fixes #123" / "Closes #123"
convention, or explicit issue-to-commit links returned by the API).

ISSUE group per file, per release:
    ISSUE_count            total linked issues
    ISSUE_bug_count        linked issues labeled as a bug
    ISSUE_severity_score   sum of a simple severity weighting derived from
                            issue labels (critical/high/medium/low)

Requires a GitHub personal access token in the GITHUB_TOKEN environment
variable to avoid the (very low) unauthenticated rate limit.

Usage (CLI):
    export GITHUB_TOKEN=ghp_xxx
    python -m mining.mine_issue_metrics \
        --owner apache --repo somelib \
        --repo-path /path/to/local/clone \
        --release-tags v1.0 v1.1 v1.2 \
        --out data/raw/issue_metrics.csv
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import time
from collections import defaultdict
from typing import Dict, List, Optional

import pandas as pd
import requests

GITHUB_API = "https://api.github.com"
CLOSES_PATTERN = re.compile(
    r"\b(close[sd]?|fix(e[sd])?|resolve[sd]?)\s+#(\d+)", re.IGNORECASE
)

SEVERITY_WEIGHTS = {
    "critical": 4,
    "blocker": 4,
    "high": 3,
    "major": 3,
    "medium": 2,
    "moderate": 2,
    "low": 1,
    "minor": 1,
    "trivial": 1,
}


def _session(token: Optional[str]) -> requests.Session:
    s = requests.Session()
    headers = {"Accept": "application/vnd.github+json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    s.headers.update(headers)
    return s


def fetch_closed_issues(owner: str, repo: str, session: requests.Session) -> List[dict]:
    """Paginate through all closed issues (GitHub's API returns PRs mixed in
    with issues; we filter those out via the absence of `pull_request`)."""
    issues = []
    page = 1
    while True:
        resp = session.get(
            f"{GITHUB_API}/repos/{owner}/{repo}/issues",
            params={"state": "closed", "per_page": 100, "page": page},
        )
        if resp.status_code == 403:
            # Rate limited -- back off once and retry.
            reset = int(resp.headers.get("X-RateLimit-Reset", time.time() + 60))
            wait = max(reset - time.time(), 1)
            print(f"[warn] rate limited, sleeping {wait:.0f}s")
            time.sleep(wait)
            continue
        resp.raise_for_status()
        batch = resp.json()
        if not batch:
            break
        issues.extend(i for i in batch if "pull_request" not in i)
        page += 1
    return issues


def _issue_severity(issue: dict) -> int:
    labels = [l["name"].lower() for l in issue.get("labels", [])]
    return max((SEVERITY_WEIGHTS.get(l, 0) for l in labels), default=1)


def _issue_is_bug(issue: dict) -> bool:
    labels = [l["name"].lower() for l in issue.get("labels", [])]
    return any("bug" in l or "defect" in l for l in labels)


def link_issues_to_files(repo_path: str, issues: List[dict]) -> Dict[int, List[str]]:
    """For each closed issue, find commits in the local clone whose message
    references it (Fixes #N / Closes #N) and return the set of files that
    commit touched."""
    issue_numbers = {i["number"] for i in issues}
    issue_to_files: Dict[int, List[str]] = defaultdict(list)

    log = subprocess.check_output(
        ["git", "-C", repo_path, "log", "--all", "--pretty=format:%H%x01%s%x02%b%x03"],
        stderr=subprocess.DEVNULL,
    ).decode(errors="ignore")

    for entry in log.split("\x03"):
        if "\x01" not in entry:
            continue
        commit_hash, rest = entry.split("\x01", 1)
        subject, _, body = rest.partition("\x02")
        message = f"{subject}\n{body}"
        referenced = {int(n) for n in CLOSES_PATTERN.findall(message.replace("\n", " ")) and
                      re.findall(r"#(\d+)", message)}
        referenced &= issue_numbers
        if not referenced:
            continue
        try:
            files_out = subprocess.check_output(
                ["git", "-C", repo_path, "show", "--name-only", "--pretty=format:", commit_hash.strip()],
                stderr=subprocess.DEVNULL,
            ).decode(errors="ignore")
        except subprocess.CalledProcessError:
            continue
        files = [f for f in files_out.splitlines() if f.strip()]
        for num in referenced:
            issue_to_files[num].extend(files)

    return issue_to_files


def build_issue_metrics(
    owner: str,
    repo: str,
    repo_path: str,
    releases: List[str],
    token: Optional[str] = None,
) -> pd.DataFrame:
    """Builds one row per (release, file) with aggregated ISSUE metrics.
    Every release gets the same cumulative issue linkage in this simplified
    implementation; refine by filtering `issues` on `closed_at` <= release
    date if release timestamps are available."""
    session = _session(token or os.environ.get("GITHUB_TOKEN"))
    issues = fetch_closed_issues(owner, repo, session)
    issue_to_files = link_issues_to_files(repo_path, issues)
    issues_by_number = {i["number"]: i for i in issues}

    file_stats: Dict[str, dict] = defaultdict(lambda: {"count": 0, "bug_count": 0, "severity_sum": 0})
    for number, files in issue_to_files.items():
        issue = issues_by_number.get(number)
        if issue is None:
            continue
        is_bug = _issue_is_bug(issue)
        severity = _issue_severity(issue)
        for f in set(files):
            file_stats[f]["count"] += 1
            file_stats[f]["bug_count"] += int(is_bug)
            file_stats[f]["severity_sum"] += severity

    rows = []
    for release in releases:
        for file_path, stats in file_stats.items():
            rows.append({
                "release": release,
                "file_path": file_path,
                "ISSUE_count": stats["count"],
                "ISSUE_bug_count": stats["bug_count"],
                "ISSUE_severity_score": stats["severity_sum"],
            })
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="Mine ISSUE metrics from the GitHub API + local git log.")
    parser.add_argument("--owner", required=True)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--repo-path", required=True, help="Path to a local clone of the same repo.")
    parser.add_argument("--release-tags", nargs="+", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    df = build_issue_metrics(args.owner, args.repo, args.repo_path, args.release_tags)
    df.to_csv(args.out, index=False)
    print(f"Wrote {len(df)} rows to {args.out}")


if __name__ == "__main__":
    main()
