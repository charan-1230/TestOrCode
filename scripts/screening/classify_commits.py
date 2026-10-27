# ruff: noqa  (one-off screening tooling, not part of the testorcode package)
"""Static classification of every non-merge commit of a candidate repository.

Usage: classify_commits.py <repo_dir> <src_prefix[,src_prefix]> <out.json>
A commit is 'both' if it changes >=1 production .py file and >=1 test .py file.
Production = .py under a src prefix and not a test path; test = test_*.py / tests dir / conftest.
"""

import json, re, subprocess, sys

repo, prefixes, out = sys.argv[1], tuple(sys.argv[2].split(",")), sys.argv[3]


def git(*a):
    return subprocess.run(
        ["git", "-C", repo, *a], capture_output=True, text=True, check=True
    ).stdout


def is_test(p):
    b = p.rsplit("/", 1)[-1]
    return p.endswith(".py") and (
        b.startswith("test_")
        or "/tests/" in "/" + p
        or b == "conftest.py"
        or b.endswith("_test.py")
    )


def is_src(p):
    return p.endswith(".py") and p.startswith(prefixes) and not is_test(p)


raw = (
    git(
        "log",
        "--no-merges",
        "--name-status",
        "--format=@@%H|%P|%ct|%s",
        "--diff-filter=AMDR",
        "--first-parent=0",
    )
    if False
    else git("log", "--no-merges", "--name-status", "--format=@@%H|%P|%ct|%s")
)
commits, cur = [], None
for line in raw.splitlines():
    if line.startswith("@@"):
        h, parents, ct, subj = line[2:].split("|", 3)
        cur = {
            "sha": h,
            "parents": parents.split(),
            "ts": int(ct),
            "subject": subj,
            "src": [],
            "test": [],
            "other": [],
        }
        commits.append(cur)
    elif line.strip():
        parts = line.split("\t")
        status, path = parts[0], parts[-1]
        (cur["src"] if is_src(path) else cur["test"] if is_test(path) else cur["other"]).append(
            (status[0], path)
        )


def numstat(sha):
    n = 0
    for l in git("show", "--numstat", "--format=", sha).splitlines():
        a, b, *_ = l.split("\t")
        n += (int(a) if a.isdigit() else 0) + (int(b) if b.isdigit() else 0)
    return n


for c in commits:
    c["kind"] = (
        "both"
        if c["src"] and c["test"]
        else "src_only"
        if c["src"]
        else "test_only"
        if c["test"]
        else "other"
    )
    if c["kind"] in ("both", "src_only"):
        c["changed_lines"] = numstat(c["sha"])
json.dump(commits, open(out, "w"), indent=1)
from collections import Counter

print(Counter(c["kind"] for c in commits), "total", len(commits))
b = [c for c in commits if c["kind"] == "both"]
print(
    "both: changed_lines<=60:",
    sum(c["changed_lines"] <= 60 for c in b),
    " <=150:",
    sum(c["changed_lines"] <= 150 for c in b),
)
s = [c for c in commits if c["kind"] == "src_only"]
print(
    "src_only: <=60:",
    sum(c["changed_lines"] <= 60 for c in s),
    " <=150:",
    sum(c["changed_lines"] <= 150 for c in s),
)
