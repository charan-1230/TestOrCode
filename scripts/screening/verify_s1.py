# ruff: noqa  (one-off screening tooling, not part of the testorcode package)
"""Dynamic S1 verification (proposal Table 3, S1) for commits that change source AND tests.

For commit c with parent p, run the repository's unit tests (Python 3.12, modern pytest) in disposable
git worktrees:
  R_p  = tests at p                     (baseline must be fully green)
  R_c  = tests at c                     (reference repair state must be fully green)
  R_so = p's tree + c's source files    (source-only change, old tests kept)
Valid S1 test = passes at p, fails in R_so, passes at c, and lives in a test file that c changed.
Also records src-only commits' baseline/after status for S2 (--src-only).
Usage: verify_s1.py <repo> <commits.json> <python> <test-args> <out.json> [--src-only] [--workers N]
"""

import concurrent.futures as cf, json, os, queue, re, shlex, subprocess, sys, threading, time

repo, cj, py, targs, out = sys.argv[1:6]
SRC_ONLY = "--src-only" in sys.argv
W = int(sys.argv[sys.argv.index("--workers") + 1]) if "--workers" in sys.argv else 6
repo = os.path.abspath(repo)
py = os.path.abspath(py)
base = os.path.dirname(repo)
name = os.path.basename(repo)
commits = json.load(open(cj))
TIMEOUT = 180


def git(cwd, *a, check=True):
    return subprocess.run(["git", "-C", cwd, *a], capture_output=True, text=True, check=check)


pool = queue.Queue()
wts = []
for i in range(W):
    wt = f"{base}/wt-{name}-{i}"
    if not os.path.isdir(wt):
        git(repo, "worktree", "add", "--detach", wt, "HEAD")
    wts.append(wt)
    pool.put(wt)

LINE = re.compile(r"^(PASSED|FAILED|ERROR|SKIPPED|XFAIL|XPASS) (\S+)")


def run_tests(wt):
    env = {
        k: v
        for k, v in os.environ.items()
        if k not in ("FORCE_COLOR", "PY_COLORS", "CLICOLOR_FORCE")
    }
    env.update(PYTHONPATH=wt, PYTHONDONTWRITEBYTECODE="1", NO_COLOR="1")
    t = time.time()
    try:
        p = subprocess.run(
            [
                py,
                "-m",
                "pytest",
                "-q",
                "--color=no",
                "--tb=no",
                "-rA",
                "-p",
                "no:cacheprovider",
                *shlex.split(targs),
            ],
            cwd=wt,
            env=env,
            capture_output=True,
            text=True,
            timeout=TIMEOUT,
        )
    except subprocess.TimeoutExpired:
        return {"status": "timeout", "tests": {}, "secs": TIMEOUT}
    tests = {}
    for l in p.stdout.splitlines():
        m = LINE.match(re.sub(r"\x1b\[[0-9;]*m", "", l))
        if m:
            tests[m.group(2)] = m.group(1)
    st = "green" if p.returncode == 0 else ("failed" if p.returncode == 1 else f"rc{p.returncode}")
    return {
        "status": st,
        "tests": tests,
        "secs": round(time.time() - t, 2),
        "tail": p.stdout.strip().splitlines()[-1:],
    }


def checkout(wt, sha):
    git(wt, "checkout", "-q", "-f", "--detach", sha)
    git(wt, "clean", "-fdxq")


memo, lock = {}, threading.Lock()


def full(sha):
    with lock:
        if sha in memo:
            return memo[sha]
    wt = pool.get()
    try:
        checkout(wt, sha)
        r = run_tests(wt)
    finally:
        pool.put(wt)
    with lock:
        memo[sha] = r
    return r


def source_only(c):
    p = c["parents"][0]
    wt = pool.get()
    try:
        checkout(wt, p)
        for st, path in c["src"]:
            if st == "D":
                git(wt, "rm", "-q", "-f", path, check=False)
            else:
                git(wt, "checkout", "-q", c["sha"], "--", path)
        return run_tests(wt)
    finally:
        pool.put(wt)


def verify(c):
    rec = {k: c[k] for k in ("sha", "subject", "ts", "kind", "changed_lines", "src", "test")}
    if not c["parents"]:
        rec["verdict"] = "root_commit"
        return rec
    p = c["parents"][0]
    rp, rc = full(p), full(c["sha"])
    rec["parent_status"], rec["commit_status"] = rp["status"], rc["status"]
    rec["secs"] = round(rp["secs"] + rc["secs"], 1)
    if c["kind"] == "src_only":
        rec["verdict"] = "s2_base_ok" if rp["status"] == rc["status"] == "green" else "env_or_red"
        rec["n_tests"] = len(rc["tests"])
        return rec
    if rp["status"] != "green" or rc["status"] != "green":
        rec["verdict"] = "baseline_not_green_on_py312"
        rec["tails"] = [rp.get("tail"), rc.get("tail")]
        return rec
    rso = source_only(c)
    rec["source_only_status"] = rso["status"]
    rec["rso_tail"] = rso.get("tail")
    rec["rso_n"] = len(rso["tests"])
    if rso["status"] == "green":
        rec["verdict"] = "source_only_still_green"
        return rec
    fails = sorted(
        t
        for t, s in rso["tests"].items()
        if s in ("FAILED", "ERROR")
        and rp["tests"].get(t) == "PASSED"
        and rc["tests"].get(t) == "PASSED"
    )
    other = sorted(
        t for t, s in rso["tests"].items() if s in ("FAILED", "ERROR") and t not in fails
    )
    changed_test_files = {p for _, p in c["test"]}
    in_changed = [t for t in fails if t.split("::")[0] in changed_test_files]
    rec.update(s1_failing=fails, s1_in_changed_test_files=in_changed, unexplained_failures=other)
    rec["verdict"] = (
        "S1_VALID"
        if in_changed and not other
        else ("S1_PARTIAL" if in_changed else "failures_not_in_changed_tests")
    )
    return rec


sel = [
    c
    for c in commits
    if c["kind"] == "both"
    or (SRC_ONLY and c["kind"] == "src_only" and c.get("changed_lines", 9999) <= 150)
]
print(f"{len(sel)} commits to verify with {W} workers", flush=True)
res = []
with cf.ThreadPoolExecutor(W) as ex:
    for i, r in enumerate(ex.map(verify, sel)):
        res.append(r)
        if i % 25 == 0:
            print(i, r["sha"][:8], r["verdict"], flush=True)
json.dump(res, open(out, "w"), indent=1)
from collections import Counter

print(Counter(r["verdict"] for r in res))
