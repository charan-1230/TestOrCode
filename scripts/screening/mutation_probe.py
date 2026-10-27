# ruff: noqa  (one-off screening tooling, not part of the testorcode package)
"""Feasibility probe for mutation-based integrity checks (NOT a full mutation campaign).

Enumerates simple AST mutants (comparison/arith/bool operator swaps, constant tweaks, return->None) of one module in a
scratch COPY of the repo, samples N with a fixed seed, runs the module's tests against each, records kill / survive /
timeout and wall time.  Usage: mutation_probe.py <repo_dir> <python> <module_relpath> <test_target> <N> <seed> <out.json>
"""

import ast, copy, json, os, random, shutil, subprocess, sys, tempfile, time

repo, py, mod, tests, N, seed, out = sys.argv[1:8]
N, seed = int(N), int(seed)
repo, py = os.path.abspath(repo), os.path.abspath(py)
work = tempfile.mkdtemp(prefix="mut-", dir=os.path.dirname(repo))
tree_dir = os.path.join(work, "t")
shutil.copytree(
    repo, tree_dir, ignore=shutil.ignore_patterns(".git", "__pycache__", ".pytest_cache")
)
src = open(os.path.join(tree_dir, mod)).read()

CMP = {
    ast.Eq: ast.NotEq,
    ast.NotEq: ast.Eq,
    ast.Lt: ast.LtE,
    ast.LtE: ast.Lt,
    ast.Gt: ast.GtE,
    ast.GtE: ast.Gt,
    ast.Is: ast.IsNot,
    ast.IsNot: ast.Is,
    ast.In: ast.NotIn,
    ast.NotIn: ast.In,
}
BIN = {
    ast.Add: ast.Sub,
    ast.Sub: ast.Add,
    ast.Mult: ast.FloorDiv,
    ast.FloorDiv: ast.Mult,
    ast.Mod: ast.Mult,
}


class Mutator(ast.NodeTransformer):
    def __init__(self, target=None):
        self.target, self.i, self.sites, self.stack = target, -1, [], []

    def _hit(self, desc):
        self.i += 1
        self.sites.append((self.i, desc, ".".join(self.stack) or "<module>"))
        return self.i == self.target

    def visit_FunctionDef(self, n):
        self.stack.append(n.name)
        self.generic_visit(n)
        self.stack.pop()
        return n

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_ClassDef(self, n):
        self.stack.append(n.name)
        self.generic_visit(n)
        self.stack.pop()
        return n

    def visit_Compare(self, n):
        self.generic_visit(n)
        for k, op in enumerate(n.ops):
            if type(op) in CMP and self._hit(f"cmp {type(op).__name__}->{CMP[type(op)].__name__}"):
                n.ops[k] = CMP[type(op)]()
        return n

    def visit_BinOp(self, n):
        self.generic_visit(n)
        if type(n.op) in BIN and self._hit(
            f"bin {type(n.op).__name__}->{BIN[type(n.op)].__name__}"
        ):
            n.op = BIN[type(n.op)]()
        return n

    def visit_BoolOp(self, n):
        self.generic_visit(n)
        if self._hit("bool and<->or"):
            n.op = ast.Or() if isinstance(n.op, ast.And) else ast.And()
        return n

    def visit_Constant(self, n):
        if isinstance(n.value, bool):
            if self._hit("const bool flip"):
                return ast.copy_location(ast.Constant(not n.value), n)
        elif isinstance(n.value, int) and self.stack:
            if self._hit("const int+1"):
                return ast.copy_location(ast.Constant(n.value + 1), n)
        return n

    def visit_Return(self, n):
        self.generic_visit(n)
        if n.value is not None and self.stack and self._hit("return->None"):
            n.value = ast.Constant(None)
        return n


base = ast.parse(src)
m = Mutator()
m.visit(copy.deepcopy(base))
sites = m.sites
random.seed(seed)
pick = sorted(random.sample(range(len(sites)), min(N, len(sites))))
env = {k: v for k, v in os.environ.items() if k not in ("FORCE_COLOR", "PY_COLORS")}
env.update(PYTHONPATH=tree_dir, PYTHONDONTWRITEBYTECODE="1", NO_COLOR="1")
target = os.path.join(tree_dir, mod)


def run():
    t = time.time()
    try:
        p = subprocess.run(
            [
                py,
                "-m",
                "pytest",
                "-x",
                "-q",
                "--color=no",
                "--tb=no",
                "-p",
                "no:cacheprovider",
                "--import-mode=importlib",
                tests,
            ],
            cwd=tree_dir,
            env=env,
            capture_output=True,
            text=True,
            timeout=60,
        )
        return p.returncode, time.time() - t
    except subprocess.TimeoutExpired:
        return "timeout", time.time() - t


rc0, t0 = run()
res = {
    "module": mod,
    "tests": tests,
    "baseline_rc": rc0,
    "baseline_s": round(t0, 2),
    "sites_total": len(sites),
    "mutants": [],
}
if rc0 != 0:
    print("baseline not green", rc0)
    json.dump(res, open(out, "w"), indent=1)
    sys.exit(1)
for idx in pick:
    mm = Mutator(target=idx)
    tree = mm.visit(copy.deepcopy(base))
    ast.fix_missing_locations(tree)
    open(target, "w").write(ast.unparse(tree))
    rc, secs = run()
    status = (
        "killed"
        if rc in (1, 2)
        else "survived"
        if rc == 0
        else "timeout"
        if rc == "timeout"
        else f"rc{rc}"
    )
    res["mutants"].append(
        {
            "site": idx,
            "desc": sites[idx][1],
            "function": sites[idx][2],
            "status": status,
            "secs": round(secs, 2),
        }
    )
open(target, "w").write(src)
from collections import Counter

c = Counter(x["status"] for x in res["mutants"])
res["summary"] = dict(c)
res["mean_secs"] = round(sum(x["secs"] for x in res["mutants"]) / max(1, len(res["mutants"])), 2)
k, s = c.get("killed", 0), c.get("survived", 0)
res["kill_rate_excl_timeouts"] = round(k / (k + s), 2) if k + s else None
json.dump(res, open(out, "w"), indent=1)
print(
    mod,
    res["summary"],
    "kill_rate",
    res["kill_rate_excl_timeouts"],
    "mean_s",
    res["mean_secs"],
    "sites",
    len(sites),
)
shutil.rmtree(work, ignore_errors=True)
