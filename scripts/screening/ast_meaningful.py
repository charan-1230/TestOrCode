# ruff: noqa  (one-off screening tooling, not part of the testorcode package)
"""Among commits with a green baseline, count those that change executable code (AST minus docstrings), excluding
version-bump-only changes to __init__.py.  Usage: ast_meaningful.py <repo> <commits.json> <verdicts.json> [<verdicts2.json>]"""

import ast, json, subprocess, sys, datetime, collections

repo, cj, vj = sys.argv[1:4]
commits = {c["sha"]: c for c in json.load(open(cj))}
ver = json.load(open(vj))
ok = [v for v in ver if v["verdict"] in ("s2_base_ok", "source_only_still_green")]


def show(sha, path):
    r = subprocess.run(["git", "-C", repo, "show", f"{sha}:{path}"], capture_output=True, text=True)
    return r.stdout if r.returncode == 0 else None


def strip(src):
    t = ast.parse(src)
    for n in ast.walk(t):
        if (
            isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Module))
            and n.body
            and isinstance(n.body[0], ast.Expr)
            and isinstance(getattr(n.body[0], "value", None), ast.Constant)
            and isinstance(n.body[0].value.value, str)
        ):
            n.body = n.body[1:] or [ast.Pass()]
    return ast.dump(t)


def funcs(src):
    return {
        n.name: ast.dump(n)
        for n in ast.walk(ast.parse(src))
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
    }


res = collections.defaultdict(list)
for v in ok:
    c = commits[v["sha"]]
    p = c["parents"][0]
    changed = 0
    fn_changed = set()
    only_init = True
    for st, path in c["src"]:
        a, b = show(p, path), show(c["sha"], path)
        if a is None or b is None:
            changed += 1
            only_init &= path.endswith("__init__.py")
            continue
        try:
            if strip(a) != strip(b):
                changed += 1
                if not path.endswith("__init__.py"):
                    only_init = False
                fa, fb = funcs(a), funcs(b)
                fn_changed |= {f"{path}::{k}" for k in set(fa) | set(fb) if fa.get(k) != fb.get(k)}
        except SyntaxError:
            pass
    if changed and (not only_init) and fn_changed:
        res[datetime.date.fromtimestamp(v["ts"]).year].append(
            (v["sha"][:8], v["kind"], len(fn_changed), v["subject"][:60])
        )
tot = sum(len(x) for x in res.values())
print(vj, "green-baseline commits:", len(ok), " with executable-code change in >=1 function:", tot)
for y in sorted(res):
    print(y, len(res[y]))
json.dump(
    {y: x for y, x in res.items()}, open(vj.replace(".json", "_meaningful.json"), "w"), indent=1
)
