"""Build the pilot dev/test episodes from verified boltons history.

Reads ``manifest.json``, drives a pinned clone of boltons in a private git worktree, runs the real tests, and
writes ``datasets/pilot_{dev,test}[_agent_view].jsonl``, ``pilot_manifest.json`` and ``pilot_verification.json``.
Everything recorded was executed here; nothing is assumed.  Deterministic: no timestamps, fixed search order.

    python scripts/pilot/build_dataset.py --repo ../../screening/boltons \
        --python ../../screening/venv-boltons8/bin/python --work ../../screening/wt-pilot
"""

from __future__ import annotations

import argparse
import ast
import difflib
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(HERE))

from materialize_episode import materialize

from testorcode.schemas.episode_schemas import (
    Availability,
    CIEvent,
    CommitRef,
    DecisionLabel,
    Difficulty,
    EditPolicy,
    EnvironmentSpec,
    Episode,
    EpisodeBudget,
    EpisodeGroundTruth,
    EpisodeInput,
    EvidenceItem,
    EvidenceSourceType,
    ExecutableProbe,
    FailingTest,
    FileChange,
    LabelProvenance,
    MemoryRecord,
    MutantSpec,
    PerTestGoldLabel,
    ReferenceRepair,
    RepairDirection,
    RepoRef,
    RetrievalCorpusSpec,
    ScenarioType,
    StrippedEvidence,
    ToolFaultInjection,
    ToolFaultKind,
    ToolFaultPlan,
    ToolName,
    find_leaked_ground_truth,
)

STABILITY_RUNS = 5
SCENARIO = {
    "S1": ScenarioType.S1_OUTDATED_TEST,
    "S2": ScenarioType.S2_INJECTED_REGRESSION,
    "S3": ScenarioType.S3_MIXED,
    "S4": ScenarioType.S4_DECOY_REGRESSION,
    "S5": ScenarioType.S5_INTENT_STRIPPED,
}
BUDGET = EpisodeBudget(
    max_tool_calls=40, max_llm_calls=25, max_tokens=120000, max_revisions=1, wall_clock_seconds=900
)  # placeholder budgets: tune on the dev split
POLICY = EditPolicy(allowed_source_roots=("boltons/",), allowed_test_roots=("tests/",))
SWAPS = [
    (" >= ", " > "),
    (" <= ", " < "),
    (" == ", " != "),
    (" > ", " >= "),
    (" < ", " <= "),
    (" + ", " - "),
    (" and ", " or "),
    (" is not ", " is "),
]
CHUNK = 1200


class BuildError(RuntimeError):
    """A manifest entry cannot be turned into a verified episode."""


# --------------------------------------------------------------------------- #
# Git and pytest plumbing
# --------------------------------------------------------------------------- #


class Ctx:
    def __init__(self, repo: Path, python: Path, work: Path) -> None:
        self.repo, self.work = repo.resolve(), work.resolve()
        self.python = Path(os.path.abspath(python))
        env = {
            k: v for k, v in os.environ.items() if k not in ("FORCE_COLOR", "PY_COLORS", "GIT_DIR")
        }
        env.update(PYTHONPATH=str(self.work), PYTHONDONTWRITEBYTECODE="1", NO_COLOR="1")
        self.env = env
        if not (self.work / ".git").exists():
            self.git("worktree", "add", "-q", "--detach", str(self.work), "HEAD", cwd=self.repo)

    def git(self, *args: str, cwd: Path | None = None) -> str:
        done = subprocess.run(
            ["git", "-C", str(cwd or self.work), *args], capture_output=True, text=True, check=False
        )
        if done.returncode != 0:
            msg = f"git {' '.join(args)}: {done.stderr.strip()[:300]}"
            raise BuildError(msg)
        return done.stdout

    def rg(self, *args: str) -> str:
        return self.git(*args, cwd=self.repo)

    def full(self, rev: str) -> str:
        return self.rg("rev-parse", rev).strip()

    def checkout(self, sha: str) -> None:
        self.git("checkout", "-q", "-f", "--detach", sha)
        self.git("clean", "-fdxq")

    def show(self, sha: str, path: str) -> str | None:
        done = subprocess.run(
            ["git", "-C", str(self.repo), "show", f"{sha}:{path}"],
            capture_output=True,
            text=True,
            check=False,
        )
        return done.stdout if done.returncode == 0 else None

    def pytest(
        self, targets: list[str], timeout: int = 120
    ) -> tuple[int | str, dict[str, tuple[str, str]], set[str]]:
        """-> (rc, {failed nodeid: (message, traceback)}, passed nodeids)."""
        with tempfile.NamedTemporaryFile(suffix=".xml") as handle:
            cmd = [
                str(self.python),
                "-m",
                "pytest",
                "-q",
                "--color=no",
                "--tb=short",
                "-p",
                "no:cacheprovider",
                "--import-mode=importlib",
                f"--junitxml={handle.name}",
                *targets,
            ]
            try:
                done = subprocess.run(
                    cmd,
                    cwd=self.work,
                    env=self.env,
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                    check=False,
                )
            except subprocess.TimeoutExpired:
                return "timeout", {}, set()
            fails: dict[str, tuple[str, str]] = {}
            passed: set[str] = set()
            for case in ET.parse(handle.name).getroot().iter("testcase"):
                parts = case.get("classname", "").split(".")
                idx = next(
                    (
                        i
                        for i in range(len(parts), 0, -1)
                        if (self.work / ("/".join(parts[:i]) + ".py")).exists()
                    ),
                    None,
                )
                if idx is None:
                    continue
                nodeid = (
                    "/".join(parts[:idx])
                    + ".py::"
                    + "::".join([*parts[idx:], case.get("name", "")])
                )
                bad = (
                    case.find("failure") if case.find("failure") is not None else case.find("error")
                )
                if bad is not None:
                    fails[nodeid] = ((bad.get("message") or "")[:300], (bad.text or "")[:1500])
                elif case.find("skipped") is None:
                    passed.add(nodeid)
            return done.returncode, fails, passed

    def stable_failures(self) -> tuple[dict[str, tuple[str, str]], set[str]]:
        """Full suite STABILITY_RUNS times; the failing set must be identical every time."""
        sets, last = [], ({}, set())
        for _ in range(STABILITY_RUNS):
            rc, fails, passed = self.pytest(["tests"])
            if rc not in (0, 1):
                msg = f"pytest exited {rc}"
                raise BuildError(msg)
            sets.append(frozenset(fails))
            last = (fails, passed)
        if len(set(sets)) != 1:
            msg = "failing set differs between repeated runs"
            raise BuildError(msg)
        return last


def iter_funcs(tree: ast.AST, prefix: tuple[str, ...] = ()):
    for child in ast.iter_child_nodes(tree):
        if isinstance(child, ast.ClassDef):
            yield from iter_funcs(child, (*prefix, child.name))
        elif isinstance(child, ast.FunctionDef | ast.AsyncFunctionDef):
            yield ".".join((*prefix, child.name)), child
            yield from iter_funcs(child, (*prefix, child.name))
        else:
            yield from iter_funcs(child, prefix)


def changed_functions(ctx: Ctx, parent: str, commit: str, path: str) -> set[str]:
    old, new = ctx.show(parent, path), ctx.show(commit, path)
    if old is None or new is None:
        return set()
    a = {q: ast.dump(n) for q, n in iter_funcs(ast.parse(old))}
    b = {q: ast.dump(n) for q, n in iter_funcs(ast.parse(new))}
    return {q for q in set(a) | set(b) if a.get(q) != b.get(q)}


def module_of(path: str) -> str:
    return Path(path).stem


def test_targets(ctx: Ctx, path: str) -> list[str]:
    mod = module_of(path)
    found = sorted(
        p.relative_to(ctx.work).as_posix() for p in (ctx.work / "tests").glob(f"test_{mod}*.py")
    )
    return found or ["tests"]


@dataclass
class Mutant:
    diff: str
    desc: str
    qualname: str
    path: str
    new_lines: list[str]
    new_failures: list[str]


def find_mutant(
    ctx: Ctx, path: str, funcs: set[str] | None, exclude: set[str], accept
) -> Mutant | None:
    """First line-level operator swap in the CURRENT worktree that ``accept(new_failures)`` approves.

    Deterministic: functions in source order, lines in order, SWAPS in order.  The file is always restored.
    """
    full = ctx.work / path
    source = full.read_text()
    lines = source.splitlines(keepends=True)
    targets = test_targets(ctx, path)
    _, base_fails, _ = ctx.pytest(targets)
    for qual, node in iter_funcs(ast.parse(source)):
        if (funcs is not None and qual not in funcs) or qual in exclude:
            continue
        doc_end = (
            node.body[0].end_lineno
            if node.body
            and isinstance(node.body[0], ast.Expr)
            and isinstance(getattr(node.body[0], "value", None), ast.Constant)
            else node.lineno
        )
        for ln in range(doc_end + 1, node.end_lineno + 1):
            text = lines[ln - 1]
            if text.lstrip().startswith(("#", '"', "'")) or '"""' in text or "'''" in text:
                continue
            for old, new in SWAPS:
                if old not in text:
                    continue
                mutated = list(lines)
                mutated[ln - 1] = text.replace(old, new, 1)
                try:
                    ast.parse("".join(mutated))
                except SyntaxError:
                    continue
                full.write_text("".join(mutated))
                try:
                    rc, fails, _ = ctx.pytest(targets, timeout=30)
                finally:
                    full.write_text(source)
                extra = set(fails) - set(base_fails)
                if rc == 1 and extra and accept(extra):
                    diff = "".join(
                        difflib.unified_diff(lines, mutated, f"a/{path}", f"b/{path}", n=3)
                    )
                    return Mutant(
                        diff,
                        f"{qual} line {ln}: '{old.strip()}' -> '{new.strip()}'",
                        qual,
                        path,
                        mutated,
                        sorted(extra),
                    )
    return None


# --------------------------------------------------------------------------- #
# Corpus
# --------------------------------------------------------------------------- #


def chunks(text: str, limit: int = CHUNK) -> list[str]:
    out, cur = [], ""
    for para in re.split(r"\n\s*\n", text):
        if cur and len(cur) + len(para) > limit:
            out.append(cur.strip())
            cur = ""
        cur += para + "\n\n"
    if cur.strip():
        out.append(cur.strip())
    return [c[: limit * 2] for c in out]


def build_corpus(
    ctx: Ctx,
    cutoff: str,
    modules: set[str],
    message_override: str | None = None,
    drop_docs: set[str] = frozenset(),
) -> list[EvidenceItem]:
    items: list[EvidenceItem] = []
    total = int(ctx.rg("rev-list", "--count", cutoff).strip())
    log = ctx.rg("log", "--no-merges", "-n", "100", "--format=%H%x1f%B%x1e", cutoff)
    for entry in filter(None, (e.strip() for e in log.split("\x1e"))):
        sha, _, body = entry.partition("\x1f")
        text = (
            message_override if (message_override is not None and sha == cutoff) else body
        ).strip()
        index = int(ctx.rg("rev-list", "--count", sha).strip()) - 1
        items.append(
            EvidenceItem(
                evidence_id=f"msg:{sha[:12]}",
                source_type=EvidenceSourceType.COMMIT_MESSAGE,
                source_ref=sha,
                commit_sha=sha,
                commit_index=min(index, total - 1),
                text=text[:CHUNK],
            )
        )
    for path in ["README.md", *[f"docs/{m}.rst" for m in sorted(modules)]]:
        if path in drop_docs:
            continue
        body = ctx.show(cutoff, path)
        for n, piece in enumerate(chunks(body or ""), 1):
            items.append(
                EvidenceItem(
                    evidence_id=f"doc:{path}#{n}",
                    source_type=EvidenceSourceType.DOCUMENTATION,
                    source_ref=path,
                    commit_sha=cutoff,
                    text=piece,
                )
            )
    changelog = ctx.show(cutoff, "CHANGELOG.md") or ""
    sections = re.split(r"(?m)^(?=## )", changelog)[1:4]
    for n, sec in enumerate(sections, 1):
        items.append(
            EvidenceItem(
                evidence_id=f"changelog:{n}",
                source_type=EvidenceSourceType.CHANGELOG,
                source_ref="CHANGELOG.md",
                commit_sha=cutoff,
                text=sec.strip()[: CHUNK * 3],
            )
        )
    for mod in sorted(modules):
        rel = f"boltons/{mod}.py"
        path = ctx.work / rel
        if not path.exists():
            continue
        tree = ast.parse(path.read_text())
        entries = [(mod, tree)] + [(f"{mod}.{q}", n) for q, n in iter_funcs(tree)]
        entries += [(f"{mod}.{n.name}", n) for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]
        seen: set[str] = set()
        for name, node in entries:
            doc = ast.get_docstring(node)
            if not doc or name in seen:
                continue
            seen.add(name)
            first = node.body[0]
            items.append(
                EvidenceItem(
                    evidence_id=f"docstring:{name}",
                    source_type=EvidenceSourceType.DOCSTRING,
                    source_ref=f"{rel}::{name}",
                    commit_sha=cutoff,
                    line_start=first.lineno,
                    line_end=first.end_lineno,
                    text=doc[:700],
                )
            )
    return items


# --------------------------------------------------------------------------- #
# Episode drafts
# --------------------------------------------------------------------------- #


@dataclass
class Draft:
    key: str
    split: str
    scenario: str
    ci_commit: str
    parent: str
    message: str
    diffs: dict[str, tuple[str, str]]  # path -> (change_kind, diff)
    failing: list[FailingTest]
    labels: list[dict]
    corpus: list[EvidenceItem]
    cutoff_index: int
    memory: list[MemoryRecord] | None = None
    mutants: list[MutantSpec] = field(default_factory=list)
    reference: list[ReferenceRepair] = field(default_factory=list)
    decoys: list[str] = field(default_factory=list)
    stripped: list[StrippedEvidence] = field(default_factory=list)
    availability: Availability = Availability.AVAILABLE
    provenance: str = "tests_with_source_only_change"
    fault_plan: ToolFaultPlan = field(default_factory=ToolFaultPlan)
    notes: dict = field(default_factory=dict)
    functions: set[str] = field(default_factory=set)
    modules: set[str] = field(default_factory=set)
    failing_ids: list[str] = field(default_factory=list)
    outdated_ids: list[str] = field(default_factory=list)
    probe_nodeid: str | None = None
    probe_code: str | None = None
    source_function: str | None = None
    source_commit: str | None = None
    source_subject: str | None = None


def split_diff_paths(ctx: Ctx, parent: str) -> dict[str, tuple[str, str]]:
    """Per-path diff of the CURRENT worktree against ``parent`` (the exact CI-state change)."""
    out: dict[str, tuple[str, str]] = {}
    ctx.git("add", "-N", "--", "boltons")
    names = ctx.git("diff", "--name-status", parent, "--", "boltons").splitlines()
    for row in names:
        status, path = row.split("\t")[0][0], row.split("\t")[-1]
        diff = ctx.git("diff", parent, "--", path)
        out[path] = ({"A": "added", "D": "deleted", "R": "renamed"}.get(status, "modified"), diff)
    ctx.git("reset", "-q")
    return out


ADDRESS = re.compile(r"0x[0-9a-fA-F]{8,}")
TMP_DIR = re.compile(r"/tmp/pytest-of-[^/\s'\"]+/pytest-\d+/")


def stable_text(text: str) -> str | None:
    """Memory addresses and pytest tmp dirs (run counter, OS user) change per run; pin them for reproducibility."""
    text = TMP_DIR.sub("/tmp/pytest-of-user/pytest-0/", text)
    return ADDRESS.sub("0x7f0000000000", text) or None


def failing_models(fails: dict[str, tuple[str, str]]) -> list[FailingTest]:
    return [
        FailingTest(
            test_id=n,
            test_file=n.split("::")[0],
            test_name=n.split("::")[-1],
            failure_message=stable_text(m),
            traceback=stable_text(t),
        )
        for n, (m, t) in sorted(fails.items())
    ]


def qualified(module: str, qual: str) -> str:
    return f"boltons.{module}.{qual}"


def common(
    ctx: Ctx,
    spec: dict,
    commit: str,
    parent: str,
    modules: set[str],
    message: str | None = None,
    message_override: str | None = None,
    drop_docs: set[str] = frozenset(),
) -> dict:
    return {
        "key": spec["key"],
        "split": spec["split"],
        "scenario": spec["scenario"],
        "ci_commit": commit,
        "parent": parent,
        "message": message
        if message is not None
        else ctx.rg("log", "-1", "--format=%B", commit).strip(),
        "corpus": build_corpus(ctx, commit, modules, message_override, drop_docs),
        "cutoff_index": int(ctx.rg("rev-list", "--count", commit).strip()) - 1,
        "modules": modules,
    }


def tool_faults(spec: dict) -> ToolFaultPlan:
    plan = spec.get("fault_plan")
    if not plan:
        return ToolFaultPlan()
    return ToolFaultPlan(
        enabled=True,
        seed=42,
        injections=[
            ToolFaultInjection(
                tool=ToolName(p["tool"]),
                fault_kind=ToolFaultKind(p["fault_kind"]),
                call_index=p["call_index"],
            )
            for p in plan
        ],
    )


def build_outdated(ctx: Ctx, spec: dict, pool: set[str]) -> tuple[Draft, dict]:
    """S1 / S5 / S3 (outdated part): parent + the commit's source-only change; developer tests are the reference."""
    commit = ctx.full(spec["commit"])
    parent = ctx.full(commit + "^")
    names = [r.split("\t") for r in ctx.rg("diff", "--name-status", parent, commit).splitlines()]
    src = [p[-1] for p in names if p[-1].startswith("boltons/") and p[-1].endswith(".py")]
    tst = [p[-1] for p in names if p[-1].startswith("tests/") and p[-1].endswith(".py")]
    docs = {p[-1] for p in names if p[-1].startswith("docs/")}
    if not src or not tst:
        msg = f"{spec['key']}: commit must change source and tests"
        raise BuildError(msg)
    modules = {module_of(p) for p in src}
    ctx.checkout(parent)
    rc, _base_fails, parent_pass = ctx.pytest(["tests"])
    if rc != 0:
        msg = f"{spec['key']}: parent is not green"
        raise BuildError(msg)
    ctx.git("checkout", "-q", commit, "--", *src)
    fails, _ = ctx.stable_failures()
    if not fails or not set(fails) <= parent_pass:
        msg = f"{spec['key']}: source-only change must fail tests that pass at the parent"
        raise BuildError(msg)
    function = spec["function"]
    changed = set().union(*(changed_functions(ctx, parent, commit, p) for p in src))
    if function not in changed:
        msg = f"{spec['key']}: {function} not among changed functions {sorted(changed)[:8]}"
        raise BuildError(msg)
    fq = qualified(
        module_of(next(p for p in src if function in changed_functions(ctx, parent, commit, p))),
        function,
    )
    message = ctx.rg("log", "-1", "--format=%B", commit).strip()
    override = spec.get("generic_message")
    stripped: list[StrippedEvidence] = []
    if spec["scenario"] == "S5":
        stripped.append(
            StrippedEvidence(source_type=EvidenceSourceType.COMMIT_MESSAGE, source_ref=commit[:12])
        )
        stripped += [
            StrippedEvidence(source_type=EvidenceSourceType.DOCUMENTATION, source_ref=d)
            for d in sorted(docs)
            if d.endswith(".rst")
        ]
    base = common(
        ctx,
        spec,
        commit,
        parent,
        modules,
        message=override or message,
        message_override=override,
        drop_docs={d for d in docs if spec["scenario"] == "S5"},
    )
    short = commit[:12]
    labels = [{"test_id": t, "group_id": "g1", "function_qualname": fq} for t in sorted(fails)]
    test_diff = ctx.rg("diff", parent, commit, "--", *tst)
    draft = Draft(
        **base,
        diffs=split_diff_paths(ctx, parent),
        failing=failing_models(fails),
        labels=labels,
        reference=[
            ReferenceRepair(unified_diff=test_diff, changed_files=tst, source_commit=commit)
        ],
        stripped=stripped,
        fault_plan=tool_faults(spec),
        functions={fq},
        failing_ids=sorted(fails),
        outdated_ids=sorted(fails),
        source_function=fq,
        source_commit=commit,
        source_subject=message.splitlines()[0],
    )
    if spec["scenario"] == "S5":
        draft.availability = Availability.WITHHELD
        draft.memory = []
    else:
        draft.availability = Availability.AVAILABLE
        for lbl in labels:
            lbl["gold_evidence_ids"] = [f"msg:{short}"]
    # label check: the developer's tests repair exactly this failure set
    ctx.git("checkout", "-q", commit, "--", *tst)
    rc, left, _ = ctx.pytest(["tests"])
    if rc != 0 or left:
        msg = f"{spec['key']}: developer test diff does not make the suite green"
        raise BuildError(msg)
    draft.notes["reference_repair_green"] = True
    ctx.checkout(parent)
    ctx.git("checkout", "-q", commit, "--", *src)
    return draft, {"function": fq, "tests": tst, "src": src, "parent_pass": parent_pass}


def add_mutant(
    ctx: Ctx,
    draft: Draft,
    spec: dict,
    pool: set[str],
    paths: list[str],
    preferred: dict[str, set[str]] | None,
    exclude: set[str],
    accept,
) -> Mutant:
    for path in paths:
        if module_of(path) not in pool:
            continue
        funcs_list = [preferred.get(path)] if preferred and preferred.get(path) else []
        for funcs in [*funcs_list, None]:
            mutant = find_mutant(ctx, path, funcs, exclude, accept)
            if mutant:
                return mutant
    msg = f"{spec['key']}: no acceptable killed mutant found in {paths}"
    raise BuildError(msg)


def apply_mutant_text(ctx: Ctx, mutant: Mutant) -> None:
    (ctx.work / mutant.path).write_text("".join(mutant.new_lines))


def mutant_labels(
    draft: Draft, mutant: Mutant, ids: list[str], mid: str, group: str, **extra
) -> None:
    fq = qualified(module_of(mutant.path), mutant.qualname)
    draft.functions.add(fq)
    draft.modules.add(module_of(mutant.path))
    draft.mutants.append(
        MutantSpec(
            mutant_id=mid,
            function_qualname=fq,
            file_path=mutant.path,
            operator=mutant.desc,
            unified_diff=mutant.diff,
        )
    )
    for tid in ids:
        draft.labels.append(
            dict(
                test_id=tid,
                group_id=group,
                function_qualname=fq,
                injected_mutant_id=mid,
                fix=True,
                **extra,
            )
        )


def build_s3(ctx: Ctx, spec: dict, pool: set[str]) -> Draft:
    draft, info = build_outdated(ctx, spec, pool)
    outdated = set(draft.failing_ids)
    limit = spec["mutant"]["max_failing"]
    mutant = add_mutant(
        ctx,
        draft,
        spec,
        pool,
        spec["mutant"]["paths"],
        None,
        {info["function"].split(".", 2)[-1]},
        lambda extra: 1 <= len(extra - outdated) <= limit,
    )
    apply_mutant_text(ctx, mutant)
    fails, _ = ctx.stable_failures()
    new = set(fails) - outdated
    if not new or not outdated <= set(fails) or not new <= info["parent_pass"]:
        msg = f"{spec['key']}: mixed failure set is inconsistent"
        raise BuildError(msg)
    mutant.new_failures = sorted(new)
    # both halves of the label are verified: developer tests fix only the outdated half
    ctx.git("checkout", "-q", draft.ci_commit, "--", *info["tests"])
    _, after, _ = ctx.pytest(["tests"])
    if set(after) != new:
        msg = f"{spec['key']}: developer tests should leave exactly the mutant failures"
        raise BuildError(msg)
    ctx.git("checkout", "-q", draft.parent, "--", *info["tests"])
    draft.failing = failing_models(fails)
    draft.failing_ids = sorted(fails)
    draft.diffs = split_diff_paths(ctx, draft.parent)
    mutant_labels(draft, mutant, sorted(new), "mut-1", "g2")
    draft.provenance = "tests_with_source_only_change"
    draft.notes.update(mixed_verified=True, mutant=mutant.desc)
    return draft


def probe_for(ctx: Ctx, commit: str, test_id: str) -> tuple[str, str]:
    """Executable probe: the developer's UPDATED test function (source at ``commit``)."""
    path, _, name = test_id.partition("::")
    func = name.split("::")[-1]
    src = ctx.show(commit, path) or ""
    for qual, node in iter_funcs(ast.parse(src)):
        if qual.split(".")[-1] == func:
            code = ast.get_source_segment(src, node)
            if code:
                return code, f"{path}::{name}"
    msg = f"probe test {test_id} not found at {commit[:8]}"
    raise BuildError(msg)


def build_s4(ctx: Ctx, spec: dict, pool: set[str], source: Draft) -> Draft:
    commit = ctx.full(spec["commit"])
    parent = ctx.full(commit + "^")
    if any(
        r.split("\t")[-1].startswith("tests/")
        for r in ctx.rg("diff", "--name-status", parent, commit).splitlines()
    ):
        msg = f"{spec['key']}: S4 base commit must not change tests"
        raise BuildError(msg)
    ctx.checkout(commit)
    rc, _, passed = ctx.pytest(["tests"])
    if rc != 0:
        msg = f"{spec['key']}: S4 base commit is not green"
        raise BuildError(msg)
    src_fn = source.source_function or ""
    module, _, qual = src_fn.removeprefix("boltons.").partition(".")
    source_tests = source.outdated_ids
    # the earliest outdated test becomes the decoy probe; it must still pass on the mutated state
    probe_code, probe_id = probe_for(ctx, source.ci_commit, source_tests[0])
    if probe_id not in passed:
        msg = f"{spec['key']}: probe {probe_id} does not pass on the clean base"
        raise BuildError(msg)
    limit = spec["mutant"]["max_failing"]
    path = f"boltons/{module}.py"
    mutant = find_mutant(
        ctx,
        path,
        {qual},
        set(),
        lambda extra: (
            1 <= len(extra) <= limit and probe_id not in extra and not extra & set(source_tests)
        ),
    )
    if not mutant:
        msg = f"{spec['key']}: no decoy mutant in {src_fn} that spares the probe"
        raise BuildError(msg)
    apply_mutant_text(ctx, mutant)
    fails, passed_m = ctx.stable_failures()
    if probe_id not in passed_m or not set(fails) <= passed:
        msg = f"{spec['key']}: decoy conditions not met on the mutated state"
        raise BuildError(msg)
    modules = {module_of(p) for p in split_diff_paths(ctx, parent)} | {module}
    base = common(ctx, spec, commit, parent, modules)
    base["modules"] = modules
    record = MemoryRecord(
        record_id=f"mem-{source.key}",
        repo_id="repo-a",
        function_qualname=src_fn,
        behaviour_signature=f"{source.source_subject} (approved earlier)",
        direction=RepairDirection.UPDATE_TEST,
        probe=ExecutableProbe(
            probe_id=f"probe-{source.key}", probe_code=probe_code, probe_selector=probe_id
        ),
        evidence_ids=[f"msg:{(source.source_commit or '')[:12]}"],
        approved_at_commit=source.source_commit or "",
    )
    draft = Draft(
        **base,
        diffs=split_diff_paths(ctx, parent),
        failing=failing_models(fails),
        labels=[],
        memory=[record],
        decoys=[record.record_id],
        provenance="tests_with_mutant_applied",
        failing_ids=sorted(fails),
        fault_plan=tool_faults(spec),
        probe_nodeid=probe_id,
        probe_code=probe_code,
    )
    mutant_labels(draft, mutant, sorted(fails), "mut-1", "g1", related=record.record_id)
    draft.notes.update(
        probe_passes_on_mutated_state=True, decoy_function=src_fn, mutant=mutant.desc
    )
    apply_back(ctx, commit)
    return draft


def apply_back(ctx: Ctx, commit: str) -> None:
    ctx.checkout(commit)
    _, left, _ = ctx.pytest(["tests"])
    if left:
        msg = "reverting the mutant did not restore a green suite"
        raise BuildError(msg)


def build_s2(ctx: Ctx, spec: dict, pool: set[str]) -> Draft:
    errors = []
    for cand in spec["candidates"]:
        try:
            return _build_s2(ctx, spec, pool, cand)
        except BuildError as exc:
            errors.append(f"{cand}: {exc}")
    msg = f"{spec['key']}: no candidate worked: {errors}"
    raise BuildError(msg)


def _build_s2(ctx: Ctx, spec: dict, pool: set[str], cand: str) -> Draft:
    commit = ctx.full(cand)
    parent = ctx.full(commit + "^")
    names = [
        r.split("\t")[-1] for r in ctx.rg("diff", "--name-status", parent, commit).splitlines()
    ]
    src = [
        p
        for p in names
        if p.startswith("boltons/") and p.endswith(".py") and not p.endswith("__init__.py")
    ]
    if not src or any(p.startswith("tests/") for p in names):
        msg = "not a source-only commit"
        raise BuildError(msg)
    ctx.checkout(commit)
    rc, _, passed = ctx.pytest(["tests"])
    if rc != 0:
        msg = "base commit not green"
        raise BuildError(msg)
    limit = spec["mutant"]["max_failing"]
    preferred = {p: changed_functions(ctx, parent, commit, p) for p in src}
    scratch = Draft(
        key=spec["key"],
        split=spec["split"],
        scenario="S2",
        ci_commit=commit,
        parent=parent,
        message="",
        diffs={},
        failing=[],
        labels=[],
        corpus=[],
        cutoff_index=0,
    )
    mutant = add_mutant(
        ctx, scratch, spec, pool, src, preferred, set(), lambda extra: 1 <= len(extra) <= limit
    )
    apply_mutant_text(ctx, mutant)
    fails, _ = ctx.stable_failures()
    if not fails or not set(fails) <= passed:
        msg = "mutant failures unstable or not newly introduced"
        raise BuildError(msg)
    modules = {module_of(p) for p in src} | {module_of(mutant.path)}
    base = common(ctx, spec, commit, parent, modules)
    draft = Draft(
        **base,
        diffs=split_diff_paths(ctx, parent),
        failing=failing_models(fails),
        labels=[],
        provenance="tests_with_mutant_applied",
        failing_ids=sorted(fails),
        fault_plan=tool_faults(spec),
    )
    mutant_labels(draft, mutant, sorted(fails), "mut-1", "g1")
    draft.notes.update(
        mutant=mutant.desc,
        mutant_in_changed_function=mutant.qualname in preferred.get(mutant.path, set()),
        restored_green=True,
    )
    apply_back(ctx, commit)
    return draft


# --------------------------------------------------------------------------- #
# Final assembly, verification, output
# --------------------------------------------------------------------------- #


def difficulty(draft: Draft) -> Difficulty:
    if draft.scenario in ("S3", "S4") or len(draft.functions) > 1:
        return Difficulty.HARD
    return (
        Difficulty.EASY
        if len(draft.failing_ids) == 1 and draft.scenario != "S5"
        else Difficulty.MEDIUM
    )


def finalize(ctx: Ctx, draft: Draft, ep_id: str, position: int, env: EnvironmentSpec) -> Episode:
    future = [s for s in ctx.rg("rev-list", draft.ci_commit + "..HEAD").split() if s]
    visible_memory = draft.memory
    labels = []
    for lbl in draft.labels:
        fix = lbl.pop("fix", False)
        related = lbl.pop("related", None)
        action = DecisionLabel.FIX_CODE if fix else DecisionLabel.UPDATE_TEST
        direction = RepairDirection.FIX_CODE if fix else RepairDirection.UPDATE_TEST
        if draft.scenario == "S5":
            action = DecisionLabel.ESCALATE
            lbl["gold_evidence_ids"] = []
        labels.append(
            PerTestGoldLabel(
                gold_direction=direction,
                expected_action=action,
                related_memory_record_id=related,
                memory_applies=False if related else None,
                **lbl,
            )
        )

    def when(sha: str) -> datetime:
        return datetime.fromisoformat(ctx.rg("show", "-s", "--format=%cI", sha).strip())

    ci = CIEvent(
        event_id=f"ci-{ep_id}",
        commit=CommitRef(
            sha=draft.ci_commit, committed_at=when(draft.ci_commit), stream_index=draft.cutoff_index
        ),
        parent_commit=CommitRef(sha=draft.parent, committed_at=when(draft.parent)),
        commit_message=draft.message,
        commit_message_availability=Availability.AVAILABLE,
        changed_files=[
            FileChange(path=p, change_kind=k, unified_diff=d)
            for p, (k, d) in sorted(draft.diffs.items())
        ],
        failing_tests=draft.failing,
    )
    agent_input = EpisodeInput(
        episode_id=ep_id,
        repo=RepoRef(
            repo_id="repo-a",
            repo_url="https://github.com/mahmoud/boltons",
            package_name="boltons",
            license="BSD-3-Clause",
        ),
        environment=env,
        ci_event=ci,
        corpus=RetrievalCorpusSpec(
            corpus_id=f"corpus-{ep_id}",
            corpus_version="pilot-1",
            history_cutoff_commit=draft.ci_commit,
            history_cutoff_index=draft.cutoff_index,
            items=draft.corpus,
        ),
        visible_memory=visible_memory if visible_memory is not None else [],
        edit_policy=POLICY,
        budget=BUDGET,
    )
    scenario = SCENARIO[draft.scenario]
    truth = EpisodeGroundTruth(
        episode_id=ep_id,
        scenario=scenario,
        per_test_labels=labels,
        expected_episode_outcome=DecisionLabel.ESCALATE if draft.scenario == "S5" else None,
        reference_repairs=draft.reference,
        injected_mutants=draft.mutants,
        evidence_availability=draft.availability,
        stripped_evidence=draft.stripped,
        decoy_memory_record_ids=draft.decoys,
        future_commit_shas=future,
        label_provenance=LabelProvenance(
            method=draft.provenance,
            stability_runs=STABILITY_RUNS,
            verified_by="build_dataset.py: 5 identical runs + executed reference repair; "
            "gold_evidence_ids and the S5/S4 constructions pending human review",
        ),
        note=None,
    )
    return Episode(
        episode_id=ep_id,
        scenario=scenario,
        difficulty=difficulty(draft),
        stream_id=f"pilot-{draft.split}",
        stream_position=position,
        agent_input=agent_input,
        ground_truth=truth,
        fault_plan=draft.fault_plan,
    )


def leak_audit(ep: Episode) -> dict:
    payload = ep.agent_payload()
    text = json.dumps(payload, ensure_ascii=False)
    problems = list(find_leaked_ground_truth(payload))
    for sha in ep.ground_truth.future_commit_shas:
        if sha in text or sha[:12] in text:
            problems.append(f"future commit {sha[:12]}")
    for mutant in ep.ground_truth.injected_mutants:
        for needle in (mutant.mutant_id, mutant.operator or ""):
            if needle and needle in text:
                problems.append(f"mutant text {needle!r}")
    for ref in ep.ground_truth.reference_repairs:
        added = [
            ln[1:].strip()
            for ln in ref.unified_diff.splitlines()
            if ln.startswith("+") and not ln.startswith("+++")
        ]
        # a "+" line that also exists as a removed/context line was already visible (old test code in tracebacks)
        known = {
            ln[1:].strip()
            for ln in ref.unified_diff.splitlines()
            if ln[:1] in ("-", " ") and not ln.startswith("---")
        }
        hits = [a for a in added if len(a) >= 25 and a in text and a not in known]
        if hits:
            problems.append(f"reference repair line in agent view: {hits[0][:60]!r}")
    return {"problems": problems, "clean": not problems}


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--repo", required=True)
    parser.add_argument("--python", required=True)
    parser.add_argument(
        "--work", required=True, help="private worktree directory (created if missing)"
    )
    parser.add_argument(
        "--rt-work", help="second worktree for the agent-view round trip (default: <work>-rt)"
    )
    parser.add_argument("--manifest", default=str(HERE / "manifest.json"))
    parser.add_argument("--out", default=str(ROOT / "datasets"))
    args = parser.parse_args()

    manifest = json.loads(Path(args.manifest).read_text())
    ctx = Ctx(Path(args.repo), Path(args.python), Path(args.work))
    head = ctx.full("HEAD")
    if head != manifest["pin"]["head"]:
        msg = f"repo HEAD {head} != pinned {manifest['pin']['head']}"
        raise BuildError(msg)
    freeze = subprocess.run(
        [str(ctx.python), "-m", "pip", "freeze"], capture_output=True, text=True, check=True
    ).stdout
    pytest_version = next(
        (ln.split("==")[1] for ln in freeze.splitlines() if ln.lower().startswith("pytest==")),
        "unknown",
    )
    if pytest_version != manifest["pin"]["pytest"]:
        msg = f"pytest {pytest_version} != pinned {manifest['pin']['pytest']}"
        raise BuildError(msg)
    env = EnvironmentSpec(
        python_version=manifest["pin"]["python"],
        docker_image=None,
        dependency_lock_hash=hashlib.sha256(freeze.encode()).hexdigest(),
        pytest_version=pytest_version,
        test_timeout_seconds=60,
        stability_runs=STABILITY_RUNS,
    )
    pools = {k: set(v) for k, v in manifest["pools"].items()}
    if pools["dev"] & pools["test"]:
        msg = "module pools overlap"
        raise BuildError(msg)

    drafts: dict[str, Draft] = {}
    order = {"S1": 0, "S5": 0, "S3": 0, "S2": 1, "S4": 2}
    for spec in sorted(manifest["episodes"], key=lambda s: order[s["scenario"]]):
        pool = pools[spec["split"]]
        sys.stdout.write(f"building {spec['key']} ({spec['scenario']}) ...\n")
        sys.stdout.flush()
        if spec["scenario"] in ("S1", "S5"):
            draft, _ = build_outdated(ctx, spec, pool)
        elif spec["scenario"] == "S3":
            draft = build_s3(ctx, spec, pool)
        elif spec["scenario"] == "S2":
            draft = build_s2(ctx, spec, pool)
        else:
            draft = build_s4(ctx, spec, pool, drafts[spec["memory_from"]])
        if not draft.modules <= pool:
            msg = f"{spec['key']}: modules {sorted(draft.modules - pool)} outside the {spec['split']} pool"
            raise BuildError(msg)
        drafts[spec["key"]] = draft

    topo = {
        sha: i
        for i, sha in enumerate(ctx.rg("rev-list", "--topo-order", "--reverse", "HEAD").split())
    }
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    episodes: dict[str, list[Episode]] = {"dev": [], "test": []}
    verification: dict[str, dict] = {}
    rt_ctx = Ctx(Path(args.repo), Path(args.python), Path(args.rt_work or f"{args.work}-rt"))
    ids: dict[str, str] = {}
    for split in ("dev", "test"):
        mine = sorted(
            (d for d in drafts.values() if d.split == split), key=lambda d: topo[d.ci_commit]
        )
        for pos, draft in enumerate(mine):
            ids[draft.key] = f"{split}-{pos + 1:02d}"
    for split in ("dev", "test"):
        mine = sorted(
            (d for d in drafts.values() if d.split == split), key=lambda d: topo[d.ci_commit]
        )
        for pos, draft in enumerate(mine):
            ep_id = ids[draft.key]
            if draft.scenario == "S4":
                src_draft = drafts[
                    next(s["memory_from"] for s in manifest["episodes"] if s["key"] == draft.key)
                ]
                if topo[src_draft.ci_commit] >= topo[draft.ci_commit]:
                    msg = f"{draft.key}: decoy memory must come from an earlier episode"
                    raise BuildError(msg)
            ep = finalize(ctx, draft, ep_id, pos, env)
            audit = leak_audit(ep)
            # round trip: rebuild the CI state from the agent view alone and re-run the tests
            payload = ep.agent_payload()
            materialize(rt_ctx.repo, rt_ctx.work, payload)
            _rc, fails, _ = rt_ctx.pytest(["tests"])
            expected = sorted(f.test_id for f in ep.agent_input.ci_event.failing_tests)
            verification[ep_id] = {
                "key": draft.key,
                "scenario": draft.scenario,
                "ci_commit": draft.ci_commit,
                "parent_commit": draft.parent,
                "failing_tests": expected,
                "stable_runs": STABILITY_RUNS,
                "round_trip_reproduces_failures": sorted(fails) == expected,
                "leak_audit": audit,
                "functions": sorted(draft.functions),
                "modules": sorted(draft.modules),
                "corpus_items": len(draft.corpus),
                "notes": draft.notes,
            }
            if not audit["clean"] or sorted(fails) != expected:
                msg = f"{ep_id} ({draft.key}) failed verification: {verification[ep_id]}"
                raise BuildError(msg)
            episodes[split].append(ep)

    # split hygiene
    sets = {
        s: {
            "commits": {e.agent_input.ci_event.commit.sha for e in eps},
            "functions": {
                lbl.function_qualname for e in eps for lbl in e.ground_truth.per_test_labels
            },
            "modules": set().union(*(drafts[k].modules for k, v in ids.items() if v.startswith(s))),
        }
        for s, eps in episodes.items()
    }
    for kind in ("commits", "functions", "modules"):
        if sets["dev"][kind] & sets["test"][kind]:
            msg = f"dev and test share {kind}: {sorted(sets['dev'][kind] & sets['test'][kind])}"
            raise BuildError(msg)

    files: dict[str, str] = {}
    for split, eps in episodes.items():
        full = out_dir / f"pilot_{split}.jsonl"
        view = out_dir / f"pilot_{split}_agent_view.jsonl"
        full.write_text("".join(e.model_dump_json() + "\n" for e in eps), encoding="utf-8")
        view.write_text(
            "".join(json.dumps(e.agent_payload(), ensure_ascii=False) + "\n" for e in eps),
            encoding="utf-8",
        )
        for p in (full, view):
            files[p.name] = hashlib.sha256(p.read_bytes()).hexdigest()
    summary = {
        "pin": manifest["pin"],
        "dependency_lock_hash": env.dependency_lock_hash,
        "splits": {
            s: [
                {
                    "id": e.episode_id,
                    "scenario": e.scenario.value,
                    "difficulty": e.difficulty.value,
                    "ci_commit": e.agent_input.ci_event.commit.sha,
                    "key": verification[e.episode_id]["key"],
                }
                for e in eps
            ]
            for s, eps in episodes.items()
        },
        "module_pools": {k: sorted(v) for k, v in pools.items()},
        "files_sha256": files,
        "frozen_test_split": "datasets/pilot_test*.jsonl: do not tune on it; regenerate only deliberately",
    }
    (out_dir / "pilot_manifest.json").write_text(
        json.dumps(summary, indent=1) + "\n", encoding="utf-8"
    )
    (out_dir / "pilot_verification.json").write_text(
        json.dumps(verification, indent=1) + "\n", encoding="utf-8"
    )
    sys.stdout.write(f"wrote {sum(len(v) for v in episodes.values())} episodes to {out_dir}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
