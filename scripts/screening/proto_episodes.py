# ruff: noqa  (one-off screening tooling, not part of the testorcode package)
"""Schema-compatibility prototype: build S1-S5 episodes from REAL boltons history and validate them
against the finalized ``testorcode.schemas.episode_schemas.Episode`` (imported read-only).

Real: commits, diffs, test ids, tracebacks, docstrings, mutants (verified killed by running tests).
Synthetic (by design of the benchmark): the S4 memory record and probe, the S5 'uninformative' message,
opaque ids.  Nothing here modifies the schema, its tests, or the TestOrCode repo.
"""

import ast, difflib, json, os, re, subprocess, sys, tempfile, xml.etree.ElementTree as ET

ROOT = "/home/charan/Desktop/4-1/LMA/projects/proj2"
sys.path.insert(0, f"{ROOT}/TestOrCode/TestOrCode/src")
from testorcode.schemas.episode_schemas import *  # noqa

SCR = f"{ROOT}/screening"
REPO, WT = f"{SCR}/boltons", f"{SCR}/wt-boltons-proto"
PY = f"{SCR}/venv-boltons8/bin/python"
ENVV = {k: v for k, v in os.environ.items() if k not in ("FORCE_COLOR", "PY_COLORS")}
ENVV.update(PYTHONPATH=WT, PYTHONDONTWRITEBYTECODE="1", NO_COLOR="1")


def sh(*a, cwd=WT, check=True):
    return subprocess.run(a, cwd=cwd, capture_output=True, text=True, check=check).stdout


def checkout(sha):
    sh("git", "checkout", "-q", "-f", "--detach", sha)
    sh("git", "clean", "-fdxq")


def pytest(args):
    """-> (rc, {nodeid: (message, traceback)} for failures, passed ids)"""
    with tempfile.NamedTemporaryFile(suffix=".xml") as f:
        p = subprocess.run(
            [
                PY,
                "-m",
                "pytest",
                "-q",
                "--color=no",
                "--tb=short",
                "-p",
                "no:cacheprovider",
                "--import-mode=importlib",
                f"--junitxml={f.name}",
                *args,
            ],
            cwd=WT,
            env=ENVV,
            capture_output=True,
            text=True,
            timeout=180,
        )
        fails, passed = {}, set()
        for tc in ET.parse(f.name).getroot().iter("testcase"):
            parts = tc.get("classname").split(".")
            i = next(
                i
                for i in range(len(parts), 0, -1)
                if os.path.exists(f"{WT}/" + "/".join(parts[:i]) + ".py")
            )
            nid = "/".join(parts[:i]) + ".py::" + "::".join([*parts[i:], tc.get("name")])
            bad = tc.find("failure") if tc.find("failure") is not None else tc.find("error")
            if bad is not None:
                fails[nid] = (bad.get("message", "")[:300], (bad.text or "")[:1500])
            elif tc.find("skipped") is None:
                passed.add(nid)
    return p.returncode, fails, passed


def commit_ref(sha):
    ts = int(sh("git", "show", "-s", "--format=%ct", sha, cwd=REPO))
    from datetime import datetime, timezone

    return CommitRef(
        sha=sh("git", "rev-parse", sha, cwd=REPO).strip(),
        committed_at=datetime.fromtimestamp(ts, timezone.utc),
    )


def docstring(path, func, sha):
    src = sh("git", "show", f"{sha}:{path}", cwd=REPO)
    for n in ast.walk(ast.parse(src)):
        if isinstance(n, ast.FunctionDef) and n.name == func:
            return ast.get_docstring(n) or ""
    return ""


def line_mutant(path, func, exclude=()):
    """First killed line-level comparison/arith mutant inside `func` in the CURRENT worktree. -> (diff, desc, killed_ids)"""
    full = f"{WT}/{path}"
    src = open(full).read()
    lines = src.splitlines(keepends=True)
    SW = [
        (" >= ", " > "),
        (" <= ", " < "),
        (" == ", " != "),
        (" > ", " >= "),
        (" < ", " <= "),
        (" + ", " - "),
        (" and ", " or "),
    ]
    nodes = [
        n
        for n in ast.walk(ast.parse(src))
        if isinstance(n, ast.FunctionDef) and (n.name == func if func else n.name not in exclude)
    ]
    assert nodes, f"function {func} not found"
    test_target = "tests/test_" + os.path.basename(path)
    _, base_fails, _ = pytest([test_target])
    base_fails = set(base_fails)
    for ln, fname in [(l, n.name) for n in nodes for l in range(n.lineno, n.end_lineno)]:
        text = lines[ln - 1]
        if text.lstrip().startswith(("#", '"', "'")) or "'''" in text or '"""' in text:
            continue
        for a, b in SW:
            if a in text:
                new = list(lines)
                new[ln - 1] = text.replace(a, b, 1)
                open(full, "w").write("".join(new))
                rc, fails, _ = pytest([test_target, "-x"]) if False else pytest([test_target])
                diff = "".join(difflib.unified_diff(lines, new, f"a/{path}", f"b/{path}", n=2))
                open(full, "w").write(src)
                if rc == 1 and set(fails) - base_fails:
                    return (
                        diff,
                        f"{fname} line {ln}: '{a.strip()}' -> '{b.strip()}'",
                        sorted(set(fails) - base_fails),
                        new,
                        fname,
                    )
    raise RuntimeError(f"no killed mutant found in {func}")


def apply_text(path, new_lines):
    open(f"{WT}/{path}", "w").write("".join(new_lines))


def failing_test(nid, msg, tb, fn):
    return FailingTest(
        test_id=nid,
        test_file=nid.split("::")[0],
        test_name=nid.split("::")[-1],
        target_function=fn,
        failure_message=msg or None,
        traceback=tb or None,
        stable=True,
    )


ENV = EnvironmentSpec(
    python_version="3.12", pytest_version="8.4.2", test_timeout_seconds=60, stability_runs=5
)
POLICY = EditPolicy(allowed_source_roots=("boltons/",), allowed_test_roots=("tests/",))
BUD = EpisodeBudget(
    max_tool_calls=40, max_llm_calls=20, max_tokens=60000, max_revisions=1, wall_clock_seconds=600
)
LP = LabelProvenance(
    method="tests_with_source_only_change", stability_runs=5, verified_by="screening script"
)


def build_s1_parts(c):
    """Parent tree + c's source-only change; returns everything needed for S1/S3/S5."""
    p = sh("git", "rev-parse", c + "^", cwd=REPO).strip()
    c = sh("git", "rev-parse", c, cwd=REPO).strip()
    names = sh("git", "diff", "--name-status", p, c, cwd=REPO).splitlines()
    src = [
        l.split("\t")[1]
        for l in names
        if l.split("\t")[1].startswith("boltons/") and l.endswith(".py")
    ]
    tst = [
        l.split("\t")[1]
        for l in names
        if l.split("\t")[1].startswith("tests/") and l.endswith(".py")
    ]
    return p, c, src, tst


def run_s1_failures(p, c, src, reps=5):
    """Source-only run repeated `reps` times: failing set must be identical (stability filter)."""
    checkout(p)
    sh("git", "checkout", "-q", c, "--", *src)
    sets = []
    for _ in range(reps):
        rc, fails, _ = pytest(["tests"])
        sets.append(frozenset(fails))
    assert len(set(sets)) == 1, "unstable failing set"
    return fails


def episode_s1(c_short, fn_qual, func, srcpath, scenario="S1", message_override=None):
    p, c, src, tst = build_s1_parts(c_short)
    checkout(p)
    _, _, passed_p = pytest(["tests"])
    fails = run_s1_failures(p, c, src)
    assert fails and set(fails) <= passed_p, "failing tests must pass at parent"
    src_diff = sh("git", "diff", p, c, "--", *src, cwd=REPO)
    test_diff = sh("git", "diff", p, c, "--", *tst, cwd=REPO)
    msg = sh("git", "log", "-1", "--format=%B", c, cwd=REPO).strip()
    doc = docstring(srcpath, func, p)
    ev = [
        EvidenceItem(
            evidence_id="ev-doc",
            source_type=EvidenceSourceType.DOCSTRING,
            source_ref=f"{srcpath}::{func}",
            commit_sha=p,
            text=doc[:400] or None,
        )
    ]
    if message_override is None:
        ev.insert(
            0,
            EvidenceItem(
                evidence_id="ev-msg",
                source_type=EvidenceSourceType.COMMIT_MESSAGE,
                source_ref=c[:12],
                commit_sha=c,
                text=msg,
            ),
        )
    ft = [failing_test(n, m, t, fn_qual) for n, (m, t) in sorted(fails.items())]
    return dict(
        p=p,
        c=c,
        src=src,
        tst=tst,
        src_diff=src_diff,
        test_diff=test_diff,
        msg=msg,
        ev=ev,
        ft=ft,
        fails=fails,
    )


def ci_event(d, extra_ft=(), extra_diff="", msg="__keep__", avail=Availability.AVAILABLE):
    ft = [*d["ft"], *extra_ft]
    groups = {}
    for t in ft:
        groups.setdefault(t.target_function, []).append(t.test_id)
    return CIEvent(
        event_id="ci-" + d["c"][:8],
        commit=commit_ref(d["c"]),
        parent_commit=commit_ref(d["p"]),
        commit_message=(d["msg"] if msg == "__keep__" else msg)
        if avail is Availability.AVAILABLE
        else None,
        commit_message_availability=avail,
        changed_files=[
            FileChange(
                path=d["src"][0], change_kind="modified", unified_diff=d["src_diff"] + extra_diff
            )
        ],
        failing_tests=ft,
        failure_groups=[
            FailureGroup(group_id=f"g{i}", function_qualname=fq, test_ids=ids)
            for i, (fq, ids) in enumerate(groups.items())
        ],
    )


def build_all():
    out = {}
    # ---- S1: rotate_file (real outdated tests) ------------------------------------------------------------------
    RF = "boltons.fileutils.rotate_file"
    d = episode_s1("1c2db466", RF, "rotate_file", "boltons/fileutils.py")
    gold = [
        PerTestGoldLabel(
            test_id=t.test_id,
            group_id="g0",
            function_qualname=RF,
            gold_direction=RepairDirection.UPDATE_TEST,
            expected_action=DecisionLabel.UPDATE_TEST,
            gold_evidence_ids=["ev-msg"],
        )
        for t in d["ft"]
    ]
    ref = ReferenceRepair(unified_diff=d["test_diff"], changed_files=d["tst"], source_commit=d["c"])

    def agent_input(eid, ev, event, mem=None):
        return EpisodeInput(
            episode_id=eid,
            repo=RepoRef(
                repo_id="lib-b",
                repo_url="https://github.com/mahmoud/boltons",
                package_name="boltons",
                license="BSD-3-Clause",
            ),
            environment=ENV,
            ci_event=event,
            corpus=RetrievalCorpusSpec(
                corpus_id="lib-b-corpus", history_cutoff_commit=event.commit.sha, items=ev
            ),
            visible_memory=mem,
            edit_policy=POLICY,
            budget=BUD,
        )

    out["S1"] = Episode(
        episode_id="ep-001",
        scenario=ScenarioType.S1_OUTDATED_TEST,
        stream_id="stream-b",
        stream_position=0,
        agent_input=agent_input("ep-001", d["ev"], ci_event(d)),
        ground_truth=EpisodeGroundTruth(
            episode_id="ep-001",
            scenario=ScenarioType.S1_OUTDATED_TEST,
            per_test_labels=gold,
            reference_repairs=[ref],
            evidence_availability=Availability.AVAILABLE,
            label_provenance=LP,
        ),
    )
    # ---- S5: same base, message replaced, no usable intent evidence --------------------------------------------
    uninf = ci_event(d, msg="update fileutils")
    ev5 = [
        EvidenceItem(
            evidence_id="ev-msg-stub",
            source_type=EvidenceSourceType.COMMIT_MESSAGE,
            source_ref=d["c"][:12],
            commit_sha=d["c"],
            text="update fileutils",
        ),
        d["ev"][1],
    ]
    gold5 = [
        PerTestGoldLabel(
            test_id=t.test_id,
            group_id="g0",
            function_qualname=RF,
            gold_direction=RepairDirection.UPDATE_TEST,
            expected_action=DecisionLabel.ESCALATE,
            gold_evidence_ids=[],
        )
        for t in d["ft"]
    ]
    out["S5"] = Episode(
        episode_id="ep-005",
        scenario=ScenarioType.S5_INTENT_STRIPPED,
        stream_id="stream-b",
        stream_position=4,
        agent_input=agent_input("ep-005", ev5, uninf, mem=[]),
        ground_truth=EpisodeGroundTruth(
            episode_id="ep-005",
            scenario=ScenarioType.S5_INTENT_STRIPPED,
            per_test_labels=gold5,
            expected_episode_outcome=DecisionLabel.ESCALATE,
            reference_repairs=[ref],
            evidence_availability=Availability.WITHHELD,
            stripped_evidence=[
                StrippedEvidence(
                    source_type=EvidenceSourceType.COMMIT_MESSAGE,
                    source_ref=d["c"][:12],
                    original_evidence_id="ev-msg",
                )
            ],
            label_provenance=LP,
        ),
    )
    # ---- S3: S1 + mutant in a DIFFERENT function of the same file ---------------------------------------------
    checkout(d["p"])
    sh("git", "checkout", "-q", d["c"], "--", *d["src"])
    mdiff, mdesc, mkilled, mnew, mfn = line_mutant(
        "boltons/fileutils.py", None, exclude=("rotate_file",)
    )
    apply_text("boltons/fileutils.py", mnew)
    rc, f3, _ = pytest(["tests"])
    new_fail = {k: v for k, v in f3.items() if k not in d["fails"]}
    assert new_fail, "mutant must create failures beyond the outdated tests"
    MK = "boltons.fileutils." + mfn
    mut = MutantSpec(
        mutant_id="mut-1",
        function_qualname=MK,
        file_path="boltons/fileutils.py",
        operator=mdesc,
        unified_diff=mdiff,
    )
    ft3 = [*d["ft"], *[failing_test(n, m, t, MK) for n, (m, t) in sorted(new_fail.items())]]
    ev3 = ci_event(d, extra_ft=ft3[len(d["ft"]) :], extra_diff=mdiff)
    gold3 = [
        *gold,
        *[
            PerTestGoldLabel(
                test_id=n,
                group_id="g1",
                function_qualname=MK,
                gold_direction=RepairDirection.FIX_CODE,
                expected_action=DecisionLabel.FIX_CODE,
                injected_mutant_id="mut-1",
            )
            for n in sorted(new_fail)
        ],
    ]
    out["S3"] = Episode(
        episode_id="ep-003",
        scenario=ScenarioType.S3_MIXED,
        stream_id="stream-b",
        stream_position=2,
        agent_input=agent_input("ep-003", d["ev"], ev3),
        ground_truth=EpisodeGroundTruth(
            episode_id="ep-003",
            scenario=ScenarioType.S3_MIXED,
            per_test_labels=gold3,
            reference_repairs=[ref],
            injected_mutants=[mut],
            evidence_availability=Availability.AVAILABLE,
            label_provenance=LP,
        ),
    )
    # ---- S2: meaningful source-only commit (bba729bf MultiReplace) + mutant a test kills -----------------------
    p2 = sh("git", "rev-parse", "bba729bf^", cwd=REPO).strip()
    c2 = sh("git", "rev-parse", "bba729bf", cwd=REPO).strip()
    checkout(c2)
    mdiff2, mdesc2, killed2, new2, fn2 = line_mutant("boltons/strutils.py", None)
    apply_text("boltons/strutils.py", new2)
    _, f2, _ = pytest(["tests"])
    FQ2 = "boltons.strutils." + fn2
    commit_diff2 = sh("git", "diff", p2, c2, "--", "boltons/strutils.py", cwd=REPO)
    msg2 = sh("git", "log", "-1", "--format=%B", c2, cwd=REPO).strip()
    ev2 = CIEvent(
        event_id="ci-" + c2[:8],
        commit=commit_ref(c2),
        parent_commit=commit_ref(p2),
        commit_message=msg2,
        changed_files=[
            FileChange(
                path="boltons/strutils.py",
                change_kind="modified",
                unified_diff=commit_diff2 + mdiff2,
            )
        ],
        failing_tests=[failing_test(n, m, t, FQ2) for n, (m, t) in sorted(f2.items())],
        failure_groups=[FailureGroup(group_id="g0", function_qualname=FQ2, test_ids=sorted(f2))],
    )
    mut2 = MutantSpec(
        mutant_id="mut-2",
        function_qualname=FQ2,
        file_path="boltons/strutils.py",
        operator=mdesc2,
        unified_diff=mdiff2,
    )
    out["S2"] = Episode(
        episode_id="ep-002",
        scenario=ScenarioType.S2_INJECTED_REGRESSION,
        stream_id="stream-b",
        stream_position=1,
        agent_input=agent_input(
            "ep-002",
            [
                EvidenceItem(
                    evidence_id="ev-msg2",
                    source_type=EvidenceSourceType.COMMIT_MESSAGE,
                    source_ref=c2[:12],
                    commit_sha=c2,
                    text=msg2,
                )
            ],
            ev2,
        ),
        ground_truth=EpisodeGroundTruth(
            episode_id="ep-002",
            scenario=ScenarioType.S2_INJECTED_REGRESSION,
            per_test_labels=[
                PerTestGoldLabel(
                    test_id=n,
                    group_id="g0",
                    function_qualname=FQ2,
                    gold_direction=RepairDirection.FIX_CODE,
                    expected_action=DecisionLabel.FIX_CODE,
                    injected_mutant_id="mut-2",
                )
                for n in sorted(f2)
            ],
            injected_mutants=[mut2],
            evidence_availability=Availability.AVAILABLE,
            label_provenance=LabelProvenance(
                method="tests_with_mutant_applied", stability_runs=5, verified_by="screening script"
            ),
        ),
    )
    # ---- S4: later in stream, mutant in rotate_file (approved earlier) -> memory must not justify an update ----
    later = sh("git", "rev-parse", "bd46e5ab", cwd=REPO).strip()
    later_p = sh("git", "rev-parse", "bd46e5ab^", cwd=REPO).strip()
    checkout(later)
    mdiff4, mdesc4, killed4, new4, fn4 = line_mutant("boltons/fileutils.py", "rotate_file")
    apply_text("boltons/fileutils.py", new4)
    _, f4, _ = pytest(["tests"])
    # decoy memory (SYNTHETIC by construction): the earlier approved rotate_file change, with a probe about generation count
    rec = MemoryRecord(
        record_id="mem-rotate-1",
        repo_id="lib-b",
        function_qualname=RF,
        behaviour_signature="rotate_file(keep=N) retains N generations",
        direction=RepairDirection.UPDATE_TEST,
        probe=ExecutableProbe(
            probe_id="probe-1", probe_code="assert generations_after_rotate(keep=5) == 5"
        ),
        evidence_ids=["ev-msg"],
        approved_at_commit=d["c"],
    )
    msg4 = sh("git", "log", "-1", "--format=%B", later, cwd=REPO).strip()
    diff4 = sh("git", "diff", later_p, later, cwd=REPO) + mdiff4
    ev4 = CIEvent(
        event_id="ci-" + later[:8],
        commit=commit_ref(later),
        parent_commit=commit_ref(later_p),
        commit_message=msg4,
        changed_files=[
            FileChange(path="boltons/timeutils.py", change_kind="modified", unified_diff=diff4)
        ],
        failing_tests=[failing_test(n, m, t, RF) for n, (m, t) in sorted(f4.items())],
        failure_groups=[FailureGroup(group_id="g0", function_qualname=RF, test_ids=sorted(f4))],
    )
    out["S4"] = Episode(
        episode_id="ep-004",
        scenario=ScenarioType.S4_DECOY_REGRESSION,
        stream_id="stream-b",
        stream_position=3,
        agent_input=agent_input(
            "ep-004",
            [
                EvidenceItem(
                    evidence_id="ev-msg",
                    source_type=EvidenceSourceType.COMMIT_MESSAGE,
                    source_ref=d["c"][:12],
                    commit_sha=d["c"],
                    text=d["msg"],
                ),
                EvidenceItem(
                    evidence_id="ev-msg4",
                    source_type=EvidenceSourceType.COMMIT_MESSAGE,
                    source_ref=later[:12],
                    commit_sha=later,
                    text=msg4,
                ),
            ],
            ev4,
            mem=[rec],
        ),
        ground_truth=EpisodeGroundTruth(
            episode_id="ep-004",
            scenario=ScenarioType.S4_DECOY_REGRESSION,
            per_test_labels=[
                PerTestGoldLabel(
                    test_id=n,
                    group_id="g0",
                    function_qualname=RF,
                    gold_direction=RepairDirection.FIX_CODE,
                    expected_action=DecisionLabel.FIX_CODE,
                    injected_mutant_id="mut-4",
                    related_memory_record_id="mem-rotate-1",
                    memory_applies=False,
                )
                for n in sorted(f4)
            ],
            injected_mutants=[
                MutantSpec(
                    mutant_id="mut-4",
                    function_qualname=RF,
                    file_path="boltons/fileutils.py",
                    operator=mdesc4,
                    unified_diff=mdiff4,
                )
            ],
            decoy_memory_record_ids=["mem-rotate-1"],
            evidence_availability=Availability.AVAILABLE,
            label_provenance=LabelProvenance(
                method="tests_with_mutant_applied", stability_runs=5, verified_by="screening script"
            ),
        ),
    )
    return out


if __name__ == "__main__":
    eps = build_all()
    for k in sorted(eps):
        e = eps[k]
        payload = e.agent_payload()
        leaks = find_leaked_ground_truth(payload)
        print(
            f"{k}: VALID  tests={len(e.agent_input.ci_event.failing_tests)} labels={[str(l.expected_action) for l in e.ground_truth.per_test_labels][:3]} leak_audit={leaks or 'clean'}"
        )
    json.dump(
        {k: json.loads(v.model_dump_json()) for k, v in eps.items()},
        open(f"{SCR}/runs/proto_episodes.json", "w"),
        indent=1,
    )
