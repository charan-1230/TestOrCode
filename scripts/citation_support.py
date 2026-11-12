"""Citation support: does each cited evidence item support the claim it is cited for?  (Proposal Table 4,
"citation faithfulness".)  Needs labels, so it is a three-step, human-first procedure:

    # 1. one CSV row per (run, test, cited id): the claim and the cited text; the "human_label" column is empty
    python scripts/citation_support.py export results/<exp>/runs.jsonl --out citations.csv
    # 2. humans fill "human_label" with supports / does_not_support / unclear (blind to the condition column if possible)
    # 3. (optional) an LLM judge labels every row into "judge_label" (spends API calls; uses the .env model)
    python scripts/citation_support.py judge citations.csv
    # 4. rates per condition from the human labels, and the judge's agreement with them (accuracy, Cohen's kappa);
    #    also written to citations.score.md next to the CSV (the final report includes it)
    python scripts/citation_support.py score citations.csv

The judge is never the primary measurement: ``score`` reports judge-based rates only next to its measured agreement
with the human labels on the rows both labelled (course rule: an LLM judge must be validated against human labels).
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from testorcode.dataset import episode_path, load_episodes
from testorcode.experiment import read_records

LABELS = ("supports", "does_not_support", "unclear")
COLUMNS = ("row_id", "run_id", "condition", "episode_id", "test_id", "cited_id", "claim", "cited_text",
           "human_label", "judge_label")
JUDGE_SYSTEM = """You check citations in a CI-triage system.  A claim about a failing test (whether the test is outdated
because behaviour changed on purpose, or the code regressed) cites one evidence item.  Decide only whether THIS item, as
written, supports the claim.  "supports": the item states or clearly implies what the claim relies on.
"does_not_support": it is unrelated, too generic, or says the opposite.  "unclear": you cannot tell from the text.
Reply with one JSON object matching the schema."""


class JudgeVerdict(BaseModel):
    model_config = ConfigDict(extra="forbid")

    label: Literal["supports", "does_not_support", "unclear"]
    reason: str = Field(min_length=1)


def export(runs: Path, out: Path, dataset: str) -> int:
    records = read_records(runs)
    split = {r.provenance.split for r in records}.pop()
    episodes = {e.episode_id: e for e in load_episodes(episode_path(split, dataset=dataset))}
    rows = []
    for rec in records:
        episode = episodes.get(rec.episode_id)
        corpus = {i.evidence_id: i.text or "" for i in (episode.agent_input.corpus.items or [])} if episode else {}
        seen = {e.evidence_id: e.snippet or "" for e in rec.retrieved_evidence}
        for p in rec.predictions:
            claim = " | ".join(x for x in (f"hypothesis: {p.hypothesis.value}" if p.hypothesis else "",
                                           f"decision: {p.predicted_action.value}",
                                           f"expected behaviour: {p.predicted_behaviour}" if p.predicted_behaviour else "",
                                           f"rationale: {p.rationale}" if p.rationale else "") if x)
            for cited in p.cited_evidence_ids:
                text = corpus.get(cited) or seen.get(cited) or ""
                rows.append({"row_id": len(rows) + 1, "run_id": rec.provenance.run_id, "condition": rec.condition.value,
                             "episode_id": rec.episode_id, "test_id": p.test_id, "cited_id": cited,
                             "claim": claim[:2000], "cited_text": text[:3000], "human_label": "", "judge_label": ""})
    with out.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    sys.stdout.write(f"{len(rows)} citation rows -> {out}\n")
    return 0


def _read(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def judge(path: Path) -> int:
    from testorcode.config import Settings
    from testorcode.llm_client import create_llm_client
    from testorcode.schemas.result_schemas import AgentRole

    settings = Settings.from_env(env_file=ROOT / ".env")
    llm = create_llm_client(settings)
    rows = _read(path)
    for row in rows:
        if row.get("judge_label") in LABELS:
            continue  # resumable
        user = f"CLAIM:\n{row['claim']}\n\nCITED ITEM [{row['cited_id']}]:\n{row['cited_text'] or '(no text available)'}"
        verdict = llm.complete_structured(role=AgentRole.INTEGRITY_GUARD, system=JUDGE_SYSTEM, user=user,
                                          response_model=JudgeVerdict).parsed
        row["judge_label"] = verdict.label
        with path.open("w", newline="", encoding="utf-8") as fh:  # checkpoint after every row
            writer = csv.DictWriter(fh, fieldnames=COLUMNS)
            writer.writeheader()
            writer.writerows(rows)
    sys.stdout.write(f"judged {len(rows)} rows with {settings.llm_model}\n")
    return 0


def cohen_kappa(pairs: list[tuple[str, str]]) -> float | None:
    """Cohen's kappa of two raters over ``pairs``; None when undefined (no pairs, or chance agreement is 1)."""
    if not pairs:
        return None
    n = len(pairs)
    observed = sum(a == b for a, b in pairs) / n
    ca, cb = Counter(a for a, _ in pairs), Counter(b for _, b in pairs)
    expected = sum(ca[k] * cb[k] for k in set(ca) | set(cb)) / (n * n)
    return None if expected == 1 else (observed - expected) / (1 - expected)


def _rate(labels: list[str]) -> tuple[int, int]:
    decided = [x for x in labels if x in ("supports", "does_not_support")]
    return sum(x == "supports" for x in decided), len(decided)


def score(path: Path) -> int:
    rows = _read(path)
    human = defaultdict(list)
    judged = defaultdict(list)
    for row in rows:
        if row["human_label"] in LABELS:
            human[row["condition"]].append(row["human_label"])
        if row["judge_label"] in LABELS:
            judged[row["condition"]].append(row["judge_label"])
    both = [(r["human_label"], r["judge_label"]) for r in rows if r["human_label"] in LABELS and r["judge_label"] in LABELS]
    out = ["| Condition | Citation support (human) | Citation support (judge) |", "|---|---|---|"]
    for cond in sorted(set(human) | set(judged)):
        hk, hn = _rate(human[cond])
        jk, jn = _rate(judged[cond])
        out.append(f"| {cond} | {f'{hk}/{hn} ({hk / hn:.2f})' if hn else '-'} | {f'{jk}/{jn} ({jk / jn:.2f})' if jn else '-'} |")
    kappa = cohen_kappa(both)
    agreement = sum(a == b for a, b in both) / len(both) if both else None
    out += ["", f"Human-labelled rows: {sum(len(v) for v in human.values())} of {len(rows)}. "
                f"Judge agreement with humans on {len(both)} rows: "
                + ("-" if agreement is None else f"accuracy {agreement:.2f}, Cohen's kappa "
                   + ("-" if kappa is None else f"{kappa:.2f}")) + ". Rates exclude 'unclear'."]
    path.with_suffix(".score.md").write_text("\n".join(out) + "\n", encoding="utf-8")
    sys.stdout.write("\n".join(out) + "\n")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    e = sub.add_parser("export")
    e.add_argument("runs", type=Path)
    e.add_argument("--out", type=Path, required=True)
    e.add_argument("--dataset", default="pilot")
    sub.add_parser("judge").add_argument("csv", type=Path)
    sub.add_parser("score").add_argument("csv", type=Path)
    args = parser.parse_args(argv)
    if args.command == "export":
        return export(args.runs, args.out, args.dataset)
    if args.command == "judge":
        return judge(args.csv)
    return score(args.csv)


if __name__ == "__main__":
    raise SystemExit(main())
