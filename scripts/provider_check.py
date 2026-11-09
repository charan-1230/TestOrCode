"""Check a provider/model before freezing it: structured-output modes, tool-step validity, returned model id.

    python scripts/provider_check.py --out results/final/provider-check-nim-qwen.json
    python scripts/provider_check.py --env-file .env.mistral --model devstral-2512 --out results/final/provider-check-devstral.json

Uses ``.env`` (or ``--env-file``) for the provider, base URL and key; ``--model`` overrides ``LLM_MODEL``.  Spends
``modes x 4 x repeats`` model calls (24 with the defaults).  Compare the candidates' ``correct_rate`` (the proposal picks
the model with the most reliable tool calling) and freeze the winner's returned, dated model id and its
``recommended_structured_mode`` (``LLM_STRUCTURED_MODE``) in the frozen config.  See ``testorcode.provider_check``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from testorcode.config import Settings
from testorcode.experiment import ROOT
from testorcode.llm_client import create_llm_client
from testorcode.logging_config import configure_logging
from testorcode.provider_check import MODES, run_check


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--env-file", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--modes", default=",".join(MODES), help="comma separated structured-output modes")
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    settings = Settings.from_env(env_file=Path(args.env_file) if args.env_file else ROOT / ".env")
    if args.model:
        settings = settings.model_copy(update={"llm_model": args.model})
    settings.require_llm()
    configure_logging(settings.log_level, secrets=settings.secret_values())
    result = run_check(settings, create_llm_client, modes=[m for m in args.modes.split(",") if m],
                       repeats=args.repeats)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    summary = {m: {"parse_rate": v["parse_rate"], "correct_rate": v["correct_rate"]} for m, v in result["modes"].items()}
    sys.stdout.write(json.dumps({"model_ids_returned": result["model_ids_returned"], "modes": summary,
                                 "recommended_structured_mode": result["recommended_structured_mode"]}, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
