"""Embed the documents of the dev stream's corpora (boltons pilot, ``pilot_dev``) into the Voyage cache, so a dev pass with
``RETRIEVAL_MODE=hybrid`` only embeds the agents' queries live.  Resumable (the cache is append-only); no LLM.

    python scripts/heldout/warm_dev_embeddings.py [--split dev] [--dataset pilot]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from testorcode.config import Settings  # noqa: E402
from testorcode.dataset import agent_view_path, load_agent_inputs  # noqa: E402
from testorcode.freeze import load_frozen  # noqa: E402
from testorcode.retrieval_hybrid import _document_text, build_components, estimate_tokens  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", default="dev")
    parser.add_argument("--dataset", default="pilot")
    args = parser.parse_args()
    frozen, _ = load_frozen(ROOT / "configs" / "final.json")
    settings = frozen.apply(Settings.from_env(env_file=ROOT / ".env"))
    if settings.embedding_cache_dir is not None and not settings.embedding_cache_dir.is_absolute():
        settings = settings.model_copy(update={"embedding_cache_dir": ROOT / settings.embedding_cache_dir})
    embedder, _ = build_components(settings)
    docs: dict[str, None] = {}
    for episode in load_agent_inputs(agent_view_path(args.split, dataset=args.dataset)):
        for item in episode.corpus.items or []:
            docs[_document_text(item)] = None
    texts = list(docs)
    sys.stdout.write(f"{len(texts)} unique documents, ~{sum(estimate_tokens(t) for t in texts)} est. tokens\n")
    for start in range(0, len(texts), 160):
        embedder.embed(texts[start:start + 160], input_type="document")
        sys.stdout.write(f"{min(start + 160, len(texts))}/{len(texts)}\n")
        sys.stdout.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
