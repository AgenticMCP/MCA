#!/usr/bin/env python3
"""Single-task driver: run one prompt file through the full pipeline.

    python3 pkg/agenticmcpe/run/run_task.py <run-id> [prompt-file]

The prompt file defaults to browser_automation/<run-id>.txt (the batch task
layout); it may also be a plain text file whose whole content is the prompt.
Reading prompts from disk avoids shell-escaping long, quote-heavy text.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from pkg.agenticmcpe.config import Settings  # noqa: E402
from pkg.agenticmcpe.orchestrator import Orchestrator  # noqa: E402


def main(argv: list[str]) -> int:
    if not argv:
        print(__doc__)
        return 2
    run_id = argv[0]
    path = Path(argv[1]) if len(argv) > 1 else REPO / "browser_automation" / f"{run_id}.txt"
    if not path.is_file():
        print(f"prompt file not found: {path}")
        return 2
    text = path.read_text(encoding="utf-8")
    _, sep, prompt = text.partition("PROMPT:")
    prompt = prompt.strip() if sep else text.strip()

    settings = Settings.load(run_id=run_id)
    print(f"[task] {run_id}: {prompt[:120]}...")
    result = Orchestrator(settings).run(prompt, max_replans=3)
    print(f"[task] {'OK' if result.ok else 'FAIL'} -> {settings.work_dir}")
    return 0 if result.ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
