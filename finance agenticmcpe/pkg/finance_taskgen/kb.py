"""JSON knowledge base of verified finance task -> tool-call-sequence mappings.

Same shape as the github taskgen KB: one file, append-only entries,
atomic writes, kb-NNNN ids.
"""

from __future__ import annotations

import json
import os
import re
import time
from collections import Counter
from pathlib import Path
from typing import Any

DEFAULT_KB_PATH = Path(__file__).resolve().parent / "knowledge_base.json"


def _utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


class KnowledgeBase:
    def __init__(self, path: str | os.PathLike[str] | None = None):
        self.path = Path(path) if path else DEFAULT_KB_PATH
        self.entries: list[dict[str, Any]] = []
        if self.path.is_file():
            raw = json.loads(self.path.read_text(encoding="utf-8"))
            self.entries = list(raw.get("entries", []))

    # ------------------------------------------------------------------ write

    def save(self) -> None:
        payload = {"version": 1, "updated_at": _utc_now(), "entries": self.entries}
        tmp = self.path.with_name(self.path.name + ".tmp")
        tmp.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        os.replace(tmp, self.path)

    def add(self, entry: dict[str, Any]) -> None:
        self.entries.append(entry)
        self.save()

    def next_id(self) -> str:
        mx = 0
        for e in self.entries:
            m = re.match(r"kb-(\d+)$", str(e.get("id", "")))
            if m:
                mx = max(mx, int(m.group(1)))
        return f"kb-{mx + 1:04d}"

    # ------------------------------------------------------------------ query

    def has_prompt(self, prompt: str) -> bool:
        norm = prompt.strip().casefold()
        return any(
            str(e.get("task_prompt", "")).strip().casefold() == norm
            for e in self.entries
        )

    def signature_count(self, signature: tuple[str, ...]) -> int:
        return sum(
            1
            for e in self.entries
            if tuple(e.get("tool_sequence", [])) == signature
        )

    def tool_coverage(self) -> Counter:
        c: Counter = Counter()
        for e in self.entries:
            c.update(e.get("tool_sequence", []))
        return c

    def category_counts(self) -> Counter:
        return Counter(e.get("category", "?") for e in self.entries)

    def summaries(self, limit: int = 30) -> list[str]:
        out = []
        for e in self.entries[-limit:]:
            out.append(e.get("task_summary") or str(e.get("task_prompt", ""))[:160])
        return out

    def prompts(self) -> list[str]:
        return [str(e.get("task_prompt", "")) for e in self.entries]

    def stats(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "entries": len(self.entries),
            "by_category": dict(self.category_counts()),
            "by_difficulty": dict(Counter(e.get("difficulty", "?") for e in self.entries)),
            "tool_coverage": dict(self.tool_coverage().most_common()),
        }


__all__ = ["DEFAULT_KB_PATH", "KnowledgeBase"]