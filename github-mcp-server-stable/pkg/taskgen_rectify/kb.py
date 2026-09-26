"""A KnowledgeBase that several rectify processes may append to at once.

``pkg.taskgen.kb.KnowledgeBase`` keeps its entries in memory and rewrites the
whole file on every ``add``; two runs sharing one file (say a readonly batch
and a write batch) silently overwrite each other's entries and hand out the
same id. Here ``add`` serialises the read-modify-write behind a lock file and
re-reads the file first, so the append always lands on the latest state.
"""

from __future__ import annotations

import fcntl
import json
from typing import Any

from pkg.taskgen.kb import KnowledgeBase


class SharedKnowledgeBase(KnowledgeBase):
    def add(self, entry: dict[str, Any]) -> None:
        lock = self.path.with_name(self.path.name + ".lock")
        with lock.open("w") as fh:
            fcntl.flock(fh, fcntl.LOCK_EX)
            try:
                if self.path.is_file():
                    raw = json.loads(self.path.read_text(encoding="utf-8"))
                    self.entries = list(raw.get("entries", []))
                entry["id"] = self.next_id()  # from the fresh state, in place
                super().add(entry)
            finally:
                fcntl.flock(fh, fcntl.LOCK_UN)


__all__ = ["SharedKnowledgeBase"]
