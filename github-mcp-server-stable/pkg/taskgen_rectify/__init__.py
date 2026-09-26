"""taskgen_rectify: a second taskgen agent for the runs the first one rejected.

Loads the rejected runs (``pkg/agenticmcpe/runs/rejected_runs``), re-runs each
task through plan -> execute -> verify with a wide replan budget, consolidates
the attempts into one from-scratch sequence, proves that sequence by
executing and verifying it again, and persists the proven mappings — with
their rectification history — into a separate knowledge base.
"""

from .kb import SharedKnowledgeBase
from .pipeline import DEFAULT_KB_PATH, DEFAULT_REJECTS_PATH, RectifyPipeline
from .sources import (BUCKETS, DEFAULT_REJECTED_RUNS, HARD_MIN_STEPS, RejectedTask,
                      load_rejected_tasks)

__all__ = [
    "BUCKETS",
    "DEFAULT_KB_PATH",
    "DEFAULT_REJECTED_RUNS",
    "DEFAULT_REJECTS_PATH",
    "HARD_MIN_STEPS",
    "RectifyPipeline",
    "RejectedTask",
    "SharedKnowledgeBase",
    "load_rejected_tasks",
]
