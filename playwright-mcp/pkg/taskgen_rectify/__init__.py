"""Rectification taskgen: turn REJECTED long tasks into verified KB entries.

A sibling of ``pkg.taskgen`` (which is left untouched). Where taskgen breeds
NEW tasks and accepts the ones that pass first time, this package re-runs tasks
that were already REJECTED and gives them a large replan budget, so a long,
hard task can be worked into a plan that both executes end-to-end AND passes
verification.

The premise: a task rectified over several replanning rounds is a more valuable
KB precedent than a first-time passer — the surviving sequence encodes the
corrections that the failures taught. Hence the acceptance window
``MIN_REPLANS <= replans <= max_replans``: a run that succeeds with 0 or 1
replans is NOT rectification, and is deliberately not written to the KB.
"""

from .pipeline import RectifyPipeline, RectifyResult
from .selector import RejectRecord, load_pool

__all__ = ["RectifyPipeline", "RectifyResult", "RejectRecord", "load_pool"]
