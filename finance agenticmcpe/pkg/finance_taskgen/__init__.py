"""Task-generation flywheel for the yahoo_finance MCP server.

Generates realistic finance-analysis tasks, runs them through the
agenticmcpe pipeline, and persists verified task -> tool-sequence
mappings to a knowledge base that the planner can RAG-consult.

Differences from the github taskgen:

* All tasks are READ-ONLY (the yfinance server has no write tools).
* No fork/cleanup machinery — nothing is created on a remote service.
* No plan gate — every tool is read-only by definition.
* Archetypes reflect finance scenarios (return calculation, sector
  survey, valuation comparison, news monitoring, etc.).
"""

from .generator import TaskGenerator, TaskSpec
from .kb import KnowledgeBase, DEFAULT_KB_PATH
from .pipeline import BenchmarkPipeline, PipelineResult

__all__ = [
    "TaskGenerator",
    "TaskSpec",
    "KnowledgeBase",
    "DEFAULT_KB_PATH",
    "BenchmarkPipeline",
    "PipelineResult",
]