"""taskgen: benchmark-building agent on top of the agenticmcpe workflow.

Generates realistic GitHub tasks (grounded in the live tool catalog, structured
the way a human formulates work), runs each through the agenticmcpe
plan -> execute -> verify pipeline, and persists only the verified
task -> tool-call-sequence mappings into a JSON knowledge base for RAG.
"""

from .generator import ARCHETYPES, TaskGenError, TaskGenerator, TaskSpec
from .kb import DEFAULT_KB_PATH, KnowledgeBase
from .pipeline import BENCH_REPO_PREFIX, BenchmarkPipeline

__all__ = [
    "ARCHETYPES",
    "TaskGenError",
    "TaskGenerator",
    "TaskSpec",
    "DEFAULT_KB_PATH",
    "KnowledgeBase",
    "BENCH_REPO_PREFIX",
    "BenchmarkPipeline",
]
