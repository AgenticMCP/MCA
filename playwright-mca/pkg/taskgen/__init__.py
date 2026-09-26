"""taskgen: benchmark-building agent on top of the playwright-agenticmcpe
workflow.

Generates realistic browser tasks (grounded in the live tool catalog,
structured the way a human formulates work), runs each through the agenticmcpe
plan -> execute -> verify pipeline, and persists only the verified
task -> tool-call-sequence mappings into a JSON knowledge base for RAG.
"""

from .generator import ALLOWED_SITES, ARCHETYPES, TaskGenError, TaskGenerator, TaskSpec
from .kb import DEFAULT_KB_PATH, KnowledgeBase
from .pipeline import BenchmarkPipeline

__all__ = [
    "ALLOWED_SITES",
    "ARCHETYPES",
    "TaskGenError",
    "TaskGenerator",
    "TaskSpec",
    "DEFAULT_KB_PATH",
    "KnowledgeBase",
    "BenchmarkPipeline",
]
