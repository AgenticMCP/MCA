"""agenticmcpe — an agentic workflow that replaces a human driving the GitHub
MCP server: plan a tool-call sequence, execute it against the real Go server,
and verify the outcome with execution-based evaluators.

Three agents over the existing ``pkg.mcp_wrapper`` execution layer:

* :class:`~pkg.agenticmcpe.planner.PlannerAgent`   — task prompt -> grounded
  s0..sn tool-call sequence (+ post-action properties) + summary.
* :class:`~pkg.agenticmcpe.executor.ExecutionAgent` — strict, retrying, fully
  logged replay of the sequence.
* :class:`~pkg.agenticmcpe.verifier.VerifierAgent`  — generates + runs a script
  with Format, Static and Dynamic evaluators.

:class:`~pkg.agenticmcpe.config.Settings` is the single base-util config
(multi-provider LLM client + GitHub token pool).
"""

from .config import (
    ConfigError,
    GitHubTokenPool,
    LLMClient,
    LLMSettings,
    Settings,
)
from .executor import ExecutionAgent, ExecutionTrace
from .orchestrator import Orchestrator, RunResult, selfcheck
from .planner import Plan, PlannerAgent, PlanStep
from .verifier import VerificationReport, VerifierAgent

__all__ = [
    "Settings",
    "LLMSettings",
    "LLMClient",
    "GitHubTokenPool",
    "ConfigError",
    "PlannerAgent",
    "Plan",
    "PlanStep",
    "ExecutionAgent",
    "ExecutionTrace",
    "VerifierAgent",
    "VerificationReport",
    "Orchestrator",
    "RunResult",
    "selfcheck",
]
