"""Agentic workflow for the finance MCP server.

This is a clean-room port of `github-mcp-server-stable/pkg/agenticmcpe`,
adapted to the yfinance server's read-only surface. See
`pkg/finance_mcp_wrapper/README.md` for the seam mapping and
`financial_analysis/` (at the repo root) for
task prompts used to seed the flywheel.

Public surface used by ``taskgen`` and the runnable entry points:

* :class:`AgenticConfig`     — runtime configuration
* :class:`Orchestrator`      — planner → executor → verifier loop
* :func:`verify`             — verify a single task independently
* :func:`plan`               — generate a plan for a task independently
* :func:`execute_sequence`   — execute a sequence without planning
* :mod:`catalog`             — tool catalog management
* :mod:`toolsource`          — extract tool definitions from a Python source file
* :mod:`rag`                 — RAG with finance benign literals
* :class:`RunResult`         — the value returned from a workflow run
"""

from __future__ import annotations

from .config import AgenticConfig, default_config
from .orchestrator import Orchestrator, RunResult, verify, plan, execute_sequence

__all__ = [
    "AgenticConfig",
    "default_config",
    "Orchestrator",
    "RunResult",
    "verify",
    "plan",
    "execute_sequence",
]