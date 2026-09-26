"""Token accounting shared by every agent architecture in this repo.

The point of measuring at the SDK boundary rather than inside each agent is
comparability: the pipeline calls the model through
``pkg.finance_agenticmcpe.llm.LLMClient`` while the ReAct baseline goes
through MCP-Universe's own backends, and neither surfaces usage in its
return value (MCP-Universe's ``_generate`` discards ``chat.usage``
entirely). Wrapping the provider SDK puts the same meter on both, so the
totals are measured identically and can be compared directly.

Usage::

    from pkg.finance_agenticmcpe.token_meter import install, snapshot, reset

    install()                 # idempotent; call once at startup
    reset()                   # zero the counters before a task
    ...run the task...
    usage = snapshot()        # {"calls", "prompt_tokens", "completion_tokens",
                              #  "total_tokens", "reasoning_tokens"}

Counters are process-global because the call sites are spread across
libraries we do not own. Only whole-process or per-task accounting is
meaningful; this is not thread-safe for concurrent tasks.
"""

from __future__ import annotations

from typing import Any

_TOTALS: dict[str, int] = {
    "calls": 0,
    "prompt_tokens": 0,
    "completion_tokens": 0,
    "total_tokens": 0,
    "reasoning_tokens": 0,
}
_installed = False


def reset() -> None:
    """Zero every counter."""
    for k in _TOTALS:
        _TOTALS[k] = 0


def snapshot() -> dict[str, int]:
    """Current totals (a copy — safe to store)."""
    return dict(_TOTALS)


def _record(resp: Any) -> None:
    """Accumulate usage off a provider response, tolerating any shape.

    Never raises: a metering failure must not take down the run it measures.
    """
    try:
        u = getattr(resp, "usage", None)
        if u is None:
            return
        _TOTALS["calls"] += 1
        prompt = getattr(u, "prompt_tokens", None)
        if prompt is None:
            prompt = (
                (getattr(u, "input_tokens", 0) or 0)
                + (getattr(u, "cache_creation_input_tokens", 0) or 0)
                + (getattr(u, "cache_read_input_tokens", 0) or 0)
            )
        completion = getattr(u, "completion_tokens", None)
        if completion is None:
            completion = getattr(u, "output_tokens", 0) or 0
        total = getattr(u, "total_tokens", None) or (prompt + completion)
        _TOTALS["prompt_tokens"] += int(prompt or 0)
        _TOTALS["completion_tokens"] += int(completion or 0)
        _TOTALS["total_tokens"] += int(total or 0)
        details = getattr(u, "completion_tokens_details", None)
        if details is not None:
            _TOTALS["reasoning_tokens"] += int(getattr(details, "reasoning_tokens", 0) or 0)
    except Exception:  # noqa: BLE001 — metering must never break the run.
        pass


def install() -> bool:
    """Patch the provider SDKs to meter every completion. Idempotent."""
    global _installed
    if _installed:
        return True
    patched = False

    try:
        from openai.resources.chat.completions import Completions

        _orig = Completions.create

        def _create(self, *args: Any, **kwargs: Any) -> Any:
            resp = _orig(self, *args, **kwargs)
            _record(resp)
            return resp

        Completions.create = _create  # type: ignore[method-assign]
        patched = True
    except Exception:  # noqa: BLE001 — provider not installed.
        pass

    try:
        from anthropic.resources.messages import Messages

        _orig_msg = Messages.create

        def _msg_create(self, *args: Any, **kwargs: Any) -> Any:
            resp = _orig_msg(self, *args, **kwargs)
            _record(resp)
            return resp

        Messages.create = _msg_create  # type: ignore[method-assign]
        patched = True
    except Exception:  # noqa: BLE001
        pass

    _installed = patched
    return patched


__all__ = ["install", "reset", "snapshot"]
