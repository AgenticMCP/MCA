"""Count LLM tokens for a whole benchmark round, identically for both agents.

Both arms reach the provider through the same SDK entry point, so this
patches that method and accumulates every response's `usage`. Nothing in
`pkg/agenticmcpe` or in mcpuniverse has to cooperate, which is the point: a
meter either side could tamper with would not be evidence.

Why not read usage where each agent already sits? The pipeline's `LLMClient`
returns only the message text, and mcpuniverse's backend builds a fresh
client per call and throws the response away. Patching the SDK is the only
vantage point that sees both.

    from token_meter import TokenMeter

    meter = TokenMeter().install()
    ...
    before = meter.snapshot()
    run_one_task()
    print(meter.delta(before))     # {'calls': 12, 'prompt': ..., 'total': ...}

A response without `usage` (streaming, or a provider that omits it) increments
`calls` and `calls_without_usage` but no token totals — so an undercount is
visible rather than silent. Stdlib + whatever SDK is already installed; imports
nothing new.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace


@dataclass(frozen=True)
class TokenCounts:
    calls: int = 0
    calls_without_usage: int = 0
    prompt: int = 0
    completion: int = 0
    total: int = 0


class TokenMeter:
    """Process-wide token accumulator. `install()` is idempotent per instance."""

    def __init__(self) -> None:
        self.counts = TokenCounts()
        self._installed: list[tuple[object, str, object]] = []

    # ------------------------------------------------------------------ record
    def _record(self, response) -> None:
        usage = getattr(response, "usage", None)
        if usage is None:
            self.counts = replace(self.counts, calls=self.counts.calls + 1,
                                  calls_without_usage=self.counts.calls_without_usage + 1)
            return
        prompt = int(getattr(usage, "prompt_tokens", 0)
                     or getattr(usage, "input_tokens", 0) or 0)
        completion = int(getattr(usage, "completion_tokens", 0)
                         or getattr(usage, "output_tokens", 0) or 0)
        total = int(getattr(usage, "total_tokens", 0) or 0) or (prompt + completion)
        self.counts = replace(
            self.counts,
            calls=self.counts.calls + 1,
            prompt=self.counts.prompt + prompt,
            completion=self.counts.completion + completion,
            total=self.counts.total + total,
        )

    # ----------------------------------------------------------------- install
    def _patch(self, cls, name: str) -> None:
        original = getattr(cls, name)
        meter = self

        def wrapper(self, *args, **kwargs):  # noqa: ANN001 - mirrors the SDK method
            response = original(self, *args, **kwargs)
            try:
                meter._record(response)
            except Exception:  # noqa: BLE001 - metering must never break a run
                pass
            return response

        wrapper.__wrapped__ = original
        setattr(cls, name, wrapper)
        self._installed.append((cls, name, original))

    def install(self) -> "TokenMeter":
        """Wrap every SDK entry point we can find. Missing SDKs are skipped."""
        try:
            from openai.resources.chat.completions import Completions
            self._patch(Completions, "create")
        except Exception:  # noqa: BLE001
            pass
        try:
            from anthropic.resources.messages import Messages
            self._patch(Messages, "create")
        except Exception:  # noqa: BLE001
            pass
        try:
            from openai.resources.responses import Responses
            self._patch(Responses, "create")
        except Exception:  # noqa: BLE001
            pass
        return self

    def uninstall(self) -> None:
        for cls, name, original in reversed(self._installed):
            setattr(cls, name, original)
        self._installed.clear()

    @property
    def active(self) -> bool:
        return bool(self._installed)

    # ---------------------------------------------------------------- readings
    def snapshot(self) -> dict[str, int]:
        return asdict(self.counts)

    def delta(self, before: dict[str, int]) -> dict[str, int]:
        """Tokens spent since `before` — the per-task reading."""
        now = self.snapshot()
        return {k: now[k] - before.get(k, 0) for k in now}


__all__ = ["TokenMeter", "TokenCounts"]
