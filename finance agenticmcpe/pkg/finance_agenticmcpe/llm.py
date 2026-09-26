"""LLM client for the planner, executor, verifier, and RAG rewriter.

Server-agnostic.
The four roles use the same transport; only the system prompt and the
temperature / model differ. We keep the surface tiny so that tests can
swap in a mock without subclassing a fat client.

Quick start::

    from pkg.finance_agenticmcpe.llm import LLMClient, Message

    client = LLMClient.from_env()
    text = client.complete(
        system="You are a planner.",
        messages=[Message("user", "Plan a tool call sequence that...")],
        max_tokens=2048,
        temperature=0.0,
    )

The :class:`Message` dataclass mirrors the SDK's input shape (role +
content); we use it to keep the rest of the package decoupled from the
SDK type so a mock or alternate backend can be plugged in.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Iterable, Sequence


@dataclass
class Message:
    role: str
    content: str


class LLMError(Exception):
    """A failure in the LLM transport (HTTP, SDK, malformed reply, ...)."""


# Ceiling for the escalate-on-truncation path in ``LLMClient.complete``.
_MAX_OUTPUT_BUDGET = 65536


class LLMClient:
    """Thin wrapper over the SDK. The constructor accepts the
    API key explicitly; ``from_env`` resolves it from the environment.

    A real SDK isn't imported until ``complete`` is called, so unit tests
    can construct an :class:`LLMClient` without the dependency installed.
    """

    def __init__(
        self,
        *,
        settings: Any = None,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float = 180.0,
        max_retries: int = 2,
    ):
        from .config import LLMSettings

        self.settings = settings or LLMSettings.from_env()
        # Explicit constructor args win over the resolved settings.
        self.api_key = api_key or self.settings.api_key
        self.base_url = base_url if base_url is not None else self.settings.base_url
        self.timeout = timeout
        self.max_retries = max_retries
        self._sdk = None  # lazy import
        # Reasoning models spend the output budget on their think trace and
        # can return finish_reason="length" with EMPTY content. When that
        # happens we retry with a bigger ceiling and remember it, so only the
        # first call of a run pays for the truncated attempt.
        self._budget_floor = 0
        # Some servers reject response_format={"type": "json_object"}. Flipped
        # off on the first such rejection; the prompts + tolerant parsers carry
        # JSON alone.
        self._json_object_ok = True
        self._budget_param = "max_tokens"

    @classmethod
    def from_env(cls, env_var: str = "ANTHROPIC_API_KEY", **kwargs: Any) -> "LLMClient":
        """Resolve the provider from AGENTICMCPE_LLM_* / .env.

        ``env_var`` is kept for backwards compatibility with callers that
        pin a key directly; it is only consulted when the
        resolved provider supplied no key of its own.
        """
        from .config import LLMSettings

        settings = LLMSettings.from_env()
        if not settings.api_key and os.environ.get(env_var):
            settings.api_key = os.environ[env_var]
        return cls(settings=settings, **kwargs)

    @property
    def model(self) -> str:
        return self.settings.model

    def _ensure_sdk(self) -> Any:
        if self._sdk is not None:
            return self._sdk
        kwargs: dict[str, Any] = {"timeout": self.timeout}
        if self.api_key:
            kwargs["api_key"] = self.api_key
        if self.base_url:
            kwargs["base_url"] = self.base_url
        if self.settings.sdk == "anthropic":
            try:
                from anthropic import Anthropic  # type: ignore
            except ImportError as e:
                raise LLMError(
                    "anthropic SDK is not installed; run "
                    "`pip install -r pkg/finance_agenticmcpe/requirements.txt`"
                ) from e
            self._sdk = Anthropic(**kwargs)
        else:
            try:
                from openai import OpenAI  # type: ignore
            except ImportError as e:
                raise LLMError(
                    "openai SDK is not installed; run "
                    "`pip install -r pkg/finance_agenticmcpe/requirements.txt`"
                ) from e
            self._sdk = OpenAI(**kwargs)
        return self._sdk

    def complete(
        self,
        *,
        model: str | None = None,
        system: str,
        messages: Sequence[Message] | Iterable[Message],
        max_tokens: int | None = None,
        temperature: float | None = None,
        stop_sequences: Sequence[str] | None = None,
        tools: Sequence[dict[str, Any]] | None = None,
        json_mode: bool = False,
    ) -> str:
        """Return the assistant text from a single completion call.

        ``model`` / ``max_tokens`` / ``temperature`` default to the resolved
        :class:`LLMSettings`, so callers that only care about the prompt
        (planner, verifier) can omit them.

        For tool-capable runs (when ``tools`` is supplied), the first
        tool use block's JSON-decoded ``input`` is returned as text. This
        keeps the planner/executor/verifier surface uniform — all of them
        ask for a string and parse it.
        """
        sdk = self._ensure_sdk()
        s = self.settings
        model = model or s.model
        max_tokens = s.max_tokens if max_tokens is None else max_tokens
        max_tokens = max(max_tokens, self._budget_floor)
        temperature = s.temperature if temperature is None else temperature
        msg_list = [{"role": m.role, "content": m.content} for m in messages]

        def invoke(budget: int) -> tuple[str, str]:
            """Return (text, finish_reason)."""
            if s.sdk == "anthropic":
                kwargs: dict[str, Any] = {
                    "model": model,
                    "max_tokens": budget,
                    "temperature": temperature,
                    "system": system,
                    "messages": msg_list,
                }
                if stop_sequences:
                    kwargs["stop_sequences"] = list(stop_sequences)
                if tools:
                    kwargs["tools"] = list(tools)
                resp = sdk.messages.create(**kwargs)
                return _extract_text(resp), str(getattr(resp, "stop_reason", "") or "")
            # The system prompt is the first message.
            kwargs = {
                "model": model,
                self._budget_param: budget,
                "temperature": temperature,
                "messages": [{"role": "system", "content": system}, *msg_list],
            }
            if stop_sequences:
                kwargs["stop"] = list(stop_sequences)
            if json_mode and self._json_object_ok:
                kwargs["response_format"] = {"type": "json_object"}
            if getattr(s, "extra_body", None):
                kwargs["extra_body"] = s.extra_body
            choice = sdk.chat.completions.create(**kwargs).choices[0]
            return (choice.message.content or ""), str(choice.finish_reason or "")

        attempt = 0
        last_err: Exception | None = None
        budget = max_tokens
        while attempt <= self.max_retries:
            try:
                text, finish = invoke(budget)
            except Exception as e:  # noqa: BLE001 — SDK raises many subclasses.
                if json_mode and self._json_object_ok and "response_format" in str(e):
                    self._json_object_ok = False  # flips once; cannot loop
                    continue
                if self._budget_param == "max_tokens" and "max_completion_tokens" in str(e):
                    self._budget_param = "max_completion_tokens"  # flips once
                    continue
                last_err = e
                attempt += 1
                if attempt > self.max_retries:
                    break
                continue
            if text.strip():
                return text
            # Empty reply. If the budget ran out (reasoning trace ate it all),
            # escalate once and remember the new floor for this client.
            if finish in ("length", "max_tokens") and budget < _MAX_OUTPUT_BUDGET:
                budget = min(max(budget * 4, 32768), _MAX_OUTPUT_BUDGET)
                self._budget_floor = budget
                continue
            last_err = LLMError(f"model returned empty content (finish_reason={finish!r})")
            attempt += 1
        raise LLMError(f"LLM call failed after {self.max_retries + 1} attempts: {last_err}")

    def stream(
        self,
        *,
        model: str,
        system: str,
        messages: Sequence[Message] | Iterable[Message],
        max_tokens: int = 4096,
        temperature: float = 0.0,
    ) -> Iterable[str]:
        """Yield text deltas from a stream. Used by the
        runnable entry point when ``--stream`` is set."""
        sdk = self._ensure_sdk()
        msg_list = [{"role": m.role, "content": m.content} for m in messages]
        with sdk.messages.stream(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system,
            messages=msg_list,
        ) as stream:
            for text in stream.text_stream:
                yield text


def _extract_text(resp: Any) -> str:
    """Extract a single string from a response, preferring the
    first tool_use block's JSON input if no text block is present."""
    blocks = getattr(resp, "content", []) or []
    texts: list[str] = []
    tool_inputs: list[Any] = []
    for block in blocks:
        btype = getattr(block, "type", None)
        if btype == "text":
            texts.append(getattr(block, "text", "") or "")
        elif btype == "tool_use":
            tool_inputs.append(getattr(block, "input", None))
    if texts:
        joined = "".join(texts).strip()
        if joined:
            return joined
    if tool_inputs:
        return json.dumps(tool_inputs[0], ensure_ascii=False)
    return ""


__all__ = ["LLMClient", "Message", "LLMError"]