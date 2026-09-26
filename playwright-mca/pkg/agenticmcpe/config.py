"""Base utility: environment, configuration, multi-provider LLM client, server config.

This is the single "base util" for the playwright-agenticmcpe workflow. It owns:

* ``.env`` / environment loading (no third-party dotenv dependency).
* A unified :class:`LLMClient` over many providers behind one ``chat()``
  method. See ``PROVIDER_REGISTRY``.
* The playwright-mcp server configuration (spawn command + browser options).
  There is NO credential: playwright-mcp needs no token — its configuration is
  the browser/launch flags. There is also NO build step: the server runs
  straight from this checkout with ``node cli.js`` (the implementation lives in
  the ``playwright-core`` npm dependency; ``npm run build`` is a no-op).
* A :class:`Settings` aggregate the three agents read from.

Nothing here imports the agents, so it is safe to import from anywhere.
"""

from __future__ import annotations

import json
import os
import shlex
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Repo root = the playwright-mcp checkout (two levels up from pkg/agenticmcpe).
REPO_ROOT = Path(__file__).resolve().parents[2]
# Headless Chromium advertises "HeadlessChrome/<ver>" in its user agent — a
# trivial bot signal (booking.com serves an empty page to it). Present a
# standard Chrome UA by default in headless runs; override or disable with
# AGENTICMCPE_PW_USER_AGENT (set it empty to disable).
DEFAULT_HEADLESS_UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                       "AppleWebKit/537.36 (KHTML, like Gecko) "
                       "Chrome/140.0.0.0 Safari/537.36")
DEFAULT_RUNS_DIR = REPO_ROOT / "pkg" / "agenticmcpe" / "runs"
# Default spawn: run the server from THIS source checkout. Override with
# AGENTICMCPE_PW_CMD (a shell-split string, e.g. "npx -y @playwright/mcp@0.0.78").
DEFAULT_SERVER_COMMAND = ["node", str(REPO_ROOT / "cli.js")]


class ConfigError(RuntimeError):
    """Raised when required configuration is missing."""


# ---------------------------------------------------------------------------
# Server readiness: playwright-mcp needs npm dependencies, not a compile step
# ---------------------------------------------------------------------------

def ensure_server(command: list[str], repo_root: Path) -> None:
    """Guarantee the playwright-mcp server can be spawned.

    Unlike the Go github-mcp-server there is nothing to build: ``cli.js``
    delegates into the ``playwright-core`` npm package. The only failure mode
    worth catching early is a checkout without ``node_modules`` (the require
    fails with MODULE_NOT_FOUND at spawn). Only the default in-checkout command
    is checked; a user-supplied command is theirs to manage.
    """
    if command[:2] != DEFAULT_SERVER_COMMAND[:2]:
        return
    if not (repo_root / "cli.js").is_file():
        raise ConfigError(f"cli.js not found in {repo_root} — is this a "
                          f"playwright-mcp checkout?")
    if not (repo_root / "node_modules" / "playwright-core").is_dir():
        raise ConfigError(
            "playwright-mcp dependencies are not installed: run `npm ci` in "
            f"{repo_root} (one-time; provisions playwright/playwright-core). "
            "If browsers are missing too, follow with `npx playwright install "
            "chromium`."
        )


# ---------------------------------------------------------------------------
# .env loading
# ---------------------------------------------------------------------------

def load_dotenv(path: str | os.PathLike[str] | None = None) -> None:
    """Load KEY=VALUE lines from a .env file into ``os.environ`` (no override
    of already-set vars). Silent no-op when the file is absent. Minimal parser:
    supports ``#`` comments, ``export KEY=...`` and single/double quotes."""
    candidates = (
        [Path(path)]
        if path
        else [Path.cwd() / ".env", Path(__file__).resolve().parent / ".env"]
    )
    for p in candidates:
        if not p.is_file():
            continue
        for raw in p.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("export "):
                line = line[len("export ") :]
            if "=" not in line:
                continue
            key, _, val = line.partition("=")
            key, val = key.strip(), val.strip()
            if (val.startswith('"') and val.endswith('"')) or (
                val.startswith("'") and val.endswith("'")
            ):
                val = val[1:-1]
            os.environ.setdefault(key, val)


# ---------------------------------------------------------------------------
# LLM provider registry + client
# ---------------------------------------------------------------------------

# provider -> (default base_url, default model, key env-var names, sdk).
# base_url/model are sensible defaults; override per run with
# AGENTICMCPE_LLM_BASE_URL / AGENTICMCPE_LLM_MODEL.
PROVIDER_REGISTRY: dict[str, dict[str, Any]] = {
    "deepseek": {
        "base_url": "https://api.deepseek.com",
        "model": "deepseek-chat",
        "key_env": ["DEEPSEEK_API_KEY"],
        "sdk": "openai",
    },
    "openai": {
        "base_url": None,
        "model": "gpt-4o",
        "key_env": ["OPENAI_API_KEY"],
        "sdk": "openai",
    },
    "gemini": {
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "model": "gemini-2.0-flash",
        "key_env": ["GEMINI_API_KEY", "GOOGLE_API_KEY"],
        "sdk": "openai",
    },
    "anthropic": {
        "base_url": None,
        "model": "claude-sonnet-4-6",
        "key_env": ["ANTHROPIC_API_KEY"],
        "sdk": "anthropic",
    },
    "minimax": {
        "base_url": "https://api.minimaxi.com/v1",
        "model": "MiniMax-M3",
        "key_env": ["MINIMAX_API_KEY"],
        "sdk": "openai",
    },
    "qwen": {
        "base_url": "https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
        "model": "qwen-plus",
        "key_env": ["DASHSCOPE_API_KEY", "QWEN_API_KEY"],
        "sdk": "openai",
    },
    "glm": {
        "base_url": "https://open.bigmodel.cn/api/paas/v4",
        "model": "glm-4.6",
        "key_env": ["ZHIPUAI_API_KEY", "GLM_API_KEY"],
        "sdk": "openai",
    },
    "kimi": {
        "base_url": "https://api.moonshot.ai/v1",
        "model": "kimi-k2-0711-preview",
        "key_env": ["MOONSHOT_API_KEY", "KIMI_API_KEY"],
        "sdk": "openai",
    },
    "grok": {
        "base_url": "https://api.x.ai/v1",
        "model": "grok-4",
        "key_env": ["XAI_API_KEY", "GROK_API_KEY"],
        "sdk": "openai",
    },
    "xiaomi": {
        "base_url": "http://localhost:8000/v1",
        "model": "MiMo-7B-RL",
        "key_env": ["XIAOMI_API_KEY", "MIMO_API_KEY"],
        "sdk": "openai",
    },
    # Self-define template: set AGENTICMCPE_LLM_BASE_URL + AGENTICMCPE_LLM_MODEL
    # and put the key in AGENTICMCPE_LLM_API_KEY (or CUSTOM_API_KEY).
    "custom": {
        "base_url": None,
        "model": None,
        "key_env": ["AGENTICMCPE_LLM_API_KEY", "CUSTOM_API_KEY"],
        "sdk": "openai",
    },
}


# Transient upstream failures (gateway/proxy overload), as opposed to a bad
# request. Retried with a short backoff inside LLMClient._create.
_TRANSIENT_ATTEMPTS = 3
_TRANSIENT_BACKOFF_S = 5.0
_TRANSIENT_MARKERS = ("Error code: 500", "Error code: 502", "Error code: 503",
                      "Error code: 504", "Error code: 520", "Error code: 524",
                      "APIConnectionError", "APITimeoutError")


def _is_transient(exc: Exception) -> bool:
    text = f"{type(exc).__name__}: {exc}"
    return any(m in text for m in _TRANSIENT_MARKERS)


def _with_transient_retry(call: Any) -> Any:
    """Run ``call()``, retrying transient upstream failures with backoff."""
    for attempt in range(_TRANSIENT_ATTEMPTS):
        try:
            return call()
        except Exception as e:
            if attempt == _TRANSIENT_ATTEMPTS - 1 or not _is_transient(e):
                raise
            time.sleep(_TRANSIENT_BACKOFF_S * (attempt + 1))


def _num_env(name: str, default: "int | float", cast: Any) -> "int | float":
    """Parse a numeric env var tolerantly: strip commas/underscores/whitespace
    (e.g. ``262,144`` or ``262_144``). Fall back to ``default`` if unparseable."""
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    cleaned = raw.strip().replace(",", "").replace("_", "")
    try:
        return cast(cleaned)
    except (TypeError, ValueError):
        return default


def _bool_env(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    return raw.strip().lower() not in ("0", "false", "no", "off")


@dataclass
class LLMSettings:
    provider: str
    model: str
    api_key: str
    base_url: str | None
    sdk: str
    temperature: float = 0.0
    max_tokens: int = 4096
    # A JSON object forwarded as `extra_body=` on every chat call.
    extra_body: dict[str, Any] | None = None

    @classmethod
    def from_env(cls, provider: str | None = None) -> "LLMSettings":
        provider = (provider or os.environ.get("AGENTICMCPE_LLM_PROVIDER") or "deepseek").lower()
        if provider not in PROVIDER_REGISTRY:
            raise ConfigError(
                f"unknown LLM provider {provider!r}; choose from {sorted(PROVIDER_REGISTRY)}"
            )
        reg = PROVIDER_REGISTRY[provider]
        api_key = ""
        for env_name in reg["key_env"]:
            if os.environ.get(env_name):
                api_key = os.environ[env_name]
                break
        # Generic override hooks so any provider can be reconfigured via env.
        model = os.environ.get("AGENTICMCPE_LLM_MODEL") or reg["model"]
        base_url = os.environ.get("AGENTICMCPE_LLM_BASE_URL") or reg["base_url"]
        if provider == "anthropic" and not base_url:
            base_url = os.environ.get("ANTHROPIC_BASE_URL") or None
        # The 'custom' template ships no defaults — require base_url + model so a
        # misconfig fails loudly.
        if not model:
            raise ConfigError(
                f"no model for provider {provider!r}: set AGENTICMCPE_LLM_MODEL"
            )
        if provider == "custom" and not base_url:
            raise ConfigError(
                "provider 'custom' requires AGENTICMCPE_LLM_BASE_URL "
                "(your OpenAI-compatible endpoint)"
            )
        temperature = _num_env("AGENTICMCPE_LLM_TEMPERATURE", 0.0, float)
        max_tokens = _num_env("AGENTICMCPE_LLM_MAX_TOKENS", 4096, int)
        # AGENTICMCPE_LLM_EXTRA_BODY: JSON object forwarded as `extra_body=` on
        # every chat call.
        extra_body: dict[str, Any] | None = None
        _raw_extra = os.environ.get("AGENTICMCPE_LLM_EXTRA_BODY", "").strip()
        if _raw_extra:
            try:
                _parsed = json.loads(_raw_extra)
            except json.JSONDecodeError as e:
                raise ConfigError(
                    f"AGENTICMCPE_LLM_EXTRA_BODY is not valid JSON: {e}"
                ) from e
            if not isinstance(_parsed, dict):
                raise ConfigError(
                    "AGENTICMCPE_LLM_EXTRA_BODY must be a JSON object, got "
                    f"{type(_parsed).__name__}"
                )
            extra_body = _parsed
        return cls(
            provider=provider,
            model=model,
            api_key=api_key,
            base_url=base_url,
            sdk=reg["sdk"],
            temperature=temperature,
            max_tokens=max_tokens,
            extra_body=extra_body,
        )


class LLMClient:
    """One ``chat()`` across providers.

    ``chat(system, user, json_mode=True)`` returns the assistant's text. With
    ``json_mode`` we request ``response_format`` json; for any provider that
    rejects the flag we fall back to prompt instructions plus a robust
    extractor in :func:`extract_json`.
    """

    def __init__(self, settings: LLMSettings) -> None:
        self.settings = settings
        self._client: Any = None
        # Cumulative API usage across every chat() on this client instance.
        # Both requests of a json_mode fallback retry are counted (both are
        # billed). Providers that omit usage contribute only to "calls".
        self.usage: dict[str, int] = {"calls": 0, "prompt_tokens": 0,
                                      "completion_tokens": 0, "total_tokens": 0}
        self._token_param = "max_tokens"
        # Cleared on the first 400 that names `temperature` (models without
        # sampling params), then never re-sent.
        self._send_temperature = True
        # Streaming is OPT-IN (AGENTICMCPE_LLM_STREAM=1). It exists because a
        # gateway read-timeout (Cloudflare 524 at 120s) drops any single long
        # generation; streaming keeps bytes flowing and changes transport only.
        # It is off by default because not every proxy streams correctly.
        self._stream = _bool_env("AGENTICMCPE_LLM_STREAM", False)

    @staticmethod
    def _collect_stream(client: Any, call: dict[str, Any]) -> Any:
        """Consume a streamed completion into a response-shaped shim carrying
        the joined text and (when the provider reports it) usage."""
        from types import SimpleNamespace

        parts: list[str] = []
        reasoning: list[str] = []
        usage = None
        stream = client.chat.completions.create(
            **call, stream=True, stream_options={"include_usage": True})
        for chunk in stream:
            if getattr(chunk, "usage", None):
                usage = chunk.usage
            for choice in (getattr(chunk, "choices", None) or []):
                delta = getattr(choice, "delta", None)
                if delta is None:
                    continue
                if getattr(delta, "content", None):
                    parts.append(delta.content)
                if getattr(delta, "reasoning_content", None):
                    reasoning.append(delta.reasoning_content)
        msg = SimpleNamespace(content="".join(parts),
                              reasoning_content="".join(reasoning) or None)
        return SimpleNamespace(choices=[SimpleNamespace(message=msg)], usage=usage)

    def _create(self, client: Any, **kwargs: Any) -> Any:
        """chat.completions.create with the output-budget parameter under
        whichever name this model accepts, retrying transient gateway errors.

        A proxied endpoint under load returns Cloudflare 520/524 ("origin took
        too long") on the long planner/verifier prompts; these are explicitly
        retryable and would otherwise crash a task and leave it with no
        verify.py, silently shrinking a cross-verified comparison."""
        budget = kwargs.pop("_max_tokens")
        # The stream collector joins text only; tool-call deltas would be lost.
        allow_stream = "tools" not in kwargs

        def _call(**extra: Any) -> Any:
            call = {**kwargs, **extra, self._token_param: budget}
            if not (self._stream and allow_stream):
                return client.chat.completions.create(**call)
            return self._collect_stream(client, call)

        def _once() -> Any:
            try:
                return _call()
            except Exception as e:
                other = ("max_completion_tokens" if self._token_param == "max_tokens"
                         else "max_tokens")
                if other in str(e):
                    self._token_param = other
                    return _call()
                # A provider that cannot stream falls back to a single buffered
                # call, permanently. Covers both an explicit rejection and a
                # broken stream (truncated chunked body -> RemoteProtocolError).
                if self._stream and ("stream" in str(e).lower()
                                     or "chunked" in str(e).lower()
                                     or type(e).__name__ == "RemoteProtocolError"):
                    self._stream = False
                    return _call()
                raise

        return _with_transient_retry(_once)

    def _record_usage(self, resp: Any) -> None:
        self.usage["calls"] += 1
        u = getattr(resp, "usage", None)
        if u is None:
            return
        pt = getattr(u, "prompt_tokens", None) or getattr(u, "input_tokens", None) or 0
        ct = getattr(u, "completion_tokens", None) or getattr(u, "output_tokens", None) or 0
        tt = getattr(u, "total_tokens", None) or (pt + ct)
        self.usage["prompt_tokens"] += int(pt)
        self.usage["completion_tokens"] += int(ct)
        self.usage["total_tokens"] += int(tt)

    def _ensure_client(self) -> Any:
        if self._client is not None:
            return self._client
        s = self.settings
        if not s.api_key:
            raise ConfigError(
                f"no API key for provider {s.provider!r}: set one of "
                f"{PROVIDER_REGISTRY[s.provider]['key_env']} (env or .env)"
            )
        # Bound every request so a stalled provider can't hang the run
        # indefinitely (a reasoning model legitimately takes a while, so the
        # default is generous; override with AGENTICMCPE_LLM_TIMEOUT). A single
        # retry avoids a long storm on a genuinely dead endpoint.
        timeout = _num_env("AGENTICMCPE_LLM_TIMEOUT", 240.0, float)
        if s.sdk == "anthropic":
            import anthropic

            kwargs: dict[str, Any] = {"api_key": s.api_key, "timeout": timeout,
                                      "max_retries": 1}
            if s.base_url:
                kwargs["base_url"] = s.base_url
            self._client = anthropic.Anthropic(**kwargs)
        else:
            from openai import OpenAI

            kwargs = {"api_key": s.api_key, "timeout": timeout, "max_retries": 1}
            if s.base_url:
                kwargs["base_url"] = s.base_url
            self._client = OpenAI(**kwargs)
        return self._client

    def chat(self, system: str, user: str, *, json_mode: bool = False,
             stop: list[str] | None = None) -> str:
        client = self._ensure_client()
        s = self.settings
        if s.sdk == "anthropic":
            kw: dict[str, Any] = {
                "model": s.model,
                "max_tokens": s.max_tokens,
                "system": system,
                "messages": [{"role": "user", "content": user}],
            }
            if stop:
                kw["stop_sequences"] = stop
            try:
                resp = client.messages.create(
                    **kw, **({"temperature": s.temperature} if self._send_temperature else {}))
            except Exception as e:
                if not (self._send_temperature and "temperature" in str(e)):
                    raise
                self._send_temperature = False
                resp = client.messages.create(**kw)
            self._record_usage(resp)
            return "".join(
                block.text for block in resp.content if getattr(block, "type", "") == "text"
            )
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        call_kwargs: dict[str, Any] = {
            "model": s.model,
            "messages": messages,
            "temperature": s.temperature,
            "_max_tokens": s.max_tokens,
        }
        if s.extra_body:
            call_kwargs["extra_body"] = s.extra_body
        if stop:
            call_kwargs["stop"] = stop

        def _content(resp: Any) -> str:
            msg = resp.choices[0].message
            return msg.content or getattr(msg, "reasoning_content", None) or ""

        if json_mode:
            try:
                resp = self._create(
                    client, response_format={"type": "json_object"}, **call_kwargs
                )
                self._record_usage(resp)
                text = _content(resp)
                if text:
                    return text
                # Empty reply under response_format — fall through to a plain
                # retry rather than handing back "".
            except Exception:
                # Provider rejected response_format — retry plain and rely on
                # extract_json downstream.
                pass
        resp = self._create(client, **call_kwargs)
        self._record_usage(resp)
        return _content(resp)

    def chat_tools(self, messages: list[dict[str, Any]],
                   tools: list[dict[str, Any]], **opts: Any) -> Any:
        """One native function-calling turn: the message
        history in, the assistant message (``.content``, ``.tool_calls``) out.
        Same settings, retries and usage accounting as :meth:`chat`; ``opts``
        pass through."""
        s = self.settings
        if s.sdk != "openai":
            raise ConfigError(f"chat_tools() needs an OpenAI-compatible provider, "
                              f"not {s.provider!r}")
        call_kwargs: dict[str, Any] = {
            "model": s.model, "messages": messages, "tools": tools,
            "temperature": s.temperature, "_max_tokens": s.max_tokens, **opts,
        }
        if s.extra_body:
            call_kwargs["extra_body"] = s.extra_body
        resp = self._create(self._ensure_client(), **call_kwargs)
        self._record_usage(resp)
        return resp.choices[0].message

    def respond(self, items: list[Any], tools: list[dict[str, Any]],
                **opts: Any) -> Any:
        """One native function-calling turn: input items in, the Response out
        (``.output`` items, ``.output_text``). Same retries and usage
        accounting as :meth:`chat`."""
        s = self.settings
        if s.sdk != "openai":
            raise ConfigError(f"respond() needs an OpenAI-compatible provider, "
                              f"not {s.provider!r}")
        extra = dict(s.extra_body or {})
        kw: dict[str, Any] = {"model": s.model, "input": items, "tools": tools,
                              "max_output_tokens": s.max_tokens, **opts}
        effort = extra.pop("reasoning_effort", None)
        if effort:
            kw["reasoning"] = {"effort": effort}
        if extra:
            kw["extra_body"] = extra
        client = self._ensure_client()

        def _once() -> Any:
            if self._send_temperature:
                try:
                    return client.responses.create(**kw, temperature=s.temperature)
                except Exception as e:
                    if "temperature" not in str(e):
                        raise
                    self._send_temperature = False
            return client.responses.create(**kw)

        resp = _with_transient_retry(_once)
        self._record_usage(resp)
        return resp


def extract_json(text: str) -> Any:
    """Best-effort: parse JSON from an LLM reply that may be fenced or chatty."""
    text = text.strip()
    if not text:
        raise ValueError(
            "model reply was EMPTY — the provider returned no content (a "
            "reasoning model can exhaust max_tokens before emitting the "
            "answer). Raise AGENTICMCPE_LLM_MAX_TOKENS or switch "
            "AGENTICMCPE_LLM_MODEL."
        )
    if "</think>" in text:
        text = text.rpartition("</think>")[2].strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # strip ```json fences
    if "```" in text:
        inner = text.split("```", 2)
        if len(inner) >= 2:
            candidate = inner[1]
            if candidate.lstrip().lower().startswith("json"):
                candidate = candidate.lstrip()[4:]
            try:
                return json.loads(candidate.strip())
            except json.JSONDecodeError:
                pass
    # locate the outermost {...} or [...]
    for open_c, close_c in (("{", "}"), ("[", "]")):
        start = text.find(open_c)
        end = text.rfind(close_c)
        if start != -1 and end > start:
            try:
                return json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                continue
    raise ValueError(f"could not extract JSON from model reply:\n{text[:500]}")


# ---------------------------------------------------------------------------
# Aggregate settings
# ---------------------------------------------------------------------------

@dataclass
class Settings:
    llm: LLMSettings
    repo_root: Path = REPO_ROOT
    server_command: list[str] = field(default_factory=lambda: list(DEFAULT_SERVER_COMMAND))
    headless: bool = True
    isolated: bool = True
    browser: str | None = None
    user_agent: str | None = None
    # Server-side action timeout (ms). The upstream default of 5000ms is tuned
    # for interactive use; heavy public SPAs (huggingface.co, booking.com)
    # routinely need longer before an element passes actionability checks.
    timeout_action: int = 15000
    runs_dir: Path = DEFAULT_RUNS_DIR
    run_id: str = ""
    work_dir: Path = field(default_factory=lambda: DEFAULT_RUNS_DIR)

    @classmethod
    def load(cls, *, provider: str | None = None, run_id: str | None = None) -> "Settings":
        load_dotenv()
        llm = LLMSettings.from_env(provider)
        rid = run_id or time.strftime("run-%Y%m%d-%H%M%S")
        runs_dir = Path(os.environ.get("AGENTICMCPE_RUNS_DIR", str(DEFAULT_RUNS_DIR)))
        work_dir = runs_dir / rid
        raw_cmd = os.environ.get("AGENTICMCPE_PW_CMD", "").strip()
        command = shlex.split(raw_cmd) if raw_cmd else list(DEFAULT_SERVER_COMMAND)
        ensure_server(command, REPO_ROOT)
        headless = _bool_env("AGENTICMCPE_PW_HEADLESS", True)
        # Env set (even to empty) wins verbatim; unset -> the standard-Chrome
        # default, but only for headless sessions (headed already looks real).
        ua_env = os.environ.get("AGENTICMCPE_PW_USER_AGENT")
        if ua_env is not None:
            user_agent = ua_env.strip() or None
        else:
            user_agent = DEFAULT_HEADLESS_UA if headless else None
        return cls(
            llm=llm,
            server_command=command,
            headless=headless,
            isolated=_bool_env("AGENTICMCPE_PW_ISOLATED", True),
            browser=os.environ.get("AGENTICMCPE_PW_BROWSER") or None,
            user_agent=user_agent,
            timeout_action=int(_num_env("AGENTICMCPE_PW_TIMEOUT_ACTION", 15000, int)),
            runs_dir=runs_dir,
            run_id=rid,
            work_dir=work_dir,
        )

    def ensure_work_dir(self) -> Path:
        self.work_dir.mkdir(parents=True, exist_ok=True)
        return self.work_dir

    def summary(self) -> dict[str, Any]:
        return {
            "provider": self.llm.provider,
            "model": self.llm.model,
            "base_url": self.llm.base_url,
            "has_api_key": bool(self.llm.api_key),
            "server_command": self.server_command,
            "headless": self.headless,
            "isolated": self.isolated,
            "browser": self.browser or "(bundled chromium)",
            "user_agent": self.user_agent or "(browser default)",
            "timeout_action_ms": self.timeout_action,
            "run_id": self.run_id,
            "work_dir": str(self.work_dir),
        }


__all__ = [
    "REPO_ROOT",
    "DEFAULT_SERVER_COMMAND",
    "ensure_server",
    "ConfigError",
    "load_dotenv",
    "PROVIDER_REGISTRY",
    "LLMSettings",
    "LLMClient",
    "extract_json",
    "Settings",
]
