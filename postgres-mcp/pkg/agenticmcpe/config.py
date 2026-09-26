"""Base utility: environment, configuration, multi-provider LLM client, database config.

This is the single "base util" for the agenticmcpe workflow. It owns:

* ``.env`` / environment loading (no third-party dotenv dependency).
* A unified :class:`LLMClient` over many providers behind one ``chat()``
  method. See ``PROVIDER_REGISTRY``.
* A :class:`DatabaseConfig` holding the PostgreSQL connection URI (the
  credential seam — postgres-mcp's analog of the GitHub token; there is no
  rotation because there are no per-credential rate limits).
* A :class:`Settings` aggregate the three agents read from.

Nothing here imports the agents, so it is safe to import from anywhere.
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Repo root = two levels up from this file (pkg/agenticmcpe/config.py).
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUNS_DIR = REPO_ROOT / "pkg" / "agenticmcpe" / "runs"


# ---------------------------------------------------------------------------
# Server launch: postgres-mcp is a Python package (no build step). The exact
# argv is resolved by pkg.mcp_wrapper.client.resolve_server_cmd (env override,
# PATH, .venv, uv, docker); this replaces the Go ensure_binary/go-build seam.
# ---------------------------------------------------------------------------

def ensure_server() -> list[str]:
    """Resolve (and thereby validate) the postgres-mcp launch command. Raises
    :class:`ConfigError` with a remediation hint when no launch style exists."""
    from pkg.mcp_wrapper.client import resolve_server_cmd
    from pkg.mcp_wrapper.types import MCPError

    try:
        return resolve_server_cmd()
    except MCPError as e:
        raise ConfigError(str(e)) from e


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
            # An inline trailing comment ("KEY=value   # note") is only a
            # comment outside quotes — stripping it here (rather than
            # requiring every value to be pre-cleaned) matches how the
            # template's own commented-out lines document each var. A quoted
            # value's own '#' is never touched.
            if not (val.startswith('"') or val.startswith("'")):
                val = re.split(r"\s+#", val, maxsplit=1)[0].strip()
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


class ConfigError(RuntimeError):
    """Raised when required configuration (key/URI) is missing."""


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


# ---------------------------------------------------------------------------
# Per-call LLM usage accounting
# ---------------------------------------------------------------------------
# Every successful chat() appends one entry here with the PROVIDER-reported
# token counts. Batch drivers drain it around a phase to attribute cost
# (agent vs verifier, workflow vs ReAct). Runs are sequential, so a plain
# module-level list is race-free. A call that RAISES reports no usage (the
# SDK surfaces no response object), so an errored request undercounts by at
# most its own tokens.
USAGE_LOG: list[dict[str, Any]] = []


def drain_usage() -> dict[str, Any]:
    """Return (and clear) accumulated usage: sums plus the per-call log."""
    calls = list(USAGE_LOG)
    USAGE_LOG.clear()
    return {
        "calls": len(calls),
        "prompt_tokens": sum(c["prompt_tokens"] for c in calls),
        "completion_tokens": sum(c["completion_tokens"] for c in calls),
        "total_tokens": sum(c["prompt_tokens"] + c["completion_tokens"]
                            for c in calls),
        "llm_seconds": round(sum(c["seconds"] for c in calls), 1),
        "log": calls,
    }


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
        self._max_tokens_key = "max_tokens"

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

    def chat(self, system: str, user: str, *, json_mode: bool = False) -> str:
        client = self._ensure_client()
        s = self.settings
        t0 = time.time()
        if s.sdk == "anthropic":
            resp = client.messages.create(
                model=s.model,
                max_tokens=s.max_tokens,
                temperature=s.temperature,
                system=system,
                messages=[{"role": "user", "content": user}],
            )
            self._log_usage(resp, t0)
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
            self._max_tokens_key: s.max_tokens,
        }
        if s.extra_body:
            call_kwargs["extra_body"] = s.extra_body
        if json_mode:
            try:
                resp = self._create(
                    client, call_kwargs, response_format={"type": "json_object"}
                )
                self._log_usage(resp, t0)
                return resp.choices[0].message.content or ""
            except Exception:
                # Provider rejected response_format — retry plain and rely on
                # extract_json downstream.
                pass
        resp = self._create(client, call_kwargs)
        self._log_usage(resp, t0)
        return resp.choices[0].message.content or ""

    def _create(self, client: Any, call_kwargs: dict[str, Any], **extra: Any) -> Any:
        """One completion call, adapting the token-budget parameter name."""
        try:
            return client.chat.completions.create(**call_kwargs, **extra)
        except Exception as exc:
            other = ("max_completion_tokens" if self._max_tokens_key == "max_tokens"
                     else "max_tokens")
            if self._max_tokens_key not in str(exc) or other not in str(exc):
                raise
            call_kwargs[other] = call_kwargs.pop(self._max_tokens_key)
            self._max_tokens_key = other
            return client.chat.completions.create(**call_kwargs, **extra)

    def _log_usage(self, resp: Any, t0: float) -> None:
        """Append provider-reported usage; accounting must never break a call."""
        try:
            u = getattr(resp, "usage", None)
            pt = getattr(u, "prompt_tokens", None)
            if pt is None:
                pt = getattr(u, "input_tokens", None)
            ct = getattr(u, "completion_tokens", None)
            if ct is None:
                ct = getattr(u, "output_tokens", None)
            USAGE_LOG.append({
                "provider": self.settings.provider,
                "model": self.settings.model,
                "prompt_tokens": int(pt or 0),
                "completion_tokens": int(ct or 0),
                "seconds": round(time.time() - t0, 2),
            })
        except Exception:
            pass


def extract_json(text: str) -> Any:
    """Best-effort: parse JSON from an LLM reply that may be fenced or chatty."""
    text = text.strip()
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
# Database configuration (the credential seam)
# ---------------------------------------------------------------------------

@dataclass
class DatabaseConfig:
    """The PostgreSQL connection URI postgres-mcp connects with.

    Comes from ``AGENTICMCPE_DATABASE_URI`` or ``DATABASE_URI`` (env or .env).
    There is no pool/rotation — SQL sessions have no per-credential rate
    limits, so the GitHub token-pool seam collapses to a single value. The
    server itself starts (and serves tools/list) without a reachable
    database, so a catalog load works with no URI at all.
    """

    uri: str = ""

    @classmethod
    def from_env(cls) -> "DatabaseConfig":
        uri = (os.environ.get("AGENTICMCPE_DATABASE_URI")
               or os.environ.get("DATABASE_URI") or "")
        return cls(uri=uri.strip())

    @property
    def available(self) -> bool:
        return bool(self.uri)

    def current(self) -> str:
        if not self.uri:
            raise ConfigError(
                "no database URI: set DATABASE_URI (or AGENTICMCPE_DATABASE_URI) "
                "in env/.env, e.g. postgresql://user:pass@localhost:5432/dbname"
            )
        return self.uri

    def obfuscated(self) -> str:
        """The URI with any password replaced, for logs/summaries."""
        import re
        return re.sub(r"://([^:/@]+):[^@]*@", r"://\1:***@", self.uri) if self.uri else ""


# ---------------------------------------------------------------------------
# Aggregate settings
# ---------------------------------------------------------------------------

@dataclass
class Settings:
    llm: LLMSettings
    database: DatabaseConfig
    repo_root: Path = REPO_ROOT
    server_cmd: list[str] = field(default_factory=list)
    runs_dir: Path = DEFAULT_RUNS_DIR
    run_id: str = ""
    work_dir: Path = field(default_factory=lambda: DEFAULT_RUNS_DIR)

    @classmethod
    def load(cls, *, provider: str | None = None, run_id: str | None = None) -> "Settings":
        load_dotenv()
        llm = LLMSettings.from_env(provider)
        database = DatabaseConfig.from_env()
        rid = run_id or time.strftime("run-%Y%m%d-%H%M%S")
        runs_dir = Path(os.environ.get("AGENTICMCPE_RUNS_DIR", str(DEFAULT_RUNS_DIR)))
        work_dir = runs_dir / rid
        server_cmd = ensure_server()
        return cls(
            llm=llm,
            database=database,
            server_cmd=server_cmd,
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
            "database_uri": self.database.obfuscated() or None,
            "server_cmd": self.server_cmd,
            "run_id": self.run_id,
            "work_dir": str(self.work_dir),
        }


__all__ = [
    "REPO_ROOT",
    "ensure_server",
    "ConfigError",
    "load_dotenv",
    "PROVIDER_REGISTRY",
    "LLMSettings",
    "LLMClient",
    "USAGE_LOG",
    "drain_usage",
    "extract_json",
    "DatabaseConfig",
    "Settings",
]
