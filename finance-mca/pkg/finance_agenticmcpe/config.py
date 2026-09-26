"""Runtime configuration for the finance agentic workflow.

Server-agnostic except where noted. No required credential (the yfinance
server doesn't use one), no token pool, no idempotency tables (the server
is read-only).
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# .env loading + LLM provider registry
#
# Kept byte-compatible with github-mcp-server's `pkg/agenticmcpe/config.py`
# (PORTING.md §4 lists the LLM stack as engine, not adapter) so the same
# AGENTICMCPE_LLM_* env vars drive both ports.
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parents[2]

# Searched in order; the first existing file wins for any given key
# (`setdefault` semantics — an already-exported var is never overridden).
DOTENV_CANDIDATES = (
    _REPO_ROOT / ".env",
    _REPO_ROOT / "pkg" / "finance_agenticmcpe" / ".env",
    _REPO_ROOT / "github-mcp-server-stable" / "pkg" / "agenticmcpe" / ".env",
)


def load_dotenv(path: str | os.PathLike[str] | None = None) -> None:
    """Load KEY=VALUE lines from a .env into ``os.environ`` (never overriding
    an already-set var). Silent no-op when absent. Supports ``#`` comments,
    ``export KEY=...`` and single/double quotes."""
    candidates = [Path(path)] if path else list(DOTENV_CANDIDATES)
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


# provider -> default base_url / model / key env-var names / sdk.
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
    "custom": {
        "base_url": None,
        "model": None,
        "key_env": ["AGENTICMCPE_LLM_API_KEY", "CUSTOM_API_KEY"],
        "sdk": "openai",
    },
}


class ConfigError(RuntimeError):
    """Misconfiguration that must fail loudly rather than silently defaulting."""


def _bool_env(name: str, default: bool) -> bool:
    """Read a boolean switch from the environment.

    Accepts 1/0, true/false, yes/no, on/off (case-insensitive). Anything
    unrecognised falls back to ``default`` rather than failing — a toggle
    should never be the thing that stops a run.
    """
    raw = os.environ.get(name, "").strip().lower()
    if not raw:
        return default
    if raw in {"1", "true", "yes", "on"}:
        return True
    if raw in {"0", "false", "no", "off"}:
        return False
    return default


def _num_env(name: str, default: Any, cast: Any) -> Any:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    # Tolerate digit grouping — "16,384" and "16_384" can only mean 16384,
    # and a hard startup failure over a thousands separator is a poor trade.
    cleaned = raw.replace(",", "").replace("_", "")
    try:
        return cast(cleaned)
    except ValueError as e:
        raise ConfigError(f"{name}={raw!r} is not a valid {cast.__name__}") from e


@dataclass
class LLMSettings:
    """Resolved LLM stack for one run (provider + credentials + knobs)."""

    provider: str
    model: str
    api_key: str
    base_url: str | None
    sdk: str
    temperature: float = 0.0
    max_tokens: int = 4096
    # JSON object forwarded as ``extra_body=`` on every chat call.
    extra_body: dict[str, Any] | None = None

    @classmethod
    def from_env(cls, provider: str | None = None) -> "LLMSettings":
        load_dotenv()
        provider = (
            provider or os.environ.get("AGENTICMCPE_LLM_PROVIDER") or "deepseek"
        ).lower()
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
        model = os.environ.get("AGENTICMCPE_LLM_MODEL") or reg["model"]
        base_url = os.environ.get("AGENTICMCPE_LLM_BASE_URL") or reg["base_url"]
        if provider == "anthropic" and not base_url:
            base_url = os.environ.get("ANTHROPIC_BASE_URL") or None
        if not model:
            raise ConfigError(
                f"no model for provider {provider!r}: set AGENTICMCPE_LLM_MODEL"
            )
        if provider == "custom" and not base_url:
            raise ConfigError(
                "provider 'custom' requires AGENTICMCPE_LLM_BASE_URL"
            )
        extra_body: dict[str, Any] | None = None
        raw_extra = os.environ.get("AGENTICMCPE_LLM_EXTRA_BODY", "").strip()
        if raw_extra:
            try:
                parsed = json.loads(raw_extra)
            except json.JSONDecodeError as e:
                raise ConfigError(
                    f"AGENTICMCPE_LLM_EXTRA_BODY is not valid JSON: {e}"
                ) from e
            if not isinstance(parsed, dict):
                raise ConfigError(
                    "AGENTICMCPE_LLM_EXTRA_BODY must be a JSON object, got "
                    f"{type(parsed).__name__}"
                )
            extra_body = parsed
        return cls(
            provider=provider,
            model=model,
            api_key=api_key,
            base_url=base_url,
            sdk=reg["sdk"],
            temperature=_num_env("AGENTICMCPE_LLM_TEMPERATURE", 0.0, float),
            max_tokens=_num_env("AGENTICMCPE_LLM_MAX_TOKENS", 4096, int),
            extra_body=extra_body,
        )

    def redacted(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "model": self.model,
            "base_url": self.base_url,
            "sdk": self.sdk,
            "api_key_set": bool(self.api_key),
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }


@dataclass
class AgenticConfig:
    """All knobs in one place.

    The agentic workflow has four cooperating LLM roles — planner,
    executor, verifier, RAG rewriter — and a single MCP transport. Each
    role can be backed by a different model (or the same one) and each
    has its own temperature and token budget. Defaults are conservative
    for the yfinance server's catalog.
    """

    # ---- MCP transport ----
    server_command: str = field(default_factory=lambda: sys.executable)
    server_args: list[str] = field(
        default_factory=lambda: ["-m", "servers.yahoo_finance", "--transport", "stdio"]
    )
    server_cwd: str | None = None
    server_env: dict[str, str] = field(default_factory=dict)

    # ---- LLM defaults ----
    # Used when a role does not override its own model.
    model: str = "claude-fable-5"
    max_tokens: int = 4096
    temperature: float = 0.0

    # Per-role overrides. ``None`` means fall back to ``model`` /
    # ``max_tokens`` / ``temperature`` above.
    planner_model: str | None = None
    executor_model: str | None = None
    verifier_model: str | None = None
    rag_model: str | None = None

    planner_temperature: float | None = None
    executor_temperature: float | None = None
    verifier_temperature: float | None = None
    rag_temperature: float | None = None

    planner_max_tokens: int | None = None
    executor_max_tokens: int | None = None
    verifier_max_tokens: int | None = None
    rag_max_tokens: int | None = None

    # ---- Workflow control ----
    max_replans: int = 3
    max_step_retries: int = 3
    verifier_runs: int = 1
    rag_top_k: int = 5
    catalog_path: str = "finance_agenticmcpe_catalog.json"
    # RAG corpus of verified task->sequence mappings, consulted by the
    # planner BEFORE the LLM. Referenced by PATH only: this package never
    # imports finance_taskgen, so the engine runs with or without the
    # flywheel installed. Empty string means "resolve the default".
    # ``AGENTICMCPE_RAG=0`` disables retrieval without touching any driver —
    # this is what makes a RAG-on/RAG-off A/B a matter of one env var.
    rag_enabled: bool = field(
        default_factory=lambda: (load_dotenv(), _bool_env("AGENTICMCPE_RAG", True))[1]
    )
    rag_kb_path: str = ""
    run_log_path: str = "finance_agenticmcpe_runs.jsonl"

    # ---- Tooling ----
    tool_definition_source: str = "servers/yahoo_finance/server.py"
    extract_tools_from_source: bool = True

    # ---- LLM client ----
    # Resolved lazily from the environment / .env by ``llm_settings()``.
    # Set explicitly to pin a provider without touching the environment.
    llm: LLMSettings | None = None
    anthropic_api_key: str | None = None

    # ---- Prompt paths (overridable for tests) ----
    planner_system_path: str = "pkg/finance_agenticmcpe/prompts/planner_system.txt"
    planner_user_path: str = "pkg/finance_agenticmcpe/prompts/planner_user.txt"
    executor_system_path: str = "pkg/finance_agenticmcpe/prompts/executor_system.txt"
    verifier_system_path: str = "pkg/finance_agenticmcpe/prompts/verifier_system.txt"
    rag_query_rewrite_path: str = "pkg/finance_agenticmcpe/prompts/rag_query_rewrite.txt"

    # ---- Environment variable names ----
    env_anthropic_api_key: str = "ANTHROPIC_API_KEY"

    # ---- Runtime derivations ----

    def resolve_kb_path(self) -> Path | None:
        """Path to the RAG corpus, or ``None`` when there is none to read.

        Order: an explicit ``rag_kb_path`` > ``AGENTICMCPE_KB_PATH`` (per-server
        corpora must stay in separate files — never mix them) > the taskgen KB
        at its default location. Never raises; a missing corpus just means the
        planner runs as a pure LLM planner.
        """
        if not self.rag_enabled:
            return None
        load_dotenv()
        candidates: list[Path] = []
        if self.rag_kb_path:
            candidates.append(Path(self.rag_kb_path))
        env_path = os.environ.get("AGENTICMCPE_KB_PATH", "").strip()
        if env_path:
            candidates.append(Path(env_path))
        candidates.append(_REPO_ROOT / "pkg" / "finance_taskgen" / "knowledge_base.json")
        for p in candidates:
            if p.is_file():
                return p
        return None

    def llm_settings(self) -> LLMSettings:
        """Resolve (and memoize) the LLM stack for this config."""
        if self.llm is None:
            self.llm = LLMSettings.from_env()
        return self.llm

    def effective_anthropic_api_key(self) -> str | None:
        """Resolve the API key from the env var if not set explicitly."""
        if self.anthropic_api_key:
            return self.anthropic_api_key
        return os.environ.get(self.env_anthropic_api_key)

    def planner_kwargs(self) -> dict[str, Any]:
        return {
            "model": self.planner_model or self.model,
            "max_tokens": self.planner_max_tokens or self.max_tokens,
            "temperature": (
                self.planner_temperature
                if self.planner_temperature is not None
                else self.temperature
            ),
        }

    def executor_kwargs(self) -> dict[str, Any]:
        return {
            "model": self.executor_model or self.model,
            "max_tokens": self.executor_max_tokens or self.max_tokens,
            "temperature": (
                self.executor_temperature
                if self.executor_temperature is not None
                else self.temperature
            ),
        }

    def verifier_kwargs(self) -> dict[str, Any]:
        return {
            "model": self.verifier_model or self.model,
            "max_tokens": self.verifier_max_tokens or self.max_tokens,
            "temperature": (
                self.verifier_temperature
                if self.verifier_temperature is not None
                else self.temperature
            ),
        }

    def rag_kwargs(self) -> dict[str, Any]:
        return {
            "model": self.rag_model or self.model,
            "max_tokens": self.rag_max_tokens or self.max_tokens,
            "temperature": (
                self.rag_temperature
                if self.rag_temperature is not None
                else self.temperature
            ),
        }

    # ---- Persistence ----

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False, default=str)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "AgenticConfig":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    @classmethod
    def from_json(cls, raw: str) -> "AgenticConfig":
        return cls.from_dict(json.loads(raw))

    @classmethod
    def load(cls, path: str | os.PathLike[str]) -> "AgenticConfig":
        with open(path, encoding="utf-8") as f:
            return cls.from_json(f.read())

    def save(self, path: str | os.PathLike[str]) -> None:
        Path(path).write_text(self.to_json(), encoding="utf-8")


def default_config() -> AgenticConfig:
    """Return a fresh default config. Use this rather than importing a
    module-level singleton — tests want to mutate per-instance."""
    return AgenticConfig()


__all__ = ["AgenticConfig", "default_config"]