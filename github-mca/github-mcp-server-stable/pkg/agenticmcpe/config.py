"""Base utility: environment, configuration, multi-provider LLM client, token pool.

This is the single "base util" for the agenticmcpe workflow. It owns:

* ``.env`` / environment loading (no third-party dotenv dependency).
* A unified :class:`LLMClient` over many providers behind one ``chat()``
  method. See ``PROVIDER_REGISTRY``.
* A :class:`GitHubTokenPool` that rotates tokens on rate-limit/auth failure.
* A :class:`Settings` aggregate the three agents read from.

Nothing here imports the agents, so it is safe to import from anywhere.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Repo root = three levels up from this file (pkg/agenticmcpe/config.py).
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BINARY = REPO_ROOT / "pkg" / "mcp_wrapper" / "bin" / "github-mcp-server"
DEFAULT_RUNS_DIR = REPO_ROOT / "pkg" / "agenticmcpe" / "runs"


# ---------------------------------------------------------------------------
# Binary freshness: keep the bundled github-mcp-server in sync with its source
# ---------------------------------------------------------------------------

# `go build` target that produces the bundled binary.
_GO_CMD_PKG = "./cmd/github-mcp-server"
# Directories that never contribute Go source compiled into the binary.
_BUILD_SKIP_DIRS = {".git", "vendor", "node_modules", "__pycache__"}


def _newest_go_source_mtime(repo_root: Path) -> float:
    """Newest mtime among the Go sources that compile into the server binary:
    every non-test ``*.go`` in the module plus ``go.mod``/``go.sum``. Test files
    are skipped — they never end up in the built binary."""
    newest = 0.0
    for name in ("go.mod", "go.sum"):
        p = repo_root / name
        if p.is_file():
            newest = max(newest, p.stat().st_mtime)
    for dirpath, dirnames, filenames in os.walk(repo_root):
        dirnames[:] = [d for d in dirnames if d not in _BUILD_SKIP_DIRS]
        for fn in filenames:
            if fn.endswith(".go") and not fn.endswith("_test.go"):
                try:
                    newest = max(newest, (Path(dirpath) / fn).stat().st_mtime)
                except OSError:
                    pass
    return newest


def ensure_binary(binary_path: Path, repo_root: Path) -> None:
    """Guarantee ``binary_path`` is compiled from the current Go source.

    Rebuilds the bundled github-mcp-server with ``go build`` whenever any Go
    source file is newer than the binary (or the binary is missing), so every
    agenticmcpe run uses the latest server. The mtime gate makes the common
    case (nothing changed) a fast no-op that never invokes the toolchain.

    No-ops when:
      * ``AGENTICMCPE_SKIP_BUILD`` is set — use the binary as-is;
      * a custom ``AGENTICMCPE_BINARY`` is in use — the caller owns that path;
      * the binary is already at least as new as every source file.

    Raises :class:`ConfigError` if a rebuild is required but fails, or is
    required while no binary exists and the Go toolchain is unavailable. If a
    rebuild is merely *stale-triggered* but ``go`` is missing, it warns and
    falls back to the existing binary rather than blocking the run.
    """
    if os.environ.get("AGENTICMCPE_SKIP_BUILD"):
        return
    # Only manage the bundled default binary; a user-supplied path is theirs.
    if binary_path.resolve() != DEFAULT_BINARY.resolve():
        return

    src_mtime = _newest_go_source_mtime(repo_root)
    if binary_path.is_file() and binary_path.stat().st_mtime >= src_mtime:
        return  # binary is at least as new as every source file

    go = shutil.which("go")
    if go is None:
        if binary_path.is_file():
            print(
                f"[agenticmcpe] WARNING: Go source is newer than "
                f"{binary_path.name} but 'go' is not on PATH; running a possibly "
                f"stale binary. Install Go or set AGENTICMCPE_SKIP_BUILD=1 to "
                f"silence this warning.",
                file=sys.stderr,
            )
            return
        raise ConfigError(
            f"github-mcp-server binary missing at {binary_path} and 'go' is not "
            f"on PATH. Install Go, or build it: "
            f"go build -o {binary_path} {_GO_CMD_PKG}"
        )

    binary_path.parent.mkdir(parents=True, exist_ok=True)
    print("[agenticmcpe] github-mcp-server source changed; rebuilding binary ...",
          file=sys.stderr)
    proc = subprocess.run(
        [go, "build", "-o", str(binary_path), _GO_CMD_PKG],
        cwd=str(repo_root), capture_output=True, text=True,
    )
    if proc.returncode != 0:
        raise ConfigError(
            "failed to rebuild github-mcp-server from source "
            f"(go build exited {proc.returncode}):\n{proc.stderr or proc.stdout}"
        )


# ---------------------------------------------------------------------------
# .env loading
# ---------------------------------------------------------------------------

def load_dotenv(path: str | os.PathLike[str] | None = None) -> None:
    """Load KEY=VALUE lines from a .env file into ``os.environ`` (no override
    of already-set vars). Silent no-op when the file is absent. Minimal parser:
    supports ``#`` comments (whole-line and trailing), ``export KEY=...`` and
    single/double quotes.

    Trailing comments are stripped from UNQUOTED values only, at the first
    whitespace-preceded ``#`` — the form ``.env.example`` itself documents
    (``AGENTICMCPE_LLM_EXTRA_BODY={...}   # comment``). Without
    this the comment lands inside the value and every entry point dies with a
    confusing ``... is not valid JSON``. Quote the value to keep a literal ``#``."""
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
            if val[:1] in ('"', "'"):
                end = val.find(val[0], 1)
                if end >= 0:  # quoted: take the quoted span, drop any trailing comment
                    val = val[1:end]
            else:
                cut = min((i for i in (val.find(" #"), val.find("\t#")) if i >= 0),
                          default=-1)
                if cut >= 0:
                    val = val[:cut].rstrip()
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
    """Raised when required configuration (key/token) is missing."""


_MAX_COMPLETION_TOKENS_MODELS = ("gpt-5", "o1", "o3", "o4")


def _wants_max_completion_tokens(model: str) -> bool:
    m = (model or "").lower()
    return any(marker in m for marker in _MAX_COMPLETION_TOKENS_MODELS)


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
        # Cleared permanently once a provider rejects `temperature` — see chat().
        self._send_temperature = True

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

    @staticmethod
    def _anthropic_chat(client, s: LLMSettings, system: str, user: str,
                        **extra: Any) -> str:
        resp = client.messages.create(
            model=s.model,
            max_tokens=s.max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
            **extra,
        )
        return "".join(
            block.text for block in resp.content if getattr(block, "type", "") == "text"
        )

    def chat(self, system: str, user: str, *, json_mode: bool = False) -> str:
        client = self._ensure_client()
        s = self.settings
        if s.sdk == "anthropic":
            if self._send_temperature:
                try:
                    return self._anthropic_chat(client, s, system, user,
                                                temperature=s.temperature)
                except Exception as e:  # noqa: BLE001 — only the temperature case
                    if "temperature" not in str(e).lower():
                        raise
                    self._send_temperature = False
            return self._anthropic_chat(client, s, system, user)
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        call_kwargs: dict[str, Any] = {
            "model": s.model,
            "messages": messages,
            "temperature": s.temperature,
        }
        if _wants_max_completion_tokens(s.model):
            call_kwargs["max_completion_tokens"] = s.max_tokens
        else:
            call_kwargs["max_tokens"] = s.max_tokens
        if s.extra_body:
            call_kwargs["extra_body"] = s.extra_body
        if json_mode:
            try:
                resp = client.chat.completions.create(
                    response_format={"type": "json_object"}, **call_kwargs
                )
                return resp.choices[0].message.content or ""
            except Exception:
                # Provider rejected response_format — retry plain and rely on
                # extract_json downstream.
                pass
        resp = client.chat.completions.create(**call_kwargs)
        return resp.choices[0].message.content or ""


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
# GitHub token pool
# ---------------------------------------------------------------------------

@dataclass
class GitHubTokenPool:
    """Round-robin pool of GitHub tokens with rotate-on-failure.

    Tokens come from (in priority order): ``GITHUB_TOKENS`` (comma-separated),
    then ``GITHUB_PERSONAL_ACCESS_TOKEN``, then ``GITHUB_TOKEN``.
    """

    tokens: list[str] = field(default_factory=list)
    _idx: int = 0

    @classmethod
    def from_env(cls) -> "GitHubTokenPool":
        tokens: list[str] = []
        pooled = os.environ.get("GITHUB_TOKENS", "")
        if pooled:
            tokens.extend(t.strip() for t in pooled.split(",") if t.strip())
        for single in ("GITHUB_PERSONAL_ACCESS_TOKEN", "GITHUB_TOKEN"):
            v = os.environ.get(single)
            if v and v not in tokens:
                tokens.append(v)
        return cls(tokens=tokens)

    @property
    def available(self) -> bool:
        return bool(self.tokens)

    def current(self) -> str:
        if not self.tokens:
            raise ConfigError(
                "no GitHub token: set GITHUB_PERSONAL_ACCESS_TOKEN, GITHUB_TOKEN, "
                "or GITHUB_TOKENS in env/.env"
            )
        return self.tokens[self._idx % len(self.tokens)]

    def rotate(self) -> str:
        """Advance to the next token (call on rate-limit / 401). Returns it."""
        if len(self.tokens) > 1:
            self._idx = (self._idx + 1) % len(self.tokens)
        return self.current()


# ---------------------------------------------------------------------------
# Aggregate settings
# ---------------------------------------------------------------------------

@dataclass
class Settings:
    llm: LLMSettings
    tokens: GitHubTokenPool
    repo_root: Path = REPO_ROOT
    binary_path: Path = DEFAULT_BINARY
    runs_dir: Path = DEFAULT_RUNS_DIR
    run_id: str = ""
    work_dir: Path = field(default_factory=lambda: DEFAULT_RUNS_DIR)

    @classmethod
    def load(cls, *, provider: str | None = None, run_id: str | None = None) -> "Settings":
        load_dotenv()
        llm = LLMSettings.from_env(provider)
        tokens = GitHubTokenPool.from_env()
        rid = run_id or time.strftime("run-%Y%m%d-%H%M%S")
        runs_dir = Path(os.environ.get("AGENTICMCPE_RUNS_DIR", str(DEFAULT_RUNS_DIR)))
        work_dir = runs_dir / rid
        binary = Path(os.environ.get("AGENTICMCPE_BINARY", str(DEFAULT_BINARY)))
        ensure_binary(binary, REPO_ROOT)
        return cls(
            llm=llm,
            tokens=tokens,
            binary_path=binary,
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
            "github_tokens": len(self.tokens.tokens),
            "binary": str(self.binary_path),
            "binary_exists": self.binary_path.is_file(),
            "run_id": self.run_id,
            "work_dir": str(self.work_dir),
        }


__all__ = [
    "REPO_ROOT",
    "DEFAULT_BINARY",
    "ensure_binary",
    "ConfigError",
    "load_dotenv",
    "PROVIDER_REGISTRY",
    "LLMSettings",
    "LLMClient",
    "extract_json",
    "GitHubTokenPool",
    "Settings",
]
