"""Settings from env. One place that knows variable names and defaults."""
from __future__ import annotations

import os
from dataclasses import dataclass, field, replace
from pathlib import Path

from dotenv import load_dotenv

PACKAGE_ROOT = Path(__file__).resolve().parent.parent


def _load_env_files() -> None:
    # First file wins per key (override=False): cwd, then repo, then ~/.hx.
    for p in (Path.cwd() / ".env", PACKAGE_ROOT / ".env", Path.home() / ".hx" / ".env"):
        if p.is_file():
            load_dotenv(p, override=False)


def _env(name: str, default: str = "") -> str:
    return (os.getenv(name) or default).strip()


def _env_int(name: str, default: int) -> int:
    try:
        return int(_env(name, str(default)))
    except ValueError:
        return default


def _env_bool(name: str, default: bool = False) -> bool:
    raw = _env(name)
    if not raw:
        return default
    return raw.lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    api_key: str
    base_url: str = "https://api.groq.com/openai/v1"
    model: str = "openai/gpt-oss-120b"          # main agent loop
    fast_model: str = "openai/gpt-oss-20b"      # case relevance selection
    reflect_model: str = "openai/gpt-oss-120b"  # post-turn self-review
    reasoning_effort: str = "low"               # gpt-oss knob; "" to omit
    max_tokens: int = 4096
    context_budget: int = 24000                 # approx tokens before old tool output is cleared
    max_steps: int = 30                         # tool-use iterations per turn
    bash_timeout: int = 60
    mongodb_uri: str = ""
    db_name: str = "hx_memory"
    home: Path = field(default_factory=lambda: Path.home() / ".hx")
    workdir: Path = field(default_factory=Path.cwd)
    auto_approve: bool = False                  # skip y/n confirmation for bash / writes outside workdir
    memory: bool = True                         # recall + reflection on/off (for A/B demos)

    def with_(self, **kw) -> "Settings":
        return replace(self, **{k: v for k, v in kw.items() if v is not None})


def load_settings(**overrides) -> Settings:
    _load_env_files()
    api_key = _env("HX_API_KEY") or _env("GROQ_API_KEY") or _env("OPENAI_API_KEY")
    model = _env("HX_MODEL", "openai/gpt-oss-120b")
    s = Settings(
        api_key=api_key,
        base_url=_env("HX_BASE_URL", "https://api.groq.com/openai/v1"),
        model=model,
        fast_model=_env("HX_FAST_MODEL", "openai/gpt-oss-20b"),
        reflect_model=_env("HX_REFLECT_MODEL", model),
        reasoning_effort=_env("HX_REASONING_EFFORT", "low"),
        max_tokens=_env_int("HX_MAX_TOKENS", 4096),
        context_budget=_env_int("HX_CONTEXT_BUDGET", 24000),
        max_steps=_env_int("HX_MAX_STEPS", 30),
        bash_timeout=_env_int("HX_BASH_TIMEOUT", 60),
        mongodb_uri=_env("MONGODB_URI"),
        db_name=_env("HX_DB", "hx_memory"),
        home=Path(_env("HX_HOME", str(Path.home() / ".hx"))).expanduser(),
        workdir=Path.cwd(),
        auto_approve=_env_bool("HX_AUTO_APPROVE"),
        memory=not _env_bool("HX_NO_MEMORY"),
    )
    return s.with_(**overrides)
