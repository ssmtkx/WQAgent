"""``create_model()`` - a ChatOpenAI factory configured from ``.env``.

Environment variables (first match wins)::

    API_KEY     / OPENAI_API_KEY      -> api key        (required)
    MODEL       / OPENAI_MODEL        -> model name     (default: gpt-4o-mini)
    BASE_URL    / OPENAI_BASE_URL     -> endpoint       (default: api.openai.com)

The ``.env`` file is looked up in this order: the explicit ``env_file``
argument, ``$WQ_ENV_FILE``, the nearest ``.env`` above the current directory,
then ``<project root>/.env``.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from dotenv import find_dotenv, load_dotenv

__all__ = ["DEFAULT_MODEL", "ProviderConfigError", "create_model", "describe_config", "load_env"]

DEFAULT_MODEL = "gpt-4o-mini"
DEFAULT_TIMEOUT_SECONDS = 120.0

ENV_KEYS_API_KEY = ("API_KEY", "OPENAI_API_KEY")
ENV_KEYS_MODEL = ("MODEL", "OPENAI_MODEL")
ENV_KEYS_BASE_URL = ("BASE_URL", "OPENAI_BASE_URL")

_LOCAL_HOSTS = ("localhost", "127.0.0.1", "0.0.0.0", "::1", "host.docker.internal")


class ProviderConfigError(RuntimeError):
    """The chat model cannot be configured (usually a missing API key)."""


def _project_root() -> Path:
    """Repository root: ``<root>/src/wqagent/providers/openai_provider.py``."""
    return Path(__file__).resolve().parents[3]


def load_env(env_file: str | Path | None = None, *, override: bool = False) -> Path | None:
    """Load a ``.env`` file if one can be found; return the path that was used."""
    candidates: list[Path] = []
    if env_file:
        candidates.append(Path(env_file).expanduser())
    from_env_var = os.environ.get("WQ_ENV_FILE")
    if from_env_var:
        candidates.append(Path(from_env_var).expanduser())
    found = find_dotenv(usecwd=True)
    if found:
        candidates.append(Path(found))
    candidates.append(_project_root() / ".env")

    for candidate in candidates:
        try:
            if candidate.is_file():
                load_dotenv(candidate, override=override)
                return candidate
        except OSError:  # pragma: no cover - unreadable .env should not be fatal
            continue
    return None


def _first_env(names: tuple[str, ...]) -> str | None:
    for name in names:
        value = os.environ.get(name)
        if value and value.strip():
            return value.strip()
    return None


def _is_local_endpoint(base_url: str | None) -> bool:
    if not base_url:
        return False
    lowered = base_url.lower()
    return any(host in lowered for host in _LOCAL_HOSTS)


def mask_secret(value: str | None) -> str:
    """Render a secret for logs: first/last four characters only."""
    if not value:
        return "(missing)"
    if len(value) <= 12:
        return "*" * len(value)
    return f"{value[:4]}...{value[-4:]}"


def _resolve_settings(
    *,
    model: str | None,
    api_key: str | None,
    base_url: str | None,
    env_file: str | Path | None,
) -> dict[str, Any]:
    env_path = load_env(env_file)
    resolved_model = (model or _first_env(ENV_KEYS_MODEL) or DEFAULT_MODEL).strip()
    resolved_base = base_url or _first_env(ENV_KEYS_BASE_URL)
    if resolved_base:
        resolved_base = resolved_base.strip().rstrip("/")
    resolved_key = (api_key or _first_env(ENV_KEYS_API_KEY) or "").strip()
    return {
        "env_file": str(env_path) if env_path else None,
        "model": resolved_model,
        "base_url": resolved_base,
        "api_key": resolved_key,
    }


def describe_config(
    *,
    model: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    env_file: str | Path | None = None,
) -> dict[str, Any]:
    """Resolve the configuration without creating a client (never raises)."""
    settings = _resolve_settings(
        model=model, api_key=api_key, base_url=base_url, env_file=env_file
    )
    return {
        "env_file": settings["env_file"],
        "model": settings["model"],
        "base_url": settings["base_url"] or "https://api.openai.com/v1 (default)",
        "api_key": mask_secret(settings["api_key"]),
        "api_key_present": bool(settings["api_key"]),
    }


def create_model(
    *,
    model: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    temperature: float = 0.0,
    timeout: float | None = None,
    max_retries: int = 2,
    env_file: str | Path | None = None,
    **kwargs: Any,
) -> Any:
    """Create the ``ChatOpenAI`` client described by ``.env`` and/or arguments.

    Args:
        model: Model name; falls back to ``MODEL`` then :data:`DEFAULT_MODEL`.
        api_key: API key; falls back to ``API_KEY``/``OPENAI_API_KEY``.
        base_url: OpenAI-compatible endpoint; falls back to ``BASE_URL``.
        temperature: Sampling temperature.
        timeout: Per-request timeout in seconds.
        max_retries: Retries for transient API failures.
        env_file: Explicit ``.env`` file.
        **kwargs: Forwarded to ``ChatOpenAI``.

    Raises:
        ProviderConfigError: No API key was found for a remote endpoint.
    """
    settings = _resolve_settings(
        model=model, api_key=api_key, base_url=base_url, env_file=env_file
    )
    resolved_key: str = settings["api_key"]
    resolved_base: str | None = settings["base_url"]
    if not resolved_key:
        if not _is_local_endpoint(resolved_base):
            raise ProviderConfigError(
                "no API key found. Put API_KEY=... in a .env file (copy .env.example), "
                "export API_KEY, or pass --api-key."
            )
        resolved_key = "not-needed"  # local servers usually ignore the key

    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        model=settings["model"],
        api_key=resolved_key,
        base_url=resolved_base,
        temperature=temperature,
        timeout=timeout if timeout is not None else DEFAULT_TIMEOUT_SECONDS,
        max_retries=max_retries,
        **kwargs,
    )
