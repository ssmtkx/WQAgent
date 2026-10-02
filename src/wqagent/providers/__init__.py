"""Model provider factories.

Only one provider for now: an OpenAI-compatible chat model built on
``langchain_openai.ChatOpenAI``. Configuration comes from ``.env`` (see
``.env.example``) and can be overridden by CLI flags.
"""

from wqagent.providers.openai_provider import (
    DEFAULT_MODEL,
    ProviderConfigError,
    create_model,
    describe_config,
    load_env,
)

__all__ = [
    "DEFAULT_MODEL",
    "ProviderConfigError",
    "create_model",
    "describe_config",
    "load_env",
]
