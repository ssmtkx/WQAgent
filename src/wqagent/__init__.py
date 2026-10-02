"""WQ Agent - a small, workspace-scoped tool-calling agent.

Public surface::

    from wqagent import RuntimeState          # runtime state shared by tools + agent loop
    from wqagent import build_tools           # StructuredTool list bound to a RuntimeState
    from wqagent import create_model          # ChatOpenAI factory that reads .env

The package itself only imports the standard library; heavy dependencies
(langchain, typer, openai) are imported by the modules that need them.
"""

from wqagent.core.state import RuntimeState

__version__ = "0.1.0"

__all__ = ["RuntimeState", "__version__"]
