"""Tool registry - the single place where the agent's tools are assembled.

``build_tools(state)`` returns the ``StructuredTool`` list that is handed to
``model.bind_tools()``. Adding a tool means adding its class to
:data:`TOOL_CLASSES`; nothing else in the project needs to change.
"""

from __future__ import annotations

from typing import Any, Sequence

from langchain_core.tools import StructuredTool

from wqagent.core.state import RuntimeState
from wqagent.tools.bash_tool import BashTool
from wqagent.tools.file_tools import FileEditTool, FileReadTool, FileWriteTool
from wqagent.tools.grep_tool import GrepTool

__all__ = [
    "TOOL_CLASSES",
    "build_tool_instances",
    "build_tools",
    "describe_tools",
]

#: Registration order is also the order the tools are listed in the system prompt.
TOOL_CLASSES: tuple[type, ...] = (
    FileReadTool,
    FileWriteTool,
    FileEditTool,
    GrepTool,
    BashTool,
)


def build_tool_instances(state: RuntimeState) -> list[Any]:
    """Instantiate every registered tool class for *state*.

    The returned objects are plain Python wrappers with a ``run`` method - handy
    for tests and direct scripting. Use :func:`build_tools` for the LangChain
    tools that are bound to a model.
    """
    return [tool_class(state) for tool_class in TOOL_CLASSES]


def build_tools(state: RuntimeState) -> list[StructuredTool]:
    """Return every tool bound to *state*, ready for ``model.bind_tools()``."""
    return [instance.as_tool() for instance in build_tool_instances(state)]


def describe_tools() -> list[dict[str, Any]]:
    """Metadata for each registered tool: name, description and argument schema."""
    return [
        {
            "name": tool_class.name,
            "description": tool_class.description,
            "args_schema": tool_class.args_schema,
        }
        for tool_class in TOOL_CLASSES
    ]


def tool_names() -> Sequence[str]:
    """Names of the registered tools, in registration order."""
    return [tool_class.name for tool_class in TOOL_CLASSES]
