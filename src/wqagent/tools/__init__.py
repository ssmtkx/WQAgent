"""Workspace tools exposed to the model.

``build_tools(state)`` is the entry point used by the CLI and the agent loop::

    tools = build_tools(state)          # list[StructuredTool]
    model = create_model().bind_tools(tools)
"""

from wqagent.tools.bash_tool import BashInput, BashTool
from wqagent.tools.file_tools import (
    FileEditInput,
    FileEditTool,
    FileReadInput,
    FileReadTool,
    FileWriteInput,
    FileWriteTool,
)
from wqagent.tools.grep_tool import GrepInput, GrepTool
from wqagent.tools.registry import (
    TOOL_CLASSES,
    build_tool_instances,
    build_tools,
    describe_tools,
    tool_names,
)

__all__ = [
    "BashInput",
    "BashTool",
    "FileEditInput",
    "FileEditTool",
    "FileReadInput",
    "FileReadTool",
    "FileWriteInput",
    "FileWriteTool",
    "GrepInput",
    "GrepTool",
    "TOOL_CLASSES",
    "build_tool_instances",
    "build_tools",
    "describe_tools",
    "tool_names",
]
