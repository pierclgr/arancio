"""Permission value types: autonomy levels, tool categories and decisions."""

from dataclasses import dataclass
from enum import Enum

from arancio.core.tools.commands.shell import ShellCommandTool
from arancio.core.tools.files.edit import EditFileTool
from arancio.core.tools.files.read import ReadFileTool
from arancio.core.tools.files.write import WriteFileTool
from arancio.core.tools.web.fetch import FetchWebTool
from arancio.core.tools.web.search import SearchWebTool


class PermissionLevel(Enum):
    """Autonomy level governing how a permitted tool category is executed.

    Attributes:
        NONE: the category is not granted; its tools are never created.
        ASK: prompt the user for confirmation before each tool call.
        AUTO: execute permitted tool calls without confirmation.
    """

    NONE = None
    ASK = "ask"
    AUTO = "auto"


class PermissionCategory(Enum):
    """Capability class grouping tools that perform the same kind of operation.

    Each member's value is the frozen set of tool classes it governs, so the
    mapping is refactor-safe: renaming or moving a tool class updates the
    category through its import rather than leaving a stale string behind.

    Attributes:
        READ: tools that read file contents.
        WRITE: tools that create, write or edit files.
        WEB: tools that access the web.
        EXECUTE: tools that run terminal commands.
    """

    READ = frozenset({ReadFileTool})
    WRITE = frozenset({WriteFileTool, EditFileTool})
    WEB = frozenset({SearchWebTool, FetchWebTool})
    EXECUTE = frozenset({ShellCommandTool})

    @classmethod
    def for_tool(cls, tool_name: str) -> "PermissionCategory | None":
        """Return the category governing the tool with the given class name.

        Args:
            tool_name: the tool's class name (``BaseTool.name``).

        Returns:
            The matching category, or ``None`` when no category governs the
            tool.
        """
        for category in cls:
            if any(tool.__name__ == tool_name for tool in category.value):
                return category
        return None


class PermissionOutcome(Enum):
    """How a requested tool call was resolved.

    Attributes:
        ALLOWED: the call may run.
        DENIED: the user refused the call.
        UNAVAILABLE: no granted category builds the requested tool.
    """

    ALLOWED = "allowed"
    DENIED = "denied"
    UNAVAILABLE = "unavailable"


@dataclass(frozen=True)
class PermissionDecision:
    """Resolved permission for one tool call, for the caller to act on.

    Attributes:
        outcome: how the call was resolved.
        note: the user's note when allowing, or the reason when denying.
    """

    outcome: PermissionOutcome
    note: str | None = None
