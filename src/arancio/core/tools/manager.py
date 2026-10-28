"""Tool catalog: registered definitions, active instances and execution."""

from collections.abc import Collection
from functools import partial

from arancio.core.clients.base import Client
from arancio.core.hooks.manager import HookManager
from arancio.core.messages import (
    Message,
    ToolCallMessage,
    ToolErrorMessage,
    ToolResultMessage,
)
from arancio.core.permissions.types import PermissionCategory
from arancio.core.tools.base import Tool
from arancio.core.tools.commands.shell import ShellCommandTool
from arancio.core.tools.definition import ToolDefinition
from arancio.core.tools.files.edit import EditFileTool
from arancio.core.tools.files.read import ReadFileTool
from arancio.core.tools.files.write import WriteFileTool
from arancio.core.tools.schema import ToolSchema
from arancio.core.tools.session import ToolSession
from arancio.core.tools.web.fetch import FetchWebTool
from arancio.core.tools.web.search import SearchWebTool


class ToolManager:
    """Own the tool catalog and run every tool the agent calls.

    The catalog maps each tool name to its :class:`ToolDefinition`. The built-in
    tools register at construction; plugin tools register through
    :meth:`register_tool_definition`. :meth:`refresh_tools` builds instances only for
    the allowed categories, so a tool whose category is not granted is never
    instantiated.
    Permission checks stay outside this class: the agent calls :meth:`run_tool_call`
    only after an allowed decision.

    Attributes:
        _definitions: registered definitions, keyed by tool name.
        _tools: the instances active for the current grants, keyed by tool name.
        _hook_manager: injected into every tool this manager builds.
        _tool_session: the file-read state injected into every tool this manager
            builds.
    """

    def __init__(
        self,
        web_summary_client: Client,
        hook_manager: HookManager,
        tool_session: ToolSession,
    ) -> None:
        """Initialize the manager and register the built-in tools.

        Args:
            web_summary_client: the client injected into :class:`FetchWebTool`;
                held by reference so its settings can be changed in place at
                runtime and seen by the tool.
            hook_manager: the hook manager injected into every tool this
                manager builds.
            tool_session: the file-read state injected into every tool this manager
                builds; the same instance survives every :meth:`refresh_tools`.
        """
        self._definitions: dict[str, ToolDefinition] = {}
        self._tools: dict[str, Tool] = {}
        self._hook_manager = hook_manager
        self._tool_session = tool_session

        builtin_definitions = [
            ToolDefinition(
                ReadFileTool.__name__, PermissionCategory.READ, ReadFileTool
            ),
            ToolDefinition(
                WriteFileTool.__name__, PermissionCategory.WRITE, WriteFileTool
            ),
            ToolDefinition(
                EditFileTool.__name__, PermissionCategory.WRITE, EditFileTool
            ),
            ToolDefinition(
                SearchWebTool.__name__, PermissionCategory.WEB, SearchWebTool
            ),
            ToolDefinition(
                FetchWebTool.__name__,
                PermissionCategory.WEB,
                partial(FetchWebTool, client=web_summary_client),
            ),
            ToolDefinition(
                ShellCommandTool.__name__, PermissionCategory.EXECUTE, ShellCommandTool
            ),
        ]
        for definition in builtin_definitions:
            self.register_tool_definition(definition)

    def __repr__(self) -> str:
        """Return a developer-friendly representation of the active tools.

        Returns:
            The class name followed by the active tool instances.
        """
        return f"{type(self).__name__}(tools={list(self._tools.values())!r})"

    def register_tool_definition(self, definition: ToolDefinition) -> None:
        """Add a tool definition to the catalog.

        The tool is built on the next :meth:`refresh_tools` that allows its category.

        Args:
            definition: the definition to add.

        Raises:
            ValueError: when a definition with the same name is already
                registered.
        """
        if definition.name in self._definitions:
            raise ValueError(f"Tool {definition.name!r} is already registered.")
        self._definitions[definition.name] = definition

    def get_tool_category(self, name: str) -> PermissionCategory | None:
        """Return the permission category of a registered tool.

        Args:
            name: the tool name.

        Returns:
            The category named by the tool's definition, or ``None`` when no
            definition has this name.
        """
        definition = self._definitions.get(name)
        return definition.category if definition is not None else None

    def refresh_tools(self, categories: Collection[PermissionCategory]) -> None:
        """Rebuild the active tools for the allowed categories.

        Every allowed tool gets a new instance. The injected session is kept,
        so the file-read guard survives the rebuild.

        Args:
            categories: the categories whose tools may be built.
        """
        self._tools = {
            name: definition.factory(
                hook_manager=self._hook_manager,
                tool_session=self._tool_session,
            )
            for name, definition in self._definitions.items()
            if definition.category in categories
        }

    @property
    def tool_schemas(self) -> list[ToolSchema]:
        """Return the schemas of the active tools.

        Returns:
            One schema per active tool, in catalog order.
        """
        return [tool.schema() for tool in self._tools.values()]

    def run_tool_call(
        self, call: ToolCallMessage
    ) -> tuple[ToolResultMessage, list[Message]]:
        """Run an allowed tool call and return its result alongside hook messages.

        Args:
            call: the tool call to run.

        Returns:
            The tool result and hook messages in dispatch order. A call naming
            no active tool returns a tool error and no hook messages.
        """
        tool = self._tools.get(call.name)
        if tool is None:
            return ToolErrorMessage(content=f"Unknown tool {call.name}", id=call.id), []
        return tool.call(call_id=call.id, **(call.arguments or {}))
