"""Abstract runnable tool interface."""

from abc import ABC, abstractmethod
from typing import Any, Type

from dynamic_markdown.types.files.base import DynamicMarkdownFile

from arancio.core.builders.tool_input_schemas.base import InputSchemaBuilder
from arancio.core.hooks.base import Hook
from arancio.core.hooks.manager import HookManager
from arancio.core.messages.base import Message
from arancio.core.messages.tool import ToolCallMessage, ToolResultMessage
from arancio.core.parsers.tool_results.base import ToolResultParser
from arancio.core.tools.constants import TOOLS_HARNESS_PATH
from arancio.core.tools.schema import ToolSchema
from arancio.core.tools.session import ToolSession
from arancio.core.utils.naming import camel_to_snake


class Tool(ABC):
    """Base class for runnable tools exposed to LLM clients.

    Subclasses implement :meth:`_call` to execute the tool and return
    raw output. The public :meth:`call` wrapper executes :meth:`_call`
    and returns its :class:`ToolResultMessage` alongside hook messages.

    Attributes:
        description: natural-language description,
            loaded from the tool's harness directory.
        input_schema: JSON Schema describing the tool arguments,
            derived from the signature of :meth:`_call`.
        _result_parser: parser converting raw tool output into a tool
            result message.
        _tool_session: the injected :class:`ToolSession` for cross-tool
            coordination (e.g. read-first guards), shared by every tool of
            the active conversation.
        _hook_manager: the hook manager :meth:`call` dispatches
            ``before_tool_call``/``after_tool_call``/``error`` through,
            injected by the caller (typically
            :class:`~arancio.core.tools.manager.ToolManager`).
    """

    _result_parser: Type[ToolResultParser] = ToolResultParser

    def __init__(self, hook_manager: HookManager, tool_session: ToolSession) -> None:
        """Initialize the tool with its dependencies, description and input schema.

        The description comes from :meth:`_load_description`; the input schema is
        derived from the signature and docstring of :meth:`_call` and wrapped as a
        closed JSON object.

        Args:
            hook_manager: the hook manager :meth:`call` dispatches through.
            tool_session: the file-read state shared by every tool of the active
                conversation.
        """
        self._hook_manager = hook_manager
        self._tool_session = tool_session
        self.description = self._load_description()
        self.input_schema = {
            "type": "object",
            "additionalProperties": False,
            **InputSchemaBuilder.build(self._call),
        }

    def _load_description(self) -> str:
        """Load the tool's description from its harness file.

        Reads ``harness/tools/<snake_name>.md``. The description is parsed as
        dynamic markdown: ``<include>``/``@path``, ``<field>`` and ``<script>``
        tags are expanded with the tool instance as the field source; relative
        include and script targets resolve against the description file's own
        directory.

        Returns:
            The description.

        Raises:
            FileNotFoundError: when the description file for this tool does not
                exist.
        """
        description_path = TOOLS_HARNESS_PATH / f"{camel_to_snake(self.name)}.md"

        if not description_path.is_file():
            raise FileNotFoundError(
                f"Description file {description_path} for {self.name} not found."
            )

        return DynamicMarkdownFile(description_path, tool=self).content

    def __repr__(self) -> str:
        """Return a developer-friendly representation of the tool.

        Returns:
            The tool's class name followed by empty parentheses.
        """
        return f"{type(self).__name__}()"

    @property
    def name(self) -> str:
        """The tool's name, derived from the Python class name.

        Returns:
            The class name of this tool instance.
        """
        return type(self).__name__

    def call(
        self, tool_call: ToolCallMessage
    ) -> tuple[ToolResultMessage, list[Message]]:
        """Execute an existing call, sharing its identity with hook handlers.

        Handlers edit tool_call.arguments before execution and tool_call.id before
        result parsing. The tool has already been selected: editing tool_call.name
        here does not route to another tool. Change the target in model-response
        hooks, before permission resolution and selection.

        Only _call failures become tool errors. Ordinary hook failures propagate.
        Returned results keep edits made by after-tool handlers.

        Args:
            tool_call: the original model call, or a call created for direct execution.

        Returns:
            The result and hook messages in dispatch order.
        """
        messages = self._hook_manager.run(Hook.BEFORE_TOOL_CALL, tool_call=tool_call)
        try:
            output = self._call(**tool_call.arguments)
            is_error = False
        except Exception as exc:
            messages.extend(
                self._hook_manager.run(
                    Hook.ERROR,
                    source="tool",
                    tool_call=tool_call,
                    error=exc,
                )
            )
            output = f"Error while executing {self.name}: {exc}"
            is_error = True

        result = self._result_parser.parse(
            call_id=tool_call.id,
            output=output,
            is_error=is_error,
        )
        messages.extend(
            self._hook_manager.run(
                Hook.AFTER_TOOL_CALL,
                tool_call=tool_call,
                result=result,
            )
        )
        return result, messages

    @abstractmethod
    def _call(self, **kwargs) -> Any:
        """Execute the tool and return its output.

        Args:
            **kwargs: the tool arguments matching ''input_schema''

        Returns:
            The output of the tool execution

        Raises:
            NotImplementedError: when the method is not implemented by the subclass.
        """
        raise NotImplementedError("Subclasses must implement this method.")

    def schema(self) -> ToolSchema:
        """Return the wire-format schema definition for this tool.

        Returns:
            A :class:`ToolSchema` carrying the tool's name, description
            and input schema for serialization into provider payloads.
        """
        return ToolSchema(
            name=self.name,
            description=self.description,
            input_schema=self.input_schema,
        )
