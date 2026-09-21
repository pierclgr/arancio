"""Abstract runnable tool interface."""

from abc import ABC, abstractmethod
from typing import Any, Type

import yaml
from dynamic_markdown.types.files.base import DynamicMarkdownFile

from arancio.core.constants.path import (
    TOOL_DESCRIPTION_FILENAME,
    TOOL_INPUT_SCHEMA_FILENAME,
    TOOLS_HARNESS_PATH,
)
from arancio.core.hooks.manager import HookManager
from arancio.core.hooks.types import Hook
from arancio.core.messages import ToolResultMessage
from arancio.core.parsers.tool_result.base import BaseToolResultParser
from arancio.core.tools.schema import ToolSchema
from arancio.core.tools.session import ToolSession, shared_session
from arancio.core.utils.naming import camel_to_snake


class BaseTool(ABC):
    """Base class for runnable tools exposed to LLM clients.

    Subclasses implement :meth:`_call` to execute the tool and return
    raw output. The public :meth:`call` wrapper executes :meth:`_call`
    and converts its output into a :class:`ToolResultMessage`.

    Attributes:
        description: natural-language description,
            loaded from the tool's harness directory.
        input_schema: JSON Schema describing the tool arguments,
            loaded from the harness directory.
        _description_file: the :class:`DynamicMarkdownFile` backing
            :attr:`description`, retained so callers can refresh the
            description from disk via ``.reload()``.
        _result_parser: parser converting raw tool output into a tool
            result message.
        _session: the process-wide :class:`ToolSession` for cross-tool
            coordination (e.g. read-first guards), shared by every tool
            through this class attribute.
        _hook_manager: the hook manager :meth:`call` dispatches
            ``before_tool_call``/``after_tool_call``/``error`` through,
            injected by the caller (typically
            :meth:`~arancio.core.tools.manager.ToolManager.create_tools`).
    """

    _result_parser: Type[BaseToolResultParser] = BaseToolResultParser
    _session: ToolSession = shared_session

    def __init__(self, hook_manager: HookManager) -> None:
        """Initialize the tool by loading description and input schema from disk.

        Reads ``description.md`` and ``input_schema.yml`` from
        ``harness/tools/<snake_name>/`` and stores them as instance
        attributes. The description is parsed as dynamic markdown:
        ``<include>``/``@path``, ``<field>`` and ``<script>`` tags are
        expanded with the tool instance as the field source; relative
        include and script targets resolve against ``description.md``'s
        own directory.

        Args:
            hook_manager: the hook manager :meth:`call` dispatches through.

        Raises:
            FileNotFoundError: when the harness directory for this
                tool does not exist.
        """
        self._hook_manager = hook_manager
        self._harness_dir = TOOLS_HARNESS_PATH / camel_to_snake(self.name)

        if not self._harness_dir.is_dir():
            raise FileNotFoundError(
                f"Harness directory {self._harness_dir} for {self.name} not found."
            )

        self._description_file = DynamicMarkdownFile(
            self._harness_dir / TOOL_DESCRIPTION_FILENAME, tool=self
        )
        self.description = self._description_file.content

        raw_input_schema = yaml.safe_load(
            (self._harness_dir / TOOL_INPUT_SCHEMA_FILENAME).read_text()
        )
        self.input_schema = {
            "type": "object",
            "additionalProperties": False,
            **raw_input_schema,
        }

    def __repr__(self) -> str:
        """Return a developer-friendly representation of the tool.

        Returns:
            The tool's class name followed by empty parentheses.
        """
        return f"{type(self).__name__}()"

    @property
    def name(self) -> str:
        """Return the tool's name, derived from the python class name.

        Returns:
            The class name of this tool instance.
        """
        return type(self).__name__

    def call(self, call_id: str, **kwargs) -> ToolResultMessage:
        """Execute the tool and build ToolResultMessage output.

        Dispatches ``before_tool_call`` before running, ``error`` (with
        ``source="tool"``) when :meth:`_call` raises, and ``after_tool_call``
        once the result is built, successful or not. Only a :meth:`_call`
        failure is caught here — a hook handler's own exception during any
        of these three dispatches propagates out of this method instead of
        becoming a :class:`~arancio.core.messages.ToolErrorMessage`.

        Args:
            call_id: identifier of the tool call this result answers.
            **kwargs: tool arguments matching ``input_schema``.

        Returns:
            The tool output build as message.
        """
        self._hook_manager.run(
            Hook.BEFORE_TOOL_CALL, name=self.name, call_id=call_id, arguments=kwargs
        )
        try:
            output = self._call(**kwargs)
            is_error = False
        except Exception as exc:
            self._hook_manager.run(
                Hook.ERROR,
                source="tool",
                name=self.name,
                call_id=call_id,
                arguments=kwargs,
                error=exc,
            )
            output = f"Error while executing {self.name}: {exc}"
            is_error = True

        result = self._result_parser.parse(
            call_id=call_id,
            output=output,
            is_error=is_error,
        )
        self._hook_manager.run(
            Hook.AFTER_TOOL_CALL,
            name=self.name,
            call_id=call_id,
            arguments=kwargs,
            result=result,
        )
        return result

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
