"""Abstract runnable tool interface."""

from abc import ABC, abstractmethod
from typing import Any, Type

import yaml
from dynamic_markdown.types.files.base import DynamicMarkdownFile

from arancio.core.constants.path.base import (
    HARNESS_DIR_ROOT_PATH,
    TOOL_DESCRIPTION_FILENAME,
    TOOL_INPUT_SCHEMA_FILENAME,
    TOOLS_HARNESS_PATH,
)
from arancio.core.parsers.tool_result.base import BaseToolResultParser
from arancio.core.tools.session import ToolSession, default_session
from arancio.core.types.messages import ToolResultMessage
from arancio.core.types.tools import ToolSchema
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
            description from disk via ``.reload()`` + re-parse.
        _result_parser: parser converting raw tool output into a tool
            result message.
        _session: shared :class:`ToolSession` for cross-tool
            coordination (e.g. read-first guards). Defaults to the
            module-level :data:`default_session` singleton; pass an
            explicit instance to the constructor to override.
    """

    _result_parser: Type[BaseToolResultParser] = BaseToolResultParser
    _session: ToolSession = default_session

    def __init__(self, session: ToolSession | None = None) -> None:
        """Initialize the tool by loading description and input schema from disk.

        Reads ``description.md`` and ``input_schema.yml`` from
        ``harness/tools/<snake_name>/`` and stores them as instance
        attributes. The description is parsed as dynamic markdown:
        ``<include>``, ``<field>`` and ``<script>`` tags are expanded
        with ``base_dir`` set to the harness root directory (so
        ``<include>`` paths are resolved relative to ``harness/``) and
        the tool instance as the field source.

        Args:
            session: optional :class:`ToolSession` to override the
                shared :data:`default_session` singleton on this
                instance. Useful in tests that need an isolated
                cross-tool state.

        Raises:
            FileNotFoundError: when the harness directory for this
                tool does not exist.
        """
        if session is not None:
            self._session = session

        self._harness_dir = TOOLS_HARNESS_PATH / camel_to_snake(self.name)

        if not self._harness_dir.is_dir():
            raise FileNotFoundError(
                f"Harness directory {self._harness_dir} for {self.name} not found."
            )

        self._description_file = DynamicMarkdownFile(
            self._harness_dir / TOOL_DESCRIPTION_FILENAME
        )
        self._description_file.parse(base_dir=HARNESS_DIR_ROOT_PATH, tool=self)
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

        Args:
            call_id: identifier of the tool call this result answers.
            **kwargs: tool arguments matching ``input_schema``.

        Returns:
            The tool output build as message.
        """
        try:
            output = self._call(**kwargs)
            is_error = False
        except Exception as exc:
            output = f"Error while executing {self.name}: {exc}"
            is_error = True

        return self._result_parser.parse(
            call_id=call_id,
            output=output,
            is_error=is_error,
        )

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
