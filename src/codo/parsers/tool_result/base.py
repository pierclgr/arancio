"""Abstract parser for tool-specific results."""

from abc import abstractmethod
from typing import Any

from codo.parsers.base import Parser
from codo.types.messages import ToolResultMessage


class BaseToolResultParser(Parser):
    """Base interface for converting raw tool output into a message."""

    @classmethod
    @abstractmethod
    def parse(
        cls,
        call_id: str,
        output: Any,
        is_error: bool = False,
    ) -> ToolResultMessage:
        """Parse raw tool output into a normalized result message.

        Args:
            call_id: identifier of the tool call this result answers.
            output: raw output returned by a tool implementation.
            is_error: whether the tool execution failed.

        Returns:
            A parsed tool result message. Implementations must return a
            :class:`ToolErrorMessage` when ``is_error`` is true.
        """
        raise NotImplementedError("Subclasses must implement this method.")
