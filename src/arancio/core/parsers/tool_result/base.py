"""Abstract parser for tool-specific results."""

from abc import abstractmethod
from typing import Any

from arancio.core.messages import ToolErrorMessage, ToolResultMessage
from arancio.core.parsers.base import Parser


class BaseToolResultParser(Parser):
    """Base interface for converting raw tool output into a message.

    Concrete parsers implement :meth:`_render` to turn a structured (dict) tool output
    into its human-readable display text, and may override :meth:`_failed` when the
    payload itself can denote a failure. The shared :meth:`parse` orchestrates both and
    builds the result message.
    """

    @classmethod
    def parse(
        cls,
        call_id: str,
        output: Any,
        is_error: bool = False,
    ) -> ToolResultMessage:
        """Parse raw tool output into a normalized result message.

        Non-dict outputs (e.g. an error string raised by the tool) are passed
        through unrendered, so ``display_text`` falls back to ``content``. Dict
        outputs are rendered by :meth:`_render`, and :meth:`_failed` decides
        whether the payload itself denotes a failure.

        Args:
            call_id: identifier of the tool call this result answers.
            output: raw output returned by a tool implementation.
            is_error: whether the tool execution failed.

        Returns:
            A :class:`ToolErrorMessage` when the call failed, otherwise a
            :class:`ToolResultMessage`.
        """
        display_text = None
        if isinstance(output, dict):
            display_text = cls._render(output)
            is_error = is_error or cls._failed(output)
        return cls._build_message(call_id, output, display_text, is_error)

    @classmethod
    @abstractmethod
    def _render(cls, output: dict) -> str:
        """Render the display text for a structured tool output.

        Args:
            output: the structured tool output to render.

        Returns:
            The human-readable string shown to the user.
        """
        raise NotImplementedError("Subclasses must implement this method.")

    @classmethod
    def _failed(cls, output: dict) -> bool:
        """Report whether a structured tool output denotes a failure.

        Args:
            output: the structured tool output to inspect.

        Returns:
            True when the payload itself denotes a failed execution. Defaults
            to False; subclasses override when the payload can carry failure.
        """
        return False

    @staticmethod
    def _build_message(
        call_id: str,
        content: Any,
        display_text: Any = None,
        is_error: bool = False,
    ) -> ToolResultMessage:
        """Build the result or error message for a parsed tool output.

        Args:
            call_id: identifier of the tool call this result answers.
            content: the raw tool output consumed by the agent.
            display_text: the rendered human-readable string shown in the UI;
                defaults to ``content`` when not specified.
            is_error: whether the tool execution failed.

        Returns:
            A :class:`ToolErrorMessage` when ``is_error`` is true, otherwise a
            :class:`ToolResultMessage`.
        """
        message_class = ToolErrorMessage if is_error else ToolResultMessage
        return message_class(content=content, id=call_id, display_text=display_text)
