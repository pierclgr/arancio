"""Tool call and result messages."""

from typing import Any, ClassVar

from arancio.core.messages.base import ErrorMessage, Message


class ToolCallMessage(Message):
    """Model request to invoke a tool.

    Attributes:
        role: the role of the message.
        id: provider-issued call identifier.
        name: name of the tool the model wants to invoke.
        arguments: keyword arguments for the tool, parsed from JSON.
    """

    role: ClassVar[str] = "tool_call"

    def __init__(
        self,
        content: Any,
        id: str,
        name: str,
        arguments: dict,
        display_text: Any = None,
        in_history: bool = True,
    ) -> None:
        """Store the tool-call identity alongside the base message fields.

        Args:
            content: the raw content of the message, consumed by the agent.
            id: provider-issued call identifier.
            name: name of the tool the model wants to invoke.
            arguments: keyword arguments for the tool, parsed from JSON.
            display_text: what is shown to the user through the UI; defaults
                to ``content`` when not specified.
            in_history: whether the message is part of the model's context.
        """
        super().__init__(content, display_text, in_history)
        self.id = id
        self.name = name
        self.arguments = arguments


class ToolResultMessage(Message):
    """Result of executing a tool, returned to the model.

    Attributes:
        role: the role of the message.
        id: identifier of the tool call this result responds to.
    """

    role: ClassVar[str] = "tool_result"

    def __init__(
        self,
        content: Any,
        id: str,
        display_text: Any = None,
        in_history: bool = True,
    ) -> None:
        """Store the answered tool-call id alongside the base message fields.

        A tool result is the outcome the model asked for, so ``in_history``
        defaults to ``True`` here, and that default is what keeps
        :class:`ToolErrorMessage` in model history: its MRO reaches
        :class:`ErrorMessage`, which defaults to ``False``, only afterwards.

        Args:
            content: the raw content of the message, consumed by the agent.
            id: identifier of the tool call this result responds to.
            display_text: what is shown to the user through the UI; defaults
                to ``content`` when not specified.
            in_history: whether the message is part of the model's context.
        """
        super().__init__(content, display_text, in_history)
        self.id = id


class ToolErrorMessage(ToolResultMessage, ErrorMessage):
    """Failed tool result surfaced as an error to the consumer.

    Attributes:
        role: the role of the message.
    """

    role: ClassVar[str] = "tool_error"
