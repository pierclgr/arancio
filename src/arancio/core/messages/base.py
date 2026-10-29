"""Base conversation message types."""

from typing import Any, ClassVar


class Message:
    """Base normalized conversation message.

    A message states whether it belongs in the model's context. Where it goes
    otherwise is each consumer's own call: core owns neither the UI nor a
    session, so it describes the message rather than routing it.

    Attributes:
        role: the role of the message.
        content: the raw content of the message, consumed by the agent.
        display_text: what is shown to the user through the UI; defaults to
            ``content`` when not specified.
        in_history: whether the message is part of the model's context.
    """

    role: ClassVar[str]

    def __init__(
        self,
        content: Any,
        display_text: Any = None,
        in_history: bool = True,
    ) -> None:
        """Store the message content, its UI display text and its routing flags.

        Args:
            content: the raw content of the message, consumed by the agent.
            display_text: what is shown to the user through the UI; defaults
                to ``content`` when not specified.
            in_history: whether the message is part of the model's context.
        """
        self.content = content
        self.display_text = content if display_text is None else display_text
        self.in_history = in_history

    def __eq__(self, other: object) -> bool:
        """Compare messages by exact type and attribute values.

        Args:
            other: the object to compare against.

        Returns:
            True when other is the same message type with equal attributes.
        """
        return type(self) is type(other) and self.__dict__ == other.__dict__


class ChunkMessage(Message):
    """Chunk of text from the model.

    Attributes:
        role: the role of the message.
    """

    role: ClassVar[str] = "chunk"


class ErrorMessage(Message):
    """Runtime error surfaced to the agent consumer.

    An error reports what went wrong in this run, not what the conversation
    was, so it stays out of the model's context unless a caller says otherwise.
    :class:`ToolErrorMessage` is the exception and keeps the tool-result
    default, since a failed call is the outcome the model asked for.

    Attributes:
        role: the role of the message.
    """

    role: ClassVar[str] = "error"

    def __init__(
        self,
        content: Any,
        display_text: Any = None,
        in_history: bool = False,
    ) -> None:
        """Store the error, kept out of model context by default.

        Args:
            content: the raw content of the message, consumed by the agent.
            display_text: what is shown to the user through the UI; defaults
                to ``content`` when not specified.
            in_history: whether the message is part of the model's context.
        """
        super().__init__(content, display_text, in_history)


class WarningMessage(Message):
    """Non-fatal problem surfaced to the agent consumer.

    Attributes:
        role: the role of the message.
    """

    role: ClassVar[str] = "warning"
