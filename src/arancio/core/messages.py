"""Normalized conversation message types shared across clients."""

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


class UserMessage(Message):
    """Message authored by the end user.

    Attributes:
        role: the role of the message.
    """

    role: ClassVar[str] = "user"


class AssistantMessage(Message):
    """Free-form text reply from the model.

    Attributes:
        role: the role of the message.
    """

    role: ClassVar[str] = "assistant"


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


class ReasoningMessage(Message):
    """Model reasoning state to preserve across stateless provider requests.

    Attributes:
        role: the role of the message.
        item: provider-native reasoning item to round-trip unchanged.
    """

    role: ClassVar[str] = "reasoning"

    def __init__(
        self,
        content: Any,
        item: dict[str, Any],
        display_text: Any = None,
        in_history: bool = True,
    ) -> None:
        """Store the provider-native reasoning item alongside base fields.

        Args:
            content: the raw content of the message, consumed by the agent.
            item: provider-native reasoning item to round-trip unchanged.
            display_text: what is shown to the user through the UI; defaults
                to ``content`` when not specified.
            in_history: whether the message is part of the model's context.
        """
        super().__init__(content, display_text, in_history)
        self.item = item


class AssistantChunkMessage(AssistantMessage, ChunkMessage):
    """Chunk of assistant text from the model.

    Attributes:
        role: the role of the message.
    """

    role: ClassVar[str] = "assistant_chunk"


class ReasoningChunkMessage(ReasoningMessage, ChunkMessage):
    """Chunk of reasoning summary text from the model.

    Attributes:
        role: the role of the message.
    """

    role: ClassVar[str] = "reasoning_chunk"


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
