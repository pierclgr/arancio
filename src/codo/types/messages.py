"""Normalized conversation message types shared across clients."""

from dataclasses import dataclass
from typing import Any, ClassVar


@dataclass(frozen=True)
class Message:
    """Base normalized conversation message.

    Attributes:
        role: the role of the message.
        content: the content of the message.
    """

    role: ClassVar[str]
    content: str


@dataclass(frozen=True)
class UserMessage(Message):
    """Message authored by the end user.

    Attributes:
        role: the role of the message.
    """

    role: ClassVar[str] = "user"


@dataclass(frozen=True)
class AssistantMessage(Message):
    """Free-form text reply from the model.

    Attributes:
        role: the role of the message.
    """

    role: ClassVar[str] = "assistant"


@dataclass(frozen=True)
class ChunkMessage(Message):
    """Chunk of text from the model.

    Attributes:
        role: the role of the message.
    """

    role: ClassVar[str] = "chunk"


@dataclass(frozen=True)
class ErrorMessage(Message):
    """Runtime error surfaced to the agent consumer.

    Attributes:
        role: the role of the message.
    """

    role: ClassVar[str] = "error"


@dataclass(frozen=True)
class ToolCallMessage(Message):
    """Model request to invoke a tool.

    Attributes:
        role: the role of the message.
        id: provider-issued call identifier.
        name: name of the tool the model wants to invoke.
        arguments: keyword arguments for the tool, parsed from JSON.
    """

    role: ClassVar[str] = "tool_call"
    id: str
    name: str
    arguments: dict


@dataclass(frozen=True)
class ReasoningMessage(Message):
    """Model reasoning state to preserve across stateless provider requests.

    Attributes:
        role: the role of the message.
        item: provider-native reasoning item to round-trip unchanged.
    """

    role: ClassVar[str] = "reasoning"
    item: dict[str, Any]


@dataclass(frozen=True)
class AssistantChunkMessage(AssistantMessage, ChunkMessage):
    """Chunk of assistant text from the model.

    Attributes:
        role: the role of the message.
    """

    role: ClassVar[str] = "assistant_chunk"


@dataclass(frozen=True)
class ReasoningChunkMessage(ReasoningMessage, ChunkMessage):
    """Chunk of reasoning summary text from the model.

    Attributes:
        role: the role of the message.
    """

    role: ClassVar[str] = "reasoning_chunk"


@dataclass(frozen=True)
class ToolResultMessage(Message):
    """Result of executing a tool, returned to the model.

    Attributes:
        role: the role of the message.
        id: identifier of the tool call this result responds to.
        output: tool output payload (string or JSON-serializable value).
    """

    role: ClassVar[str] = "tool_result"
    id: str
    output: Any


@dataclass(frozen=True)
class ToolErrorMessage(ToolResultMessage, ErrorMessage):
    """Failed tool result surfaced as an error to the consumer.

    Attributes:
        role: the role of the message.
    """

    role: ClassVar[str] = "tool_error"
