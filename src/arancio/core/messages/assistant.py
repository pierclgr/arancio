"""Assistant conversation messages."""

from typing import ClassVar

from arancio.core.messages.base import ChunkMessage, Message


class AssistantMessage(Message):
    """Free-form text reply from the model.

    Attributes:
        role: the role of the message.
    """

    role: ClassVar[str] = "assistant"


class AssistantChunkMessage(AssistantMessage, ChunkMessage):
    """Chunk of assistant text from the model.

    Attributes:
        role: the role of the message.
    """

    role: ClassVar[str] = "assistant_chunk"
