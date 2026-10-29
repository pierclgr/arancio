"""Model reasoning messages."""

from typing import Any, ClassVar

from arancio.core.messages.base import ChunkMessage, Message


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


class ReasoningChunkMessage(ReasoningMessage, ChunkMessage):
    """Chunk of reasoning summary text from the model.

    Attributes:
        role: the role of the message.
    """

    role: ClassVar[str] = "reasoning_chunk"
