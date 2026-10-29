"""User conversation messages."""

from typing import ClassVar

from arancio.core.messages.base import Message


class UserMessage(Message):
    """Message authored by the end user.

    Attributes:
        role: the role of the message.
    """

    role: ClassVar[str] = "user"
