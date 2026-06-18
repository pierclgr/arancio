"""Tests for normalized message types."""

from arancio.core.types.messages import UserMessage


def test_user_message_display_text_defaults_to_content() -> None:
    """Omitting display_text sets it to the message content."""
    message = UserMessage(content="hello")

    assert message.display_text == "hello"


def test_user_message_display_text_can_be_set_explicitly() -> None:
    """An explicit display_text is kept as given."""
    message = UserMessage(content="hello", display_text={"k": "v"})

    assert message.display_text == {"k": "v"}
