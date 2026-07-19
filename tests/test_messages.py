"""Tests for normalized message types."""

from arancio.core.messages import UserMessage, WarningMessage


def test_user_message_display_text_defaults_to_content() -> None:
    """Omitting display_text sets it to the message content."""
    message = UserMessage(content="hello")

    assert message.display_text == "hello"


def test_user_message_display_text_can_be_set_explicitly() -> None:
    """An explicit display_text is kept as given."""
    message = UserMessage(content="hello", display_text={"k": "v"})

    assert message.display_text == {"k": "v"}


def test_warning_message_role_and_display_text_default() -> None:
    """A WarningMessage has the ``warning`` role and defaults display_text."""
    message = WarningMessage(content="careful")

    assert message.role == "warning"
    assert message.display_text == "careful"
