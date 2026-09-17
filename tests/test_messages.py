"""Tests for the normalized message hierarchy and its routing defaults.

The multiple-inheritance variants are the reason this file exists: which ``__init__``
wins the MRO decides whether a message reaches the model, and the answer is not the one
the class name suggests.
"""

import pytest

from arancio.core.messages import (
    AssistantChunkMessage,
    AssistantMessage,
    ChunkMessage,
    ErrorMessage,
    Message,
    ReasoningChunkMessage,
    ReasoningMessage,
    ToolCallMessage,
    ToolErrorMessage,
    ToolResultMessage,
    UserMessage,
    WarningMessage,
)


@pytest.mark.parametrize(
    "message",
    [
        UserMessage(content="x"),
        AssistantMessage(content="x"),
        WarningMessage(content="x"),
        ChunkMessage(content="x"),
        AssistantChunkMessage(content="x"),
        ReasoningMessage(content="x", item={}),
        ReasoningChunkMessage(content="x", item={}),
        ToolCallMessage(content="x", id="c1", name="ReadFileTool", arguments={}),
        ToolResultMessage(content="x", id="c1"),
        ToolErrorMessage(content="x", id="c1"),
    ],
    ids=lambda message: type(message).__name__,
)
def test_every_message_but_an_error_defaults_into_model_history(
    message: Message,
) -> None:
    """Only an error opts out of the model's context by default."""
    assert message.in_history is True


def test_an_error_stays_out_of_model_history_by_default() -> None:
    """An error reports this run, not the conversation, so the model never sees it."""
    assert ErrorMessage(content="boom").in_history is False


def test_a_failed_tool_call_still_reaches_the_model() -> None:
    """``ToolErrorMessage`` keeps the tool-result default, not the error one.

    Its bases are ``(ToolResultMessage, ErrorMessage)``, so
    ``ToolResultMessage.__init__`` wins the MRO and ``in_history`` stays True: a failed
    call is the outcome the model asked for, and hiding it would leave the model waiting
    on a tool call it never gets an answer to.
    """
    error = ToolErrorMessage(content="tool blew up", id="c1")

    assert isinstance(error, ErrorMessage)
    assert error.in_history is True


def test_an_assistant_chunk_is_also_an_assistant_message() -> None:
    """A chunk passes an ``isinstance`` check for its own finalized parent.

    This is why the session codec tests for ``ChunkMessage`` before anything
    else: reach the ``AssistantMessage`` branch with a chunk and every stream
    fragment is saved as a finished reply.
    """
    chunk = AssistantChunkMessage(content="par")

    assert isinstance(chunk, AssistantMessage)
    assert isinstance(chunk, ChunkMessage)


def test_a_reasoning_chunk_needs_the_provider_item_like_its_parent() -> None:
    """``ReasoningChunkMessage`` inherits ``ReasoningMessage.__init__``."""
    chunk = ReasoningChunkMessage(content="think", item={})

    assert isinstance(chunk, ChunkMessage)
    assert chunk.item == {}
    with pytest.raises(TypeError):
        ReasoningChunkMessage(content="think")


def test_display_text_falls_back_to_content_only_when_unset() -> None:
    """``None`` takes the content; an empty string is a deliberate choice."""
    assert AssistantMessage(content="shown").display_text == "shown"
    assert AssistantMessage(content="shown", display_text="").display_text == ""


def test_messages_compare_by_exact_type() -> None:
    """A chunk never equals the finalized message it is a fragment of."""
    assert AssistantMessage(content="x") == AssistantMessage(content="x")
    assert AssistantMessage(content="x") != AssistantChunkMessage(content="x")
    assert AssistantMessage(content="x") != AssistantMessage(content="y")
