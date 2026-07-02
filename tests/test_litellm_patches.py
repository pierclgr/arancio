"""Tests for the LiteLLM runtime patches."""

from types import SimpleNamespace

from litellm.responses.litellm_completion_transformation.streaming_iterator import (
    LiteLLMCompletionStreamingIterator as _Iter,
)
from litellm.types.llms.openai import ResponsesAPIStreamEvents

# importing the client applies the patches at module import time
import arancio.core.clients.litellm  # noqa: F401


def _chunk(reasoning: str | None, content: str | None) -> SimpleNamespace:
    """Build a fake chat-completions chunk with the given delta fields.

    Args:
        reasoning: the ``reasoning_content`` delta, or ``None``.
        content: the ``content`` delta, or ``None``.

    Returns:
        A namespace mimicking a ``ModelResponseStream`` chunk.
    """
    delta = SimpleNamespace(reasoning_content=reasoning, content=content)
    return SimpleNamespace(id="chatcmpl-1", choices=[SimpleNamespace(delta=delta)])


class _FakeIterator:
    """Minimal stand-in exposing the attributes the patch touches."""

    def __init__(self) -> None:
        """Initialize the transform-relevant iterator state."""
        self._cached_item_id = "msg_1"
        self._sequence_number = 0
        self._pending_response_events: list = []
        self.sent_annotation_events = False
        self._pending_tool_events: list = []

    transform = _Iter._transform_chat_completion_chunk_to_response_api_chunk
    _get_delta_string_from_streaming_choices = (
        _Iter._get_delta_string_from_streaming_choices
    )


def _event_type(event: object) -> str | None:
    """Return an event's dotted type string, unwrapping enum values.

    Args:
        event: a Responses streaming event.

    Returns:
        The event type string, or ``None``.
    """
    value = getattr(event, "type", None)
    return getattr(value, "value", value)


def test_mixed_chunk_emits_both_reasoning_and_text() -> None:
    """A chunk with both reasoning and text yields both deltas in order."""
    it = _FakeIterator()

    returned = it.transform(_chunk(reasoning=".", content="Async turn one works"))

    # the text delta is returned; the reasoning delta is queued ahead of it
    assert _event_type(returned) == ResponsesAPIStreamEvents.OUTPUT_TEXT_DELTA.value
    assert returned.delta == "Async turn one works"
    assert len(it._pending_response_events) == 1
    queued = it._pending_response_events[0]
    assert (
        _event_type(queued)
        == ResponsesAPIStreamEvents.REASONING_SUMMARY_TEXT_DELTA.value
    )
    assert queued.delta == "."


def test_reasoning_only_chunk_unchanged() -> None:
    """A reasoning-only chunk still yields a single reasoning delta."""
    it = _FakeIterator()

    returned = it.transform(_chunk(reasoning="thinking", content=None))

    assert (
        _event_type(returned)
        == ResponsesAPIStreamEvents.REASONING_SUMMARY_TEXT_DELTA.value
    )
    assert returned.delta == "thinking"
    assert it._pending_response_events == []


def test_text_only_chunk_unchanged() -> None:
    """A text-only chunk still yields a single text delta."""
    it = _FakeIterator()

    returned = it.transform(_chunk(reasoning=None, content="hello"))

    assert _event_type(returned) == ResponsesAPIStreamEvents.OUTPUT_TEXT_DELTA.value
    assert returned.delta == "hello"
    assert it._pending_response_events == []
