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
        A namespace mimicking a ``ModelResponseStream`` chunk. Carries a
        ``model_dump`` method so it also satisfies the real ``__next__``
        loop's end-of-chunk snapshotting, not just the isolated
        ``transform``/``ensure`` calls.
    """
    delta = SimpleNamespace(reasoning_content=reasoning, content=content)
    chunk = SimpleNamespace(id="chatcmpl-1", choices=[SimpleNamespace(delta=delta)])
    chunk.model_dump = lambda: {"id": chunk.id}
    return chunk


class _FakeIterator:
    """Minimal stand-in exposing the attributes the patch touches."""

    def __init__(self) -> None:
        """Initialize the transform-relevant iterator state."""
        self._cached_item_id = "msg_1"
        self._sequence_number = 0
        self._pending_response_events: list = []
        self.sent_annotation_events = False
        self._pending_tool_events: list = []
        self.sent_output_item_added_event = False
        self.sent_content_part_added_event = False
        self._cached_reasoning_item_id = None
        self._reasoning_item_id = None
        self._reasoning_active = False

    transform = _Iter._transform_chat_completion_chunk_to_response_api_chunk
    _transform_chat_completion_chunk_to_response_api_chunk = (
        _Iter._transform_chat_completion_chunk_to_response_api_chunk
    )
    ensure = _Iter._ensure_output_item_for_chunk
    create_content_part_added_event = _Iter.create_content_part_added_event
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
    """A chunk with both reasoning and text queues both deltas, in order.

    Returning ``None`` (rather than the text delta) matters: the sync ``__next__``
    returns a non-``None`` transform result immediately instead of queueing it, so
    returning the text delta here would ship it before the already-queued reasoning
    delta for any mixed chunk after the first. Queuing both and returning ``None`` keeps
    ordering to the shared FIFO queue every caller drains the same way.
    """
    it = _FakeIterator()

    returned = it.transform(_chunk(reasoning=".", content="Async turn one works"))

    assert returned is None
    assert len(it._pending_response_events) == 2
    reasoning_event, text_event = it._pending_response_events
    assert (
        _event_type(reasoning_event)
        == ResponsesAPIStreamEvents.REASONING_SUMMARY_TEXT_DELTA.value
    )
    assert reasoning_event.delta == "."
    assert _event_type(text_event) == ResponsesAPIStreamEvents.OUTPUT_TEXT_DELTA.value
    assert text_event.delta == "Async turn one works"


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


def test_first_text_chunk_buffers_delta() -> None:
    """The first text chunk queues its delta behind the item events."""
    it = _FakeIterator()

    it.ensure(_chunk(reasoning=None, content="Hello"))

    # output_item.added, content_part.added, then the text delta
    assert it.sent_output_item_added_event is True
    types = [_event_type(e) for e in it._pending_response_events]
    assert types == [
        ResponsesAPIStreamEvents.OUTPUT_ITEM_ADDED.value,
        ResponsesAPIStreamEvents.CONTENT_PART_ADDED.value,
        ResponsesAPIStreamEvents.OUTPUT_TEXT_DELTA.value,
    ]
    assert it._pending_response_events[-1].delta == "Hello"


def test_first_reasoning_chunk_buffers_delta() -> None:
    """The first reasoning chunk queues its delta behind the item event."""
    it = _FakeIterator()

    it.ensure(_chunk(reasoning="thinking", content=None))

    types = [_event_type(e) for e in it._pending_response_events]
    assert types == [
        ResponsesAPIStreamEvents.OUTPUT_ITEM_ADDED.value,
        ResponsesAPIStreamEvents.REASONING_SUMMARY_TEXT_DELTA.value,
    ]
    assert it._pending_response_events[-1].delta == "thinking"


def test_first_mixed_chunk_buffers_both_deltas() -> None:
    """A mixed first chunk queues its item, reasoning and text deltas."""
    it = _FakeIterator()

    it.ensure(_chunk(reasoning="R", content="T"))

    types = [_event_type(e) for e in it._pending_response_events]
    assert types == [
        ResponsesAPIStreamEvents.OUTPUT_ITEM_ADDED.value,
        ResponsesAPIStreamEvents.REASONING_SUMMARY_TEXT_DELTA.value,
        ResponsesAPIStreamEvents.OUTPUT_TEXT_DELTA.value,
    ]
    assert it._pending_response_events[1].delta == "R"
    assert it._pending_response_events[2].delta == "T"


def test_subsequent_chunk_not_rebuffered() -> None:
    """After the first item event, later chunks queue no extra delta."""
    it = _FakeIterator()
    it.sent_output_item_added_event = True

    it.ensure(_chunk(reasoning=None, content="world"))

    assert it._pending_response_events == []


class _FakeStreamWrapper:
    """Fake ``litellm.CustomStreamWrapper`` yielding fixed chunks then stopping."""

    logging_obj = None

    def __init__(self, chunks: list) -> None:
        """Store the chunks to yield in order.

        Args:
            chunks: the chat-completions chunks ``__next__`` yields in turn.
        """
        self._chunks = iter(chunks)

    def __next__(self):
        """Return the next configured chunk, then raise ``StopIteration``.

        Returns:
            The next chunk in the configured sequence.
        """
        return next(self._chunks)


def test_sync_next_orders_reasoning_before_text_for_a_later_mixed_chunk() -> None:
    """A mixed chunk after the first one still yields reasoning before text.

    Regression for the sync ``__next__`` path (the one arancio actually drives): a
    reasoning-to-text transition packed into one chunk, when that chunk isn't the very
    first of the stream, used to have its text delta returned immediately while the
    reasoning delta sat queued for the next call — reordering reasoning after the start
    of the answer.
    """
    chunks = [
        _chunk(reasoning="thinking", content=None),
        _chunk(reasoning=".", content="Sure"),
        _chunk(reasoning=None, content=", here you go"),
    ]
    iterator = _Iter(
        model="test-model",
        litellm_custom_stream_wrapper=_FakeStreamWrapper(chunks),
        request_input="hi",
        responses_api_request={},
    )

    delta_types = {
        ResponsesAPIStreamEvents.REASONING_SUMMARY_TEXT_DELTA.value,
        ResponsesAPIStreamEvents.OUTPUT_TEXT_DELTA.value,
    }
    deltas = []
    for _ in range(10):
        try:
            event = next(iterator)
        except StopIteration:
            break
        if _event_type(event) in delta_types:
            deltas.append((_event_type(event), event.delta))
        if len(deltas) == 4:
            break

    assert deltas == [
        (ResponsesAPIStreamEvents.REASONING_SUMMARY_TEXT_DELTA.value, "thinking"),
        (ResponsesAPIStreamEvents.REASONING_SUMMARY_TEXT_DELTA.value, "."),
        (ResponsesAPIStreamEvents.OUTPUT_TEXT_DELTA.value, "Sure"),
        (ResponsesAPIStreamEvents.OUTPUT_TEXT_DELTA.value, ", here you go"),
    ]


def test_first_tool_chunk_left_untouched() -> None:
    """A tool-first chunk queues nothing into the response buffer."""
    it = _FakeIterator()
    delta = SimpleNamespace(
        reasoning_content=None,
        content=None,
        tool_calls=[SimpleNamespace(index=0)],
    )
    chunk = SimpleNamespace(id="chatcmpl-1", choices=[SimpleNamespace(delta=delta)])

    it.ensure(chunk)

    # tool deltas stream through _pending_tool_events, not the response buffer
    assert it.sent_output_item_added_event is True
    assert it._pending_response_events == []
    assert it._pending_tool_events == []
