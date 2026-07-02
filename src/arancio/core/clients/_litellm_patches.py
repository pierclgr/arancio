"""Runtime patches for bugs in the pinned LiteLLM version.

Temporary monkeypatches applied at import time by
:mod:`arancio.core.clients.litellm`. Each patch is written to be a no-op
once the corresponding upstream bug is fixed, so bumping LiteLLM and then
deleting this module is a safe two-step cleanup.
"""

import litellm
from litellm.responses.litellm_completion_transformation.streaming_iterator import (
    LiteLLMCompletionStreamingIterator,
)
from litellm.types.llms.openai import (
    OutputTextDeltaEvent,
    ReasoningSummaryTextDeltaEvent,
    ResponsesAPIStreamEvents,
)

# marker so the patch is applied at most once per process
_MIXED_CHUNK_PATCH_FLAG = "_arancio_mixed_chunk_patched"

# marker so the patch is applied at most once per process
_FIRST_CHUNK_PATCH_FLAG = "_arancio_first_chunk_patched"


def _delta_has_text_and_reasoning(chunk) -> bool:
    """Return whether a chat chunk carries both reasoning and text.

    Args:
        chunk: a ``ModelResponseStream`` chunk from the chat stream.

    Returns:
        ``True`` when the first choice's delta has non-empty
        ``reasoning_content`` and non-empty ``content``.
    """
    if not getattr(chunk, "choices", None):
        return False
    delta = chunk.choices[0].delta
    reasoning = getattr(delta, "reasoning_content", None)
    content = getattr(delta, "content", None)
    return bool(reasoning) and bool(content)


def _patch_mixed_reasoning_text_chunk() -> None:
    """Stop LiteLLM's Responses bridge dropping text from mixed chunks.

    LiteLLM's chat-to-Responses bridge
    (``LiteLLMCompletionStreamingIterator``) transforms each chat chunk in
    ``_transform_chat_completion_chunk_to_response_api_chunk``. When a
    single chunk carries both ``reasoning_content`` and ``content`` (some
    providers, e.g. ``ollama_chat/*`` models, pack the reasoning-to-answer
    transition into one chunk), the method returns the reasoning delta and
    never emits the text delta, so the start of the answer is silently
    lost. This wraps the method: for a mixed chunk it queues the reasoning
    delta into the iterator's pending buffer and returns the text delta, so
    both reach the consumer in order. Non-mixed chunks fall through to the
    original implementation unchanged.

    The wrapper is idempotent and self-disabling: if a future LiteLLM
    version stops producing mixed chunks (or fixes the transform), the
    ``_delta_has_text_and_reasoning`` guard is simply never true and the
    original method runs for every chunk.
    """
    cls = LiteLLMCompletionStreamingIterator
    if getattr(cls, _MIXED_CHUNK_PATCH_FLAG, False):
        return

    original = cls._transform_chat_completion_chunk_to_response_api_chunk

    def _patched(self, chunk):
        """Emit both reasoning and text events for a mixed chat chunk.

        Args:
            self: the streaming-iterator instance.
            chunk: the chat-completions chunk being transformed.

        Returns:
            The text delta event for a mixed chunk (with the reasoning
            delta queued ahead of the caller's own append), otherwise the
            original method's result.
        """
        if not _delta_has_text_and_reasoning(chunk):
            return original(self, chunk)

        delta = chunk.choices[0].delta
        reasoning_delta = ReasoningSummaryTextDeltaEvent(
            type=ResponsesAPIStreamEvents.REASONING_SUMMARY_TEXT_DELTA,
            item_id=f"rs_{hash(str(delta.reasoning_content))}",
            output_index=0,
            delta=delta.reasoning_content,
        )
        # the caller appends our return value to the end of
        # _pending_response_events and pops from the front, so queue the
        # reasoning delta first to keep reasoning-before-text ordering
        self._pending_response_events.append(reasoning_delta)

        self._sequence_number += 1
        text_delta = OutputTextDeltaEvent(
            type=ResponsesAPIStreamEvents.OUTPUT_TEXT_DELTA,
            item_id=self._cached_item_id or chunk.id,
            output_index=0,
            content_index=0,
            delta=delta.content,
        )
        text_delta.__dict__["sequence_number"] = self._sequence_number
        return text_delta

    cls._transform_chat_completion_chunk_to_response_api_chunk = _patched
    setattr(cls, _MIXED_CHUNK_PATCH_FLAG, True)


def _patch_sync_first_chunk_drop() -> None:
    """Stop LiteLLM's sync Responses iterator dropping the first chunk.

    LiteLLM's chat-to-Responses bridge
    (``LiteLLMCompletionStreamingIterator``) drives the synchronous
    ``litellm.responses`` path through ``__next__``. On the first
    content-bearing chunk, ``_ensure_output_item_for_chunk`` queues an
    ``output_item.added`` (and, for messages, ``content_part.added``)
    event, and ``__next__`` returns that queued event *before* transforming
    the chunk into a delta, discarding the chunk without ever emitting its
    text or reasoning; the start of the streamed answer is silently lost.
    The async ``__anext__`` does not have this bug: it transforms the chunk
    and appends the delta to the pending buffer before draining it.

    This wraps ``_ensure_output_item_for_chunk`` to mirror the async
    ordering: on the first-content-chunk transition it also transforms that
    chunk and appends its delta to ``_pending_response_events``, so the
    delta survives the queued ``output_item.added`` and reaches the
    consumer. It acts only for text/reasoning chunks; tool-call chunks are
    left untouched (their deltas stream through ``_pending_tool_events``,
    not ``_pending_response_events``, so appending would misroute them).
    Later chunks hit the ``sent_output_item_added_event`` guard and are
    left to the original transform, so nothing is double-emitted.

    Unlike the mixed-chunk patch there is no per-chunk guard that goes
    quiet once upstream fixes the bug, so re-verify and delete this patch
    when upgrading LiteLLM: against a fixed iterator it would double-emit
    the first delta.
    """
    cls = LiteLLMCompletionStreamingIterator
    if getattr(cls, _FIRST_CHUNK_PATCH_FLAG, False):
        return

    original = cls._ensure_output_item_for_chunk

    def _patched(self, chunk) -> None:
        """Buffer the first content chunk's delta before it is dropped.

        Args:
            self: the streaming-iterator instance.
            chunk: the chat-completions chunk being processed.
        """
        was_sent = self.sent_output_item_added_event
        original(self, chunk)
        # only act on the first-content-chunk transition
        if was_sent or not self.sent_output_item_added_event:
            return
        delta = chunk.choices[0].delta if getattr(chunk, "choices", None) else None
        has_text = bool(getattr(delta, "content", None))
        has_reasoning = bool(getattr(delta, "reasoning_content", None))
        # leave tool-first chunks to the original transform/tool queue
        if not (has_text or has_reasoning):
            return
        delta_event = self._transform_chat_completion_chunk_to_response_api_chunk(chunk)
        if delta_event:
            self._pending_response_events.append(delta_event)

    cls._ensure_output_item_for_chunk = _patched
    setattr(cls, _FIRST_CHUNK_PATCH_FLAG, True)


def apply() -> None:
    """Apply all LiteLLM runtime patches (idempotent)."""
    _ = litellm  # ensure litellm is imported before patching its internals
    _patch_mixed_reasoning_text_chunk()
    _patch_sync_first_chunk_drop()
