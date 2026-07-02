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


def apply() -> None:
    """Apply all LiteLLM runtime patches (idempotent)."""
    _ = litellm  # ensure litellm is imported before patching its internals
    _patch_mixed_reasoning_text_chunk()
