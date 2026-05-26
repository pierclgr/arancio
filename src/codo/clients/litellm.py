"""Generic LiteLLM-backed client using the Responses API."""

import warnings
from collections.abc import Iterator
from typing import Type

import litellm

from codo.builders.payload.litellm import LiteLLMPayloadBuilder
from codo.builders.request.litellm import LiteLLMRequestBuilder
from codo.clients.base import BaseClient
from codo.constants.litellm import (
    LITELLM_DEFAULT_THINKING_EFFORT,
    LITELLM_DEFAULT_THINKING_SUMMARY,
)
from codo.parsers.response.litellm import LiteLLMResponseParser
from codo.types.messages import Message
from codo.types.requests import LiteLLMRequest

# silence LiteLLM's pydantic serializer warnings when streaming Responses
# usage objects; the warning is purely informational noise from a third-
# party serialization path we do not control
warnings.filterwarnings(
    "ignore",
    message="Pydantic serializer warnings:",
    category=UserWarning,
)


class LiteLLMClient(BaseClient):
    """Generic LLM client backed by the LiteLLM Responses API.

    Routes any LiteLLM-supported provider through a single class. The
    provider is selected by the ``model_id`` prefix carried on the
    request (for example ``"openai/gpt-4o"`` or
    ``"anthropic/claude-3-5-sonnet-latest"``).

    The Responses API is preferred over chat completions because it
    natively round-trips reasoning items: ``ReasoningMessage`` objects
    can be sent back as input and reasoning summaries are returned as
    structured output items.

    Credentials are resolved entirely by LiteLLM based on the
    ``model_id`` prefix: it reads the provider-specific environment
    variable itself (``OPENAI_API_KEY``, ``ANTHROPIC_API_KEY``,
    ``GEMINI_API_KEY``, ``OPENROUTER_API_KEY``, AWS credentials for
    Bedrock, GCP credentials for Vertex, …). The user is responsible
    for exporting the env var(s) appropriate to the model(s) they
    intend to use.
    """

    _payload_builder: Type[LiteLLMPayloadBuilder] = LiteLLMPayloadBuilder
    _request_builder: Type[LiteLLMRequestBuilder] = LiteLLMRequestBuilder
    _response_parser: Type[LiteLLMResponseParser] = LiteLLMResponseParser

    _default_thinking_effort: str = LITELLM_DEFAULT_THINKING_EFFORT
    _default_thinking_summary: str = LITELLM_DEFAULT_THINKING_SUMMARY

    def __init__(
        self,
        max_output_tokens: int | None = None,
        stream: bool = False,
        *args,
        **kwargs,
    ) -> None:
        """Initialize the client with an optional output-token cap.

        Args:
            max_output_tokens: per-request cap forwarded as
                ``max_output_tokens`` to ``litellm.responses``. Useful
                with credit-gated routers (e.g. OpenRouter) which
                reserve cost for the upper bound on every call. When
                ``None`` LiteLLM/provider defaults apply.
            stream: whether to stream responses incrementally via
                Responses SSE events. When ``True`` the client forwards
                ``stream=True`` to ``litellm.responses`` and delegates
                to :meth:`LiteLLMResponseParser.parse_stream`, which
                emits both chunk and finalized messages. When ``False``
                (the default, matching LiteLLM's own default) the
                client receives a single response object and emits only
                finalized messages. Note: some providers (e.g.
                ``chatgpt/*``) return an empty ``output`` list in
                non-streaming mode and require ``stream=True`` to
                produce content.
            *args: positional arguments forwarded to :class:`BaseClient`.
            **kwargs: keyword arguments forwarded to :class:`BaseClient`.
        """
        super().__init__(*args, **kwargs)
        self._max_output_tokens = max_output_tokens
        self._stream = stream

    def __repr__(self) -> str:
        """Return a developer-friendly representation of the client.

        Returns:
            A compact string with client type and key configuration.
        """
        return (
            f"{type(self).__name__}("
            f"model_id={self._model_id!r}, "
            f"thinking_effort={self._thinking_effort!r}, "
            f"thinking_summary={self._thinking_summary!r}, "
            f"stream={self._stream!r}"
            ")"
        )

    def send_request(self, request: LiteLLMRequest) -> Iterator[Message]:
        """Send a request through LiteLLM and yield normalized messages.

        Builds the kwargs payload via ``_payload_builder``, optionally
        applies ``max_output_tokens`` and ``stream``, invokes
        ``litellm.responses`` and forwards the response to
        ``_response_parser``. When ``stream`` is enabled, the parser's
        :meth:`LiteLLMResponseParser.parse_stream` walks the SSE event
        iterator and emits both chunk and finalized messages; otherwise
        :meth:`LiteLLMResponseParser.parse` walks the single response
        object and emits only finalized messages. Credentials are
        resolved by LiteLLM from the environment based on ``model_id``.

        Args:
            request: the generic request to send.

        Yields:
            Each assistant, tool-call or reasoning message produced by
            the model, in output order. When streaming, chunk messages
            are interleaved with finalized messages.
        """
        kwargs = self._payload_builder.build(request)
        if self._max_output_tokens is not None:
            kwargs["max_output_tokens"] = self._max_output_tokens

        if self._stream:
            kwargs["stream"] = True
            stream = litellm.responses(**kwargs)
            yield from self._response_parser.parse_stream(stream)
        else:
            response = litellm.responses(**kwargs)
            yield from self._response_parser.parse(response)
