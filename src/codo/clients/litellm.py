"""Generic LiteLLM-backed client using the Responses API."""

from collections.abc import Iterator
from typing import Type

import litellm

from codo.builders.payload.litellm import LiteLLMPayloadBuilder
from codo.builders.request.litellm import LiteLLMRequestBuilder
from codo.clients.base import APIClient
from codo.constants.litellm import LITELLM_DEFAULT_THINKING_EFFORT
from codo.parsers.response.litellm import LiteLLMResponseParser
from codo.types.messages import Message
from codo.types.requests import LiteLLMRequest


class LiteLLMClient(APIClient):
    """Generic LLM client backed by the LiteLLM Responses API.

    Routes any LiteLLM-supported provider through a single class. The
    provider is selected by the ``model_id`` prefix carried on the
    request (for example ``"openai/gpt-4o"`` or
    ``"anthropic/claude-3-5-sonnet-latest"``).

    The Responses API is preferred over chat completions because it
    natively round-trips reasoning items: ``ReasoningMessage`` objects
    can be sent back as input and reasoning summaries are returned as
    structured output items.

    The API key is taken from the constructor when provided, otherwise
    the environment variable named by ``env_var`` is consulted (e.g.
    ``"OPENROUTER_API_KEY"`` for OpenRouter, ``"OPENAI_API_KEY"`` for
    OpenAI). When neither yields a key, construction fails.
    """

    _payload_builder: Type[LiteLLMPayloadBuilder] = LiteLLMPayloadBuilder
    _request_builder: Type[LiteLLMRequestBuilder] = LiteLLMRequestBuilder
    _response_parser: Type[LiteLLMResponseParser] = LiteLLMResponseParser

    _default_thinking_effort: str = LITELLM_DEFAULT_THINKING_EFFORT

    def __init__(
        self,
        api_key: str | None = None,
        env_var: str = "",
        max_output_tokens: int | None = None,
        *args,
        **kwargs,
    ) -> None:
        """Initialize the client with an optional API key and output cap.

        Args:
            api_key: API key forwarded to ``litellm.responses``; when
                None, ``env_var`` is read instead.
            env_var: name of the environment variable consulted when
                ``api_key`` is not provided. Pick the variable that
                matches the provider routed by ``model_id`` (e.g.
                ``"OPENROUTER_API_KEY"`` for OpenRouter,
                ``"OPENAI_API_KEY"`` for direct OpenAI).
            max_output_tokens: per-request cap forwarded as
                ``max_output_tokens`` to ``litellm.responses``. Useful
                with credit-gated routers (e.g. OpenRouter) which
                reserve cost for the upper bound on every call. When
                ``None`` LiteLLM/provider defaults apply.
            *args: positional arguments forwarded to the next class in
                the MRO.
            **kwargs: keyword arguments forwarded to the next class in
                the MRO.
        """
        self._env_var = env_var
        super().__init__(api_key=api_key, *args, **kwargs)
        self._max_output_tokens = max_output_tokens

    def send_request(self, request: LiteLLMRequest) -> Iterator[Message]:
        """Send a request through LiteLLM and yield normalized messages.

        Builds the kwargs payload via ``_payload_builder``, augments it
        with client-level state (``api_key``, ``max_output_tokens``),
        invokes ``litellm.responses`` and forwards the response to
        ``_response_parser``.

        Args:
            request: the generic request to send.

        Yields:
            Each assistant, tool-call or reasoning message produced by
            the model, in output order.
        """
        kwargs = self._payload_builder.build(request)
        if self._max_output_tokens is not None:
            kwargs["max_output_tokens"] = self._max_output_tokens
        kwargs["api_key"] = self._token

        response = litellm.responses(**kwargs)
        yield from self._response_parser.parse(response)
