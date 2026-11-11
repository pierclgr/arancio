"""Provider-agnostic base class for LLM clients."""

from abc import ABC, abstractmethod
from collections.abc import Callable, Generator
from typing import List, Type

from arancio.core.builders.payload.base import PayloadBuilder
from arancio.core.builders.request.base import RequestBuilder
from arancio.core.clients.types import Reasoning
from arancio.core.events import LoginNotice
from arancio.core.messages.base import Message
from arancio.core.parsers.responses.base import ResponseParser
from arancio.core.requests.base import Request
from arancio.core.tools.schema import ToolSchema


class Client(ABC):
    """Provider-agnostic base for LLM clients.

    Subclasses pick a transport (library call, HTTP, …) by overriding
    :meth:`send_request`. Request building is generic and driven by the strategy classes
    ``_request_builder`` / ``_payload_builder`` / ``_response_parser`` declared as class
    attributes; concrete clients override them to plug in provider-specific behavior.

    Most provider integrations should reuse :class:`LiteLLMClient` and route through
    LiteLLM; subclass this base directly only when a provider is not reachable through
    LiteLLM and requires a bespoke transport (e.g. a custom subscription endpoint that
    LiteLLM does not yet support).
    """

    _payload_builder: Type[PayloadBuilder] = PayloadBuilder
    _request_builder: Type[RequestBuilder] = RequestBuilder
    _response_parser: Type[ResponseParser] = ResponseParser

    _model_options: List[str] | None = None

    _default_model_id: str | None = None
    _default_reasoning: Reasoning = Reasoning()

    def __init__(
        self,
        on_login: Callable[[LoginNotice], None],
        model_id: str | None = None,
        reasoning: Reasoning | None = None,
    ) -> None:
        """Initialize the client with its login callback and per-instance overrides.

        Credentials are not passed here: they are resolved by the underlying
        transport (e.g. LiteLLM's native env-var lookup).

        Args:
            on_login: one-way callback receiving the login details while the
                provider waits for a sign-in. Required, so a caller cannot
                forget it and hide a login code; a client without a login flow
                never calls it.
            model_id: the model identifier to use for outgoing requests;
                when ``None`` the class default ``_default_model_id`` is
                kept.
            reasoning: the reasoning effort and summary level to use for
                outgoing requests; when ``None`` the class default
                ``_default_reasoning`` is kept.
        """
        self._on_login: Callable[[LoginNotice], None] = on_login
        self._model_id: str | None = self._default_model_id
        self._reasoning: Reasoning = self._default_reasoning
        if model_id is not None:
            self.model_id = model_id
        if reasoning is not None:
            self.reasoning = reasoning

    def __repr__(self) -> str:
        """Return a developer-friendly representation of the client.

        Returns:
            A compact string with client type and key configuration.
        """
        return (
            f"{type(self).__name__}("
            f"model_id={self._model_id!r}, "
            f"reasoning={self._reasoning!r}"
            ")"
        )

    @property
    def model_id(self) -> str | None:
        """The model ID used for outgoing requests.

        Returns:
            The current model id, or ``None`` when no default exists and
            none has been set.
        """
        return self._model_id

    @model_id.setter
    def model_id(self, value: str) -> None:
        """Set the model id, validating against ``_model_options`` when set.

        Args:
            value: the model id to use for subsequent requests.

        Raises:
            ValueError: when ``_model_options`` is set and ``value`` is
                not in it.
        """
        if self._model_options and value not in self._model_options:
            raise ValueError(
                f"Invalid model identifier: {value}. Choose between "
                f"{self._model_options}"
            )
        self._model_id = value

    @staticmethod
    def get_model_id(provider: str | None, name: str | None) -> str | None:
        """Join a provider prefix and a model name into this client's model id.

        Subclasses override this when their provider expects another format.

        Args:
            provider: the provider prefix (e.g. ``"openai"``), or ``None`` when unset.
            name: the model name without the provider prefix, or ``None`` when unset.

        Returns:
            The ``provider/name`` model id, or ``None`` when either part is unset.
        """
        if not provider or not name:
            return None
        return f"{provider}/{name}"

    @property
    def reasoning(self) -> Reasoning:
        """The reasoning effort and summary level used for outgoing requests.

        Returns:
            The current reasoning options; a field is ``None`` when not set.
        """
        return self._reasoning

    @reasoning.setter
    def reasoning(self, value: Reasoning) -> None:
        """Set the reasoning options used for subsequent requests.

        Args:
            value: the reasoning effort tier (e.g. ``"low"``) and summary level
                (e.g. ``"auto"``, ``"concise"``, ``"detailed"``).
        """
        self._reasoning = value

    def build_request(
        self,
        messages: List[Message],
        system_prompt: str = "",
        tools: List[ToolSchema] | None = None,
    ) -> Request:
        """Build a normalized request bound to the client's current state.

        Delegates to the client's :class:`RequestBuilder` subclass so
        provider-specific fields are populated without callers needing to
        know which fields the underlying request type carries.

        Args:
            messages: the message list to send.
            system_prompt: the system prompt string.
            tools: the tool catalog to expose to the model; when None
                an empty list is used.

        Returns:
            A normalized :class:`Request` ready for ``send_request``.
        """
        return self._request_builder.build(
            client=self,
            messages=messages,
            system_prompt=system_prompt,
            tools=tools,
        )

    @abstractmethod
    def send_request(self, request: Request) -> Generator[Message, None, None]:
        """Send a request through the underlying transport.

        Subclasses implement the actual network or library call and
        return a generator of normalized messages parsed from the
        provider response. A subclass with a login flow passes the login
        details to ``self._on_login`` instead of yielding them.

        Args:
            request: the generic request to send.

        Yields:
            Each normalized message produced by the provider, in
            output order.
        """
        raise NotImplementedError("Subclasses must implement this method")
