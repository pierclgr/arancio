"""Abstract request builder interface."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Type

from codo.builders.base import Builder
from codo.types.messages import Message
from codo.types.requests import BaseRequest
from codo.types.tools import ToolSchema

if TYPE_CHECKING:
    from codo.clients.base import BaseClient


class BaseRequestBuilder(Builder):
    """Build a normalized client request from client state and caller args."""

    request_class: Type[BaseRequest] = BaseRequest

    @classmethod
    def build(
        cls,
        client: BaseClient,
        messages: List[Message],
        system_prompt: str = "",
        tools: List[ToolSchema] | None = None,
    ) -> BaseRequest:
        """Build a normalized request bound to the client's current state.

        Reads the model id and thinking effort from the client so callers
        do not need to pass them explicitly. Subclasses contribute
        provider-specific fields by overriding :meth:`_extra_kwargs`.

        Args:
            client: the client whose state populates request fields.
            messages: the message list to send.
            system_prompt: the system prompt string.
            tools: the tool catalog to expose to the model; when None
                an empty list is used.

        Returns:
            A normalized request of the type declared on ``request_class``.
        """
        kwargs: Dict[str, Any] = {
            "model_id": client.model_id,
            "thinking_effort": client.thinking_effort,
            "system_prompt": system_prompt,
            "tool_list": tools or [],
            "message_list": messages,
        }
        kwargs.update(cls._extra_kwargs(client))
        return cls.request_class(**kwargs)

    @classmethod
    def _extra_kwargs(cls, client: BaseClient) -> Dict[str, Any]:
        """Return provider-specific extra kwargs for the request constructor.

        Override in subclasses to contribute fields beyond the base set.
        The default returns an empty mapping.

        Args:
            client: the client whose state may carry extra fields.

        Returns:
            A mapping of extra keyword arguments to pass to
            ``request_class``.
        """
        return {}
