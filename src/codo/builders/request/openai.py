"""OpenAI request builder implementation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, Type

from codo.builders.request.base import BaseRequestBuilder
from codo.types.requests import OpenAIRequest

if TYPE_CHECKING:
    from codo.clients.openai import OpenAIClient


class OpenAIRequestBuilder(BaseRequestBuilder):
    """Build :class:`OpenAIRequest` instances from OpenAI client state."""

    request_class: Type[OpenAIRequest] = OpenAIRequest

    @classmethod
    def _extra_kwargs(cls, client: OpenAIClient) -> Dict[str, Any]:
        """Add the OpenAI-specific ``thinking_summary`` field.

        Args:
            client: the OpenAI client whose state populates the field.

        Returns:
            A mapping with the ``thinking_summary`` keyword argument.
        """
        return {"thinking_summary": client.thinking_summary}
