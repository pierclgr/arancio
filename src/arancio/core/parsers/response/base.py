"""Abstract response parser interface."""

from abc import ABC, abstractmethod
from collections.abc import Iterator

import requests

from arancio.core.parsers.base import Parser
from arancio.core.types.messages import Message


class BaseResponseParser(Parser, ABC):
    """Parse provider responses into normalized message streams."""

    @classmethod
    @abstractmethod
    def parse(cls, response: requests.Response) -> Iterator[Message]:
        """Parse a provider response into normalized messages.

        Concrete subclasses know the response shape (e.g. a streaming
        :class:`requests.Response` for HTTP/SSE clients, or a
        higher-level Python object for library-backed clients).

        Args:
            response: the provider response returned by the LLM client.

        Yields:
            Each normalized message extracted from the response, in the
            order it is produced.

        Raises:
            NotImplementedError: when not implemented by a subclass.
        """
        raise NotImplementedError("Subclasses must implement this method")
