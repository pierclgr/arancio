"""Port through which core reaches out to a frontend and waits for a reply.

This is the seam decoupling the agent core from any frontend (a TUI, a
console, a remote client). Core emits a :class:`BaseControllerRequest` through
a :class:`Controller` and blocks for the matching
:class:`BaseControllerResponse`; concrete controllers live outside core and
implement the transport.
"""

from abc import ABC, abstractmethod

from arancio.core.controllers.requests import BaseControllerRequest
from arancio.core.controllers.responses import BaseControllerResponse


class Controller(ABC):
    """Port through which core sends a request and blocks for its response."""

    @abstractmethod
    def request(self, request: BaseControllerRequest) -> BaseControllerResponse:
        """Send a request to the frontend and block until its response.

        Args:
            request: the request to dispatch.

        Returns:
            The response the frontend produced for the request.
        """
