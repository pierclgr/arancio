"""Requests core sends to a frontend through a controller.

Each request kind is a :class:`BaseControllerRequest` subclass; the kind is the subclass
itself, dispatched with ``isinstance`` by the controller.
"""

from arancio.core.types.messages import ToolCallMessage


class BaseControllerRequest:
    """Base class for requests core sends through a controller."""


class PermissionRequest(BaseControllerRequest):
    """Request asking the user to approve or deny a tool call.

    Attributes:
        call: the tool call awaiting a decision.
    """

    def __init__(self, call: ToolCallMessage) -> None:
        """Store the tool call awaiting a decision.

        Args:
            call: the tool call awaiting a decision.
        """
        self.call = call
