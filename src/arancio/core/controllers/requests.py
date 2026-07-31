"""Requests core sends to a frontend through a controller.

Each request kind is a :class:`BaseControllerRequest` subclass; the kind is the subclass
itself, dispatched with ``isinstance`` by the controller.
"""

from arancio.core.messages import ToolCallMessage


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


class ChatGPTLoginRequest(BaseControllerRequest):
    """Request displaying a ChatGPT device-code login in the frontend.

    Attributes:
        verification_url: browser address where the user enters the device code.
        user_code: short code authorizing the current device-login attempt.
    """

    def __init__(self, verification_url: str, user_code: str) -> None:
        """Store the ChatGPT device-login details.

        Args:
            verification_url: browser address where the user enters the code.
            user_code: short code issued for the current login attempt.
        """
        self.verification_url = verification_url
        self.user_code = user_code
