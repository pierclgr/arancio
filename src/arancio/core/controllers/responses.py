"""Responses a frontend returns for a controller request."""

from enum import Enum


class BaseControllerResponse:
    """Base class for responses returned for a controller request."""


class Decision(Enum):
    """User's decision on whether a tool call may run."""

    ALLOW = "allow"
    DENY = "deny"


class PermissionResponse(BaseControllerResponse):
    """Response carrying the user's decision on a tool call.

    Attributes:
        decision: whether the call may run.
        message: optional text passed to the model — a note when allowing, a
            reason when denying.
    """

    def __init__(self, decision: Decision, message: str | None = None) -> None:
        """Store the decision and its optional accompanying message.

        Args:
            decision: whether the call may run.
            message: optional text passed to the model — a note when allowing,
                a reason when denying.
        """
        self.decision = decision
        self.message = message
