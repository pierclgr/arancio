"""Interaction data exchanged between core and the caller driving it.

Core runs as a generator: it yields messages and permission requests, and the caller
answers a permission request with ``.send(response)``. A login notice needs no answer,
so a client passes it to its one-way login callback instead of yielding it.
"""

from collections.abc import Generator
from dataclasses import dataclass, field
from enum import Enum
from uuid import uuid4

from arancio.core.messages.base import Message
from arancio.core.messages.tool import ToolCallMessage


@dataclass(frozen=True)
class PermissionRequest:
    """Request asking the caller to approve or deny a tool call.

    Attributes:
        call: the tool call awaiting a decision.
        id: unique identifier the matching response must carry.
    """

    call: ToolCallMessage
    id: str = field(default_factory=lambda: uuid4().hex)


class Decision(Enum):
    """User's decision on whether a tool call may run."""

    ALLOW = "allow"
    DENY = "deny"


@dataclass(frozen=True)
class PermissionResponse:
    """Answer to a permission request.

    Attributes:
        request_id: identifier of the request this response answers.
        decision: whether the call may run.
        message: optional text passed to the model, a note when allowing and a
            reason when denying.
    """

    request_id: str
    decision: Decision
    message: str | None = None


@dataclass(frozen=True)
class LoginNotice:
    """Login details a client shows while the provider waits for a sign-in.

    Attributes:
        provider: display name of the provider asking for the sign-in.
        verification_url: browser address where the user enters the code.
        user_code: short code issued for the current login attempt.
    """

    provider: str
    verification_url: str
    user_code: str


CoreEvent = Message | PermissionRequest
CoreAnswer = PermissionResponse | None
Execution = Generator[CoreEvent, CoreAnswer, None]
