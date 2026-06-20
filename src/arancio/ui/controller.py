"""Textual adapter implementing the core :class:`Controller` port.

This is the UI side of the core/UI seam: it receives requests core emits and drives the
Textual app to answer them, blocking the worker thread (never the UI thread) until the
user responds.
"""

import threading

from arancio.core.controllers.base import Controller
from arancio.core.controllers.requests import BaseControllerRequest, PermissionRequest
from arancio.core.controllers.responses import (
    BaseControllerResponse,
    Decision,
    PermissionResponse,
)
from arancio.ui.app import App
from arancio.ui.widgets.question import QuestionScreen


class UIController(Controller):
    """Bridge core requests to the Textual app and block for the answer.

    The app reference is set after construction: the app needs the agent,
    which needs the permission manager, which needs this controller, so the
    controller is built first and ``app`` is assigned once the app exists.

    Attributes:
        app: the running Textual app requests are dispatched to.
    """

    def __init__(self) -> None:
        """Initialize the controller with no app attached yet."""
        self.app: App | None = None

    def request(self, request: BaseControllerRequest) -> BaseControllerResponse:
        """Dispatch a request to the app and block until the user answers.

        Args:
            request: the request to dispatch.

        Returns:
            The response produced by the user through the app.

        Raises:
            TypeError: when the request kind is not supported.
        """
        if isinstance(request, PermissionRequest):
            answer = self._ask(
                f"Allow {request.call.name} with arguments {request.call.arguments}?",
                ["yes", "no"],
            )
            return self._to_permission_response(answer)
        raise TypeError(f"Unsupported request type: {type(request).__name__}")

    def _ask(self, question: str, answers: list[str]) -> str:
        """Show a question screen and block the worker thread for the answer.

        Args:
            question: the question shown to the user.
            answers: the preset answers offered as buttons.

        Returns:
            The chosen answer label or the typed custom text.
        """
        event = threading.Event()
        result: list[str] = []

        def _store(answer: str) -> None:
            result.append(answer)
            event.set()

        self.app.call_from_thread(
            self.app.push_screen, QuestionScreen(question, answers), _store
        )
        event.wait()
        return result[0]

    @staticmethod
    def _to_permission_response(answer: str) -> PermissionResponse:
        """Map a raw answer to a permission response.

        Args:
            answer: ``"yes"``, ``"no"`` or custom free-text.

        Returns:
            ALLOW for ``"yes"``; DENY for ``"no"``; DENY with the text as the
            reason for any custom answer.
        """
        if answer == "yes":
            return PermissionResponse(decision=Decision.ALLOW)
        if answer == "no":
            return PermissionResponse(decision=Decision.DENY)
        return PermissionResponse(decision=Decision.DENY, message=answer)
