"""Tests for the UI adapter's decisions, without booting a Textual app.

Two things here are ours rather than Textual's: the controller, which blocks a worker
thread until the user answers, and the mapping from a message type to the style it is
shown with. Both are testable without a running app.
"""

import pytest
from fakes import FakeApp

from arancio.core.controllers.requests import (
    BaseControllerRequest,
    ChatGPTLoginRequest,
    PermissionRequest,
)
from arancio.core.controllers.responses import Decision, PermissionResponse
from arancio.core.messages import (
    AssistantMessage,
    ErrorMessage,
    Message,
    ReasoningMessage,
    ToolCallMessage,
    ToolErrorMessage,
    ToolResultMessage,
    UserMessage,
    WarningMessage,
)
from arancio.ui.app import App
from arancio.ui.controller import UIController
from arancio.ui.widgets.question import QuestionScreen


def _permission_request() -> PermissionRequest:
    """Build a permission request for a shell call.

    Returns:
        A request naming the call the user must approve.
    """
    return PermissionRequest(
        ToolCallMessage(
            content="",
            id="c1",
            name="ShellCommandTool",
            arguments={"command": "ls"},
        )
    )


def test_dispatching_before_the_app_exists_fails_loudly() -> None:
    """The window is real: the app needs the agent, which needs this controller.

    It closes during wiring and never reopens, so an unattached dispatch is a wiring bug
    and says so instead of raising ``AttributeError`` somewhere else.
    """
    controller = UIController()

    assert controller.app is None
    with pytest.raises(ValueError, match="No app attached"):
        controller.require_app()


def test_the_attached_app_is_returned_once_wiring_finished() -> None:
    """After ``main`` assigns it, every dispatch goes through this accessor."""
    controller = UIController()
    app = FakeApp()
    controller.app = app

    assert controller.require_app() is app


def test_a_permission_request_shows_the_call_being_approved() -> None:
    """The user has to see the arguments, not just the tool name."""
    controller = UIController()
    controller.app = FakeApp(answer="yes")

    controller.request(_permission_request())

    screen = controller.app.screens[0]
    assert isinstance(screen, QuestionScreen)


@pytest.mark.parametrize(
    ("answer", "decision", "message"),
    [
        ("yes", Decision.ALLOW, None),
        ("no", Decision.DENY, None),
        ("not that file", Decision.DENY, "not that file"),
    ],
    ids=["allow", "deny", "deny-with-reason"],
)
def test_each_answer_maps_to_a_decision(
    answer: str, decision: Decision, message: str | None
) -> None:
    """Anything the user types instead of yes or no is a denial with a reason."""
    controller = UIController()
    controller.app = FakeApp(answer=answer)

    response = controller.request(_permission_request())

    assert isinstance(response, PermissionResponse)
    assert response.decision is decision
    assert response.message == message


def test_a_login_notice_is_shown_and_answered_immediately() -> None:
    """Sign-in details are UI-only, so nothing waits and nothing is returned."""
    controller = UIController()
    app = FakeApp()
    controller.app = app

    response = controller.request(
        ChatGPTLoginRequest(verification_url="https://verify", user_code="ABCD")
    )

    assert app.logins == [("https://verify", "ABCD")]
    assert type(response) is not PermissionResponse


def test_an_unsupported_request_is_refused() -> None:
    """Dispatch is by type, so a new request cannot be silently dropped."""
    controller = UIController()
    controller.app = FakeApp()

    with pytest.raises(TypeError):
        controller.request(BaseControllerRequest())


@pytest.mark.parametrize(
    ("message", "expected"),
    [
        (ErrorMessage(content="boom"), "error"),
        (ToolErrorMessage(content="boom", id="c1"), "error"),
        (WarningMessage(content="careful"), "warning"),
        (ToolResultMessage(content="out", id="c1"), "tool-result"),
        (UserMessage(content="hi"), "user"),
        (ReasoningMessage(content="think", item={}), "reasoning"),
    ],
    ids=lambda value: value if isinstance(value, str) else type(value).__name__,
)
def test_each_message_is_styled_by_what_it_is(message: Message, expected: str) -> None:
    """A failed tool call must read as an error, not as an ordinary result.

    ``ToolErrorMessage`` subclasses ``ToolResultMessage``, so the order of the checks is
    what keeps it styled as a failure.
    """
    widget = App._render(message)

    assert expected in widget.classes


def test_assistant_text_is_rendered_as_markdown() -> None:
    """Replies are formatted, unlike the single-line statuses around them."""
    widget = App._render(AssistantMessage(content="**bold**"))

    assert widget.classes == frozenset()


def test_a_tool_call_is_shown_with_its_arguments() -> None:
    """The user decides on a call by reading what it would do."""
    widget = App._render(
        ToolCallMessage(
            content="", id="c1", name="ReadFileTool", arguments={"file_path": "/a"}
        )
    )

    assert "tool-call" in widget.classes
