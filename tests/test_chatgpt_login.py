"""Tests for ChatGPT device-code login notifications."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from textual.widgets import Link, Static

from arancio.core.clients import _litellm_patches
from arancio.core.clients import litellm as litellm_module
from arancio.core.clients.litellm import LiteLLMClient
from arancio.core.controllers.requests import ChatGPTLoginRequest
from arancio.core.controllers.responses import BaseControllerResponse
from arancio.ui.controller import UIController
from arancio.ui.widgets.chatgpt_login import ChatGPTLoginNotice


class _RecordingController:
    """Controller stand-in recording requests from a client."""

    def __init__(self) -> None:
        """Initialize an empty request log."""
        self.requests = []

    def request(self, request):
        """Record a request and return the login-notice acknowledgement.

        Returns:
            The acknowledgement expected by the client.
        """
        self.requests.append(request)
        return BaseControllerResponse()


def test_chatgpt_device_code_notifier_forwards_url_and_code() -> None:
    """The scoped LiteLLM notifier receives the device-code details."""
    notices = []

    with _litellm_patches.chatgpt_device_code_notifier(
        lambda url, code: notices.append((url, code))
    ):
        _litellm_patches.notify_chatgpt_device_code({"user_code": "ABCD-EFGH"})

    assert notices == [("https://auth.openai.com/codex/device", "ABCD-EFGH")]


def test_client_forwards_chatgpt_login_to_controller(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A device-code callback becomes a one-way controller request."""
    controller = _RecordingController()
    client = LiteLLMClient(controller=controller)

    def _responses(**kwargs):
        _litellm_patches.notify_chatgpt_device_code({"user_code": "ABCD-EFGH"})
        return SimpleNamespace(output=[])

    monkeypatch.setattr(
        litellm_module.litellm,
        "responses",
        Mock(side_effect=_responses),
    )
    list(
        client.send_request(
            litellm_module.LiteLLMRequest(
                model_id="openai/gpt-4o",
                message_list=[],
            )
        )
    )

    assert len(controller.requests) == 1
    request = controller.requests[0]
    assert isinstance(request, ChatGPTLoginRequest)
    assert request.verification_url == "https://auth.openai.com/codex/device"
    assert request.user_code == "ABCD-EFGH"


def test_ui_controller_displays_chatgpt_login_without_waiting() -> None:
    """The UI controller schedules the login notice and returns immediately."""
    app = Mock()
    controller = UIController()
    controller.app = app
    request = ChatGPTLoginRequest(
        verification_url="https://example.com/device",
        user_code="ABCD-EFGH",
    )

    response = controller.request(request)

    assert type(response) is BaseControllerResponse
    app.call_from_thread.assert_called_once_with(
        app.show_chatgpt_login,
        "https://example.com/device",
        "ABCD-EFGH",
    )


def test_chatgpt_login_notice_shows_a_clickable_copyable_url_and_code() -> None:
    """The login notice includes a clickable URL and visible device code."""
    notice = ChatGPTLoginNotice(
        verification_url="https://example.com/device",
        user_code="ABCD-EFGH",
    )

    children = list(notice.compose())
    link = next(child for child in children if isinstance(child, Link))
    code = next(
        child
        for child in children
        if isinstance(child, Static) and "ABCD-EFGH" in str(child.render())
    )

    assert link.url == "https://example.com/device"
    assert str(link.render()) == "https://example.com/device"
    assert str(code.render()) == "Code: ABCD-EFGH"
