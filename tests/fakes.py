"""Hand-written doubles for the boundaries a test cannot cross for real.

Only three things in this repository are genuinely external: the model behind
:class:`~arancio.core.clients.base.BaseClient`, the user behind
:class:`~arancio.core.controllers.base.Controller`, and the Textual app behind
:class:`~arancio.ui.controller.UIController`. Everything else is exercised as
the real object against ``tmp_path``, so every double lives in this one module
rather than spreading through the suite.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Iterator, List, Tuple

from arancio.core.clients.base import BaseClient
from arancio.core.controllers.base import Controller
from arancio.core.controllers.requests import BaseControllerRequest, PermissionRequest
from arancio.core.controllers.responses import (
    BaseControllerResponse,
    Decision,
    PermissionResponse,
)
from arancio.core.messages import AssistantMessage, Message
from arancio.core.requests import BaseRequest


class ScriptedClient(BaseClient):
    """Client replaying canned messages instead of calling a provider.

    Each entry of ``turns`` is one agent turn: the messages to yield, in order.
    An :class:`Exception` placed inside a turn is raised when it is reached,
    which is how a provider failure mid-stream is scripted.

    Attributes:
        turns: the scripted turns still to replay, consumed from the front.
        requests: every request the agent asked this client to send, in order.
        histories: the model context of each turn, copied at send time. The
            request itself holds the agent's live history list, so it keeps
            growing after the turn; these snapshots do not.
    """

    _default_model_id = "fake/model"

    def __init__(self, turns: List[List[Any]] | None = None) -> None:
        """Initialize the client with the turns it should replay.

        Args:
            turns: the per-turn message lists. A turn past the end of the
                script answers with a plain assistant message, which ends the
                agent loop. Defaults to none.
        """
        super().__init__()
        self.turns: List[List[Any]] = [list(turn) for turn in (turns or [])]
        self.requests: List[BaseRequest] = []
        self.histories: List[List[Message]] = []

    def send_request(self, request: BaseRequest) -> Iterator[Message]:
        """Record the request and replay the next scripted turn.

        Args:
            request: the request built by the agent.

        Returns:
            An iterator over the messages of the next scripted turn.
        """
        self.requests.append(request)
        self.histories.append(list(request.message_list))
        turn = self.turns.pop(0) if self.turns else [AssistantMessage(content="done")]
        return self._replay(turn)

    @staticmethod
    def _replay(turn: List[Any]) -> Iterator[Message]:
        """Yield one turn's messages, raising any exception scripted into it.

        Args:
            turn: the scripted messages, any of which may be an exception.

        Yields:
            Each scripted message, in order. A scripted exception is raised
            rather than yielded, standing in for a provider failure.
        """
        for item in turn:
            if isinstance(item, Exception):
                raise item
            yield item


class ScriptedController(Controller):
    """Controller answering permission requests from a canned script.

    Attributes:
        answers: the decisions still to give, consumed from the front. Each is
            a ``(decision, note)`` pair; an exhausted script allows.
        requests: every request core sent, in order.
    """

    def __init__(
        self, answers: List[Tuple[Decision, str | None]] | None = None
    ) -> None:
        """Initialize the controller with the answers it should give.

        Args:
            answers: the ``(decision, note)`` pairs to reply with, in order.
                Defaults to none, which allows every call.
        """
        self.answers: List[Tuple[Decision, str | None]] = list(answers or [])
        self.requests: List[BaseControllerRequest] = []

    def request(self, request: BaseControllerRequest) -> BaseControllerResponse:
        """Record the request and answer it from the script.

        Args:
            request: the request core is asking the frontend to resolve.

        Returns:
            The scripted permission response, or a bare response for any
            one-way request such as a login notice.
        """
        self.requests.append(request)
        if not isinstance(request, PermissionRequest):
            return BaseControllerResponse()
        decision, note = self.answers.pop(0) if self.answers else (Decision.ALLOW, None)
        return PermissionResponse(decision=decision, message=note)


class FakeApp:
    """The three touchpoints :class:`UIController` uses on a Textual app.

    ``call_from_thread`` runs the callable inline. That is what makes the
    controller testable without a running app: ``_ask`` sets its
    :class:`threading.Event` before it waits on it, so nothing blocks.

    Attributes:
        answer: what the question screen replies with.
        screens: every screen pushed, in order.
        logins: every ``(verification_url, user_code)`` notice shown.
    """

    def __init__(self, answer: str = "yes") -> None:
        """Initialize the fake app with the answer it should give.

        Args:
            answer: the reply the pushed question screen produces. Defaults to
                ``"yes"``.
        """
        self.answer = answer
        self.screens: List[Any] = []
        self.logins: List[Tuple[str, str]] = []

    @staticmethod
    def call_from_thread(func: Callable[..., Any], *args: Any) -> Any:
        """Run the callable inline instead of hopping to the UI thread.

        Args:
            func: the callable the worker thread wants the UI to run.
            *args: the positional arguments to call it with.

        Returns:
            Whatever the callable returned.
        """
        return func(*args)

    def push_screen(self, screen: Any, callback: Callable[[str], None]) -> None:
        """Record the screen and answer it immediately.

        Args:
            screen: the modal screen the controller wants shown.
            callback: the continuation the controller passed to receive the
                answer.
        """
        self.screens.append(screen)
        callback(self.answer)

    def show_chatgpt_login(self, verification_url: str, user_code: str) -> None:
        """Record a ChatGPT device-code notice.

        Args:
            verification_url: the URL the user must open.
            user_code: the code the user must enter there.
        """
        self.logins.append((verification_url, user_code))


class RecordingApp:
    """The app surface the commands and the action executor reach for.

    The real :class:`~arancio.ui.app.App` owns a Textual widget tree, but the
    commands only ever ask it to move the working directory, refresh the
    toolbar, clear the log or quit. This records those calls.

    Attributes:
        working_directory: the directory commands resolve against.
        cleared: how many times the log was cleared.
        exited: whether the app was asked to quit.
        displayed_model_ids: every model id pushed to the toolbar, in order.
        displayed_efforts: every thinking effort pushed to the toolbar.
    """

    def __init__(self, working_directory: Path | None = None) -> None:
        """Initialize the fake app.

        Args:
            working_directory: the starting directory. Defaults to the
                process working directory.
        """
        self.working_directory = working_directory or Path.cwd()
        self.cleared = 0
        self.exited = False
        self.displayed_model_ids: List[str | None] = []
        self.displayed_efforts: List[str | None] = []

    def set_working_directory(self, path: Path | str) -> Path:
        """Move the working directory, validating it the way the app does.

        Args:
            path: the target directory, absolute or relative to the current
                working directory.

        Returns:
            The resolved directory.

        Raises:
            ValueError: when the target is not an existing directory.
        """
        resolved = (self.working_directory / Path(path).expanduser()).resolve()
        if not resolved.is_dir():
            raise ValueError(f"{resolved} does not exist")
        self.working_directory = resolved
        return resolved

    def clear_log(self) -> None:
        """Record that the displayed log was emptied."""
        self.cleared += 1

    def exit(self) -> None:
        """Record that the app was asked to quit."""
        self.exited = True

    def set_displayed_model_id(self, model_id: str | None) -> None:
        """Record a toolbar model id update.

        Args:
            model_id: the model id now shown.
        """
        self.displayed_model_ids.append(model_id)

    def set_displayed_effort(self, effort: str | None) -> None:
        """Record a toolbar thinking effort update.

        Args:
            effort: the effort level now shown.
        """
        self.displayed_efforts.append(effort)
