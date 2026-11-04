"""Textual adapter implementing the abstract frontend controller.

Application and the model clients call the controller on the frontend's worker thread.
The adapter moves each call to the Textual UI thread and, for questions, blocks that
worker (never the UI thread) until the user answers or the frontend closes.
"""

import inspect
import threading
from collections.abc import Callable
from typing import Any

from arancio.core.events import LoginNotice
from arancio.core.messages.base import Message
from arancio.ui.apps.tui import TUIApp
from arancio.ui.controllers.base import UIController
from arancio.ui.widgets.question import QuestionScreen


class TUIController(UIController):
    """Bridge frontend operations to the Textual app and block for question answers.

    The app reference is set after construction: the app needs the Application, which
    needs this controller, so the controller is built first and ``app`` is assigned
    once the app exists.

    Cancellation is permanent: once the frontend is closing, questions return None and
    display or exit calls do nothing. Only one question may be pending at a time.

    Attributes:
        app: the Textual app operations are dispatched to.
        __lock: protects the cancellation flag and the pending-answer state.
        __cancelled: whether the frontend is closing.
        __pending_event: releases the worker waiting for the current question.
        __pending_answer: the selected text, or None after cancellation.
    """

    def __init__(self) -> None:
        """Initialize the controller with no app attached yet."""
        self.app: TUIApp | None = None
        self.__lock: threading.Lock = threading.Lock()
        self.__cancelled: bool = False
        self.__pending_event: threading.Event | None = None
        self.__pending_answer: str | None = None

    def ask(self, question: str, answers: list[str]) -> str | None:
        """Show a question screen and block the calling worker for the answer.

        Args:
            question: the text to display.
            answers: the preset answer labels offered as buttons.

        Returns:
            The selected label, the typed free text, or None when the frontend is
            closing or was cancelled.

        Raises:
            RuntimeError: when another question is still pending.
        """
        event = threading.Event()
        with self.__lock:
            if self.__cancelled:
                return None
            if self.__pending_event is not None:
                raise RuntimeError(
                    f"A question is already pending; cannot ask {question!r}."
                )
            self.__pending_event = event
            self.__pending_answer = None

        try:
            shown = self.__call_ui(
                lambda app: app.push_screen(
                    QuestionScreen(question, answers),
                    lambda answer: self.__store_answer(event, answer),
                )
            )
            if not shown:
                return None
            # the lock is not held while waiting, so cancellation can release the wait
            event.wait()
            with self.__lock:
                return self.__pending_answer
        finally:
            with self.__lock:
                if self.__pending_event is event:
                    self.__pending_event = None
                    self.__pending_answer = None

    def cancel_questions(self) -> None:
        """Mark the frontend as closing and release a pending question with None.

        Uses no Textual call and never waits, so it is safe on any thread.
        """
        with self.__lock:
            self.__cancelled = True
            self.__pending_answer = None
            if self.__pending_event is not None:
                self.__pending_event.set()

    def show_login(self, notice: LoginNotice) -> None:
        """Show the login details on the UI thread without waiting for sign-in.

        Args:
            notice: the provider, verification URL and user code.
        """
        self.__call_ui(lambda app: app.show_login(notice))

    def reset_conversation(self, messages: list[Message]) -> None:
        """Replace the visible conversation on the UI thread.

        Args:
            messages: the messages to display, or an empty list for a new chat.
        """
        replay = list(messages)
        self.__call_ui(lambda app: app.replace_conversation(replay))

    def exit(self) -> None:
        """Request that the app closes; its unmount calls Application.shutdown."""
        self.__call_ui(lambda app: app.exit())

    def __require_app(self) -> TUIApp:
        """Return the attached app or fail before dispatching to it.

        Every call arrives once the app is running, long after wiring assigned
        ``app``, so this states that expectation instead of leaving each dispatch to
        assume it.

        Returns:
            The attached Textual app.

        Raises:
            ValueError: when no app has been attached to this controller.
        """
        if self.app is None:
            raise ValueError("No app attached to the controller.")
        return self.app

    def __store_answer(self, event: threading.Event, answer: str | None) -> None:
        """Store the answer of a question and release its waiting worker.

        A late callback, from a cancelled or an outdated question, changes nothing.

        Args:
            event: the event of the question being answered.
            answer: the selected label or typed text.
        """
        with self.__lock:
            if self.__cancelled or self.__pending_event is not event:
                return
            self.__pending_answer = answer
            event.set()

    def __call_ui(self, call: Callable[[TUIApp], Any]) -> bool:
        """Run a call on the UI thread, unless the frontend is closing or stopped.

        A teardown ``RuntimeError`` is suppressed only when the app has stopped;
        any other error propagates to the caller.

        Args:
            call: the operation to run with the app on the UI thread.

        Returns:
            True when the call ran, False when it was skipped.

        Raises:
            RuntimeError: when dispatching fails while the app is still running.
        """
        app = self.__require_app()
        if self.__is_cancelled() or not app.is_running:
            return False
        try:
            return app.call_from_thread(self.__dispatch_if_active, app, call)
        except RuntimeError:
            if app.is_running:
                raise
            return False

    async def __dispatch_if_active(
        self, app: TUIApp, call: Callable[[TUIApp], Any]
    ) -> bool:
        """Recheck cancellation on the UI thread, then run the call to completion.

        Args:
            app: the Textual app the call operates on.
            call: the operation to run with the app.

        Returns:
            True when the call ran, False when the frontend was cancelled first.
        """
        if self.__is_cancelled():
            return False
        result = call(app)
        if inspect.isawaitable(result):
            await result
        return True

    def __is_cancelled(self) -> bool:
        """Return whether the frontend is closing.

        Returns:
            True once cancel_questions has been called.
        """
        with self.__lock:
            return self.__cancelled
