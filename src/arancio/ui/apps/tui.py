"""Textual terminal UI streaming the agent's output and gating its tool use."""

from collections.abc import Callable
from typing import Any

from textual import work
from textual.app import App as TextualApp
from textual.app import ComposeResult
from textual.widgets import Input, Static

from arancio.application.base import Application
from arancio.core.events import LoginNotice
from arancio.core.messages.base import Message
from arancio.core.messages.user import UserMessage
from arancio.ui.widgets.message_log import MessageLog


class TUIApp(TextualApp):
    """Terminal UI streaming the agent's output and gating its tool use.

    The app holds only the :class:`Application` and presentation state. A worker thread
    iterates ``Application.submit`` so the blocking agent loop never freezes the UI;
    each produced message is rendered on the main thread via :meth:`call_from_thread`.
    Rendering and streaming belong to :class:`MessageLog`.
    """

    CSS = """
    #log { height: 1fr; padding: 0 1; }
    #toolbar { height: 1; padding: 0 1; color: $text-muted; background: $panel; }
    Markdown { margin: 0 0 1 0; padding: 0; }
    .user { color: $accent; }
    .reasoning, .reasoning * { color: $text-muted; text-style: italic; }
    .tool-call { color: $text-muted; }
    .error { color: $error; }
    .warning { color: $warning; }
    .login {
        height: auto; margin: 0 0 1 0; padding: 1;
        border: round $warning;
    }
    .login-code { text-style: bold; }
    QuestionScreen { align: center middle; }
    #question-dialog {
        width: 70%; height: auto; padding: 1 2;
        background: $surface; border: round $primary;
    }
    #question-answers { height: auto; }
    """

    BINDINGS = [("ctrl+c", "quit", "Quit")]

    def __init__(self, application: Application) -> None:
        """Initialize the presentation state and the message log.

        Args:
            application: the only backend entry point the app talks to.
        """
        super().__init__()
        self.__application: Application = application
        self.__message_log: MessageLog = MessageLog(id="log")
        self.__busy: bool = False

    def compose(self) -> ComposeResult:
        """Build the main page: message log, prompt input and toolbar.

        Yields:
            The scrolling message log, the prompt input and the toolbar, in
            top-to-bottom order.
        """
        yield self.__message_log
        yield Input(placeholder="Type a message…", id="prompt")
        # markup is off so a working directory with brackets shows verbatim
        yield Static(self.__toolbar_text(), id="toolbar", markup=False)

    async def on_mount(self) -> None:
        """Prepare the log, show the startup messages and draw the toolbar."""
        # keep the log pinned to the bottom as streamed content grows, until
        # the user scrolls up
        self.__message_log.anchor()
        for message in self.__application.start():
            await self.__message_log.show(message)
        # the app starts idle: this redraws the toolbar and focuses the prompt
        self.__set_busy(False)

    def on_unmount(self) -> None:
        """Shut the application down before the app finishes closing.

        Widgets are already unmounted, so messages returned by the shutdown hooks are
        not displayed.
        """
        self.__application.shutdown()

    async def on_input_submitted(self, event: Input.Submitted) -> None:
        """Start a submission when the prompt input is submitted.

        Args:
            event: the input-submitted event (ignored for non-prompt inputs and
                while a submission is already running).
        """
        if event.input.id != "prompt":
            return
        if self.__busy:
            return
        text = event.value.strip()
        if not text:
            return
        # busy is set before any await, so a second submit cannot slip in
        self.__set_busy(True)
        event.input.value = ""
        await self.__message_log.reset_stream()
        await self.__message_log.show(UserMessage(text))
        self.__run_submission(text)

    async def show_login(self, notice: LoginNotice) -> None:
        """Show the sign-in details of a provider in the log.

        Called on the UI thread by the controller; the worker stays free to wait for
        the provider's authorization.

        Args:
            notice: the provider, verification URL and user code.
        """
        await self.__message_log.show_login(
            notice.provider, notice.verification_url, notice.user_code
        )

    async def replace_conversation(self, messages: list[Message]) -> None:
        """Replace the visible conversation with the supplied messages.

        Called on the UI thread by the controller.

        Args:
            messages: the messages to display, or an empty list for a new chat.
        """
        await self.__message_log.replace(messages)

    @work(thread=True)
    def __run_submission(self, text: str) -> None:
        """Run one submission on a worker thread, rendering each message.

        Args:
            text: the nonempty input line that starts the submission.
        """
        messages = self.__application.submit(text)
        try:
            for message in messages:
                if not self.__call_ui(self.__message_log.append, message):
                    break
        finally:
            # closing runs Core cleanup, so it must happen on the thread that
            # consumed the iterator
            messages.close()
            self.__call_ui(self.__message_log.end_stream)
            self.__call_ui(self.__set_busy, False)

    def __call_ui(self, callback: Callable[..., Any], *args: Any) -> bool:
        """Run a callback on the UI thread, unless the app is stopping or stopped.

        A teardown ``RuntimeError`` is suppressed only when the app has stopped;
        any other error propagates to the worker.

        Args:
            callback: the function, or coroutine function, to run on the UI thread.
            *args: the arguments passed to the callback.

        Returns:
            True when the callback ran, False when the app was no longer running.

        Raises:
            RuntimeError: when dispatching fails while the app is still running.
        """
        if not self.is_running:
            return False
        try:
            self.call_from_thread(callback, *args)
        except RuntimeError:
            if self.is_running:
                raise
            return False
        return True

    def __set_busy(self, busy: bool) -> None:
        """Update the busy state, redraw the toolbar and refocus the prompt when idle.

        The prompt is never disabled (disabling and re-enabling it corrupts its
        focus/render state); concurrent submissions are instead blocked by the busy
        guard in :meth:`on_input_submitted`. The flag is named ``__busy`` (not
        ``running``) to avoid colliding with Textual's own ``App`` attributes.

        Args:
            busy: whether a submission is currently in progress.
        """
        self.__busy = busy
        self.query_one("#toolbar", Static).update(self.__toolbar_text())
        if not busy:
            self.query_one("#prompt", Input).focus()

    def __toolbar_text(self) -> str:
        """Build the toolbar text from the Application status.

        Returns:
            The provider, model name, reasoning effort, working directory and the
            current working/ready state.
        """
        status = self.__application.status()
        state = "working" if self.__busy else "ready"
        effort = status.model.reasoning.effort
        if effort is None:
            effort = "null"
        return (
            f"provider: {status.model.provider}  ·  model: {status.model.name}  ·  "
            f"effort: {effort}  ·  {status.working_directory}  ·  {state}"
        )
