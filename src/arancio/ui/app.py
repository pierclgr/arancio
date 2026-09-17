"""Textual terminal UI streaming the agent's output and gating its tool use."""

import os
from pathlib import Path

from textual import work
from textual.app import App as TextualApp
from textual.app import ComposeResult
from textual.containers import VerticalScroll
from textual.widget import Widget
from textual.widgets import Input, Markdown, Static
from textual.widgets.markdown import MarkdownStream

from arancio.core.agents import Agent
from arancio.core.messages import (
    AssistantChunkMessage,
    AssistantMessage,
    ErrorMessage,
    Message,
    ReasoningChunkMessage,
    ReasoningMessage,
    ToolCallMessage,
    ToolErrorMessage,
    ToolResultMessage,
    WarningMessage,
)
from arancio.prompt.actions.executor import ActionExecutor
from arancio.prompt.actions.factory import ActionFactory
from arancio.prompt.manager import PromptManager
from arancio.sessions.manager import SessionManager
from arancio.settings.manager import SettingsManager
from arancio.ui.widgets.chatgpt_login import ChatGPTLoginNotice


class App(TextualApp):
    """Terminal UI streaming the agent's output and gating its tool use.

    A worker thread iterates ``Agent.run`` so the blocking agent loop never freezes the
    UI; each produced message is rendered on the main thread via
    :meth:`call_from_thread`. Assistant and reasoning text render as markdown (reasoning
    dimmed), for both finalized messages and streaming chunks.
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
    .chatgpt-login {
        height: auto; margin: 0 0 1 0; padding: 1;
        border: round $warning;
    }
    .chatgpt-login-code { text-style: bold; }
    QuestionScreen { align: center middle; }
    #question-dialog {
        width: 70%; height: auto; padding: 1 2;
        background: $surface; border: round $primary;
    }
    #question-answers { height: auto; }
    """

    BINDINGS = [("ctrl+c", "quit", "Quit")]

    def __init__(
        self,
        agent: Agent,
        model_id: str,
        settings_manager: SettingsManager,
        session_manager: SessionManager,
        startup_messages: list[Message] | None = None,
    ) -> None:
        """Initialize the app with the agent it drives and the model label.

        Args:
            agent: the agent whose run loop the app streams.
            model_id: the model identifier shown in the toolbar.
            settings_manager: the manager used to apply and persist settings
                changes made through commands (e.g. ``/model``, ``/effort``).
            session_manager: the active persistent chat session.
            startup_messages: messages to render once on mount (e.g. settings
                validation warnings/errors produced while loading
                ``settings.yml``). Defaults to none.
        """
        super().__init__()
        self._agent = agent
        self._session_manager = session_manager
        self._working_directory: Path = Path.cwd()
        self._action_executor = ActionExecutor(
            agent=agent,
            application=self,
            settings_manager=settings_manager,
            session_manager=session_manager,
        )
        self._model_id = model_id
        self._effort = settings_manager.settings.thinking_effort
        self._busy = False
        self._streaming_kind: type | None = None
        self._stream_widget: Markdown | None = None
        self._stream: MarkdownStream | None = None
        self._suppress: set[type] = set()
        self._startup_messages = list(startup_messages or [])

    def compose(self) -> ComposeResult:
        """Build the main page: message log, prompt input and toolbar.

        Yields:
            The scrolling message log, the prompt input and the toolbar, in
            top-to-bottom order.
        """
        yield VerticalScroll(id="log")
        yield Input(placeholder="Type a message…", id="prompt")
        yield Static(self._toolbar_text(), id="toolbar")

    def on_mount(self) -> None:
        """Focus the prompt input, anchor the log, and show startup messages."""
        self.query_one("#prompt", Input).focus()
        # keep the log pinned to the bottom as streamed content grows, until
        # the user scrolls up
        self.query_one("#log", VerticalScroll).anchor()
        if self._session_manager.current:
            for message in self._session_manager.visible_messages():
                self._mount(self._render(message))
        for message in self._startup_messages:
            self._mount(self._render(message))

    def on_input_submitted(self, event: Input.Submitted) -> None:
        """Start an agent turn when the prompt input is submitted.

        Args:
            event: the input-submitted event (ignored for non-prompt inputs and
                while a turn is already running).
        """
        if event.input.id != "prompt":
            return
        if self._busy:
            return
        text = event.value.strip()
        if not text:
            return
        event.input.value = ""
        self._mount(Static(text, classes="user"))
        self._reset_stream()
        self._set_busy(True)
        self._run_agent(text)

    def _reset_stream(self) -> None:
        """Clear per-turn streaming state so turns render independently."""
        self._streaming_kind = None
        self._stream_widget = None
        self._stream = None
        self._suppress.clear()

    @work(thread=True)
    def _run_agent(self, text: str) -> None:
        """Run the agent loop on a worker thread, rendering each message.

        Args:
            text: the user message that starts the turn.
        """
        try:
            # resolve the prompt into action arguments, build the action, then run
            # it: slash and shell commands are handled locally, while a prompt
            # goes to the model. all yield a message stream rendered the same way
            resolved_arguments = PromptManager.resolve_prompt(text)
            action = ActionFactory.create_action(raw_input=text, **resolved_arguments)
            for message in self._action_executor.execute(action):
                self.call_from_thread(self._handle_message, message)
        except ValueError as exc:
            # a malformed prompt (e.g. an unclosed quote) is reported like any
            # other failure instead of killing the worker thread, and saved so a
            # resumed log keeps the only trace of that turn
            error = ErrorMessage(content=f"Invalid prompt: {exc}")
            save_error = self._session_manager.session_recorder.message(error)
            self.call_from_thread(self._handle_message, error)
            if save_error:
                self.call_from_thread(self._handle_message, save_error)
        finally:
            self.call_from_thread(self._end_stream)
            self.call_from_thread(self._set_busy, False)

    async def _handle_message(self, message: Message) -> None:
        """Render a single message, streaming chunks into one widget.

        Mirrors the console renderer: assistant and reasoning chunks build
        inline in a single markdown widget per span, and the finalized echo of
        a streamed span is suppressed so its text is not shown twice.

        Args:
            message: the message produced by the agent run.
        """
        if isinstance(message, AssistantChunkMessage):
            kind, classes = AssistantMessage, None
        elif isinstance(message, ReasoningChunkMessage):
            kind, classes = ReasoningMessage, "reasoning"
        else:
            kind, classes = None, None

        if kind is not None:
            await self._handle_chunk(kind, classes, message.display_text)
            return

        await self._end_stream()
        if type(message) in self._suppress:
            self._suppress.discard(type(message))
            return
        self._mount(self._render(message))

    async def _handle_chunk(self, kind: type, classes: str | None, delta: str) -> None:
        """Stream a delta into the current markdown span.

        A new span (different ``kind``) starts a fresh widget and
        :class:`MarkdownStream`; deltas are written to the stream, which
        coalesces bursts into as few renders as the UI can keep up with so a
        fast token stream never saturates the UI thread (which also handles
        input).

        Args:
            kind: the finalized message type the chunk belongs to.
            classes: CSS classes for the span's markdown widget.
            delta: the incremental text to append.
        """
        if self._streaming_kind is not kind:
            await self._end_stream()
            self._streaming_kind = kind
            self._suppress.add(kind)
            self._stream_widget = Markdown(classes=classes)
            # await the mount so the widget's on-mount update() completes before
            # the stream's first append, otherwise it would wipe the first delta
            await self.query_one("#log", VerticalScroll).mount(self._stream_widget)
            self._stream = Markdown.get_stream(self._stream_widget)
        await self._stream.write(delta)

    async def _end_stream(self) -> None:
        """Stop the active stream, flushing buffered deltas, and reset state.

        Resets the per-span streaming state so the next span renders into its own
        widget.
        """
        if self._stream is not None:
            await self._stream.stop()
            self._stream = None
        self._streaming_kind = None
        self._stream_widget = None

    @staticmethod
    def _render(message: Message) -> Widget:
        """Build the widget rendering a finalized message.

        Args:
            message: the finalized message to render.

        Returns:
            A markdown widget for assistant/reasoning text, otherwise a styled
            static line.
        """
        if isinstance(message, ReasoningMessage):
            return Markdown(message.display_text, classes="reasoning")
        if isinstance(message, AssistantMessage):
            return Markdown(message.display_text)
        if isinstance(message, ToolCallMessage):
            return Static(f"→ {message.name}({message.arguments})", classes="tool-call")
        if isinstance(message, (ToolErrorMessage, ErrorMessage)):
            return Static(message.display_text, classes="error")
        if isinstance(message, WarningMessage):
            return Static(message.display_text, classes="warning")
        if isinstance(message, ToolResultMessage):
            return Static(message.display_text, classes="tool-result")
        return Static(message.display_text, classes="user")

    def _mount(self, widget: Widget) -> None:
        """Mount a widget at the bottom of the log and scroll to it.

        Args:
            widget: the widget to append to the message log.
        """
        log = self.query_one("#log", VerticalScroll)
        log.mount(widget)
        log.scroll_end(animate=False)

    def _set_busy(self, busy: bool) -> None:
        """Update the busy state and refocus the prompt when idle.

        The prompt is never disabled (disabling and re-enabling it corrupts its
        focus/render state); concurrent turns are instead blocked by the
        ``_busy`` guard in :meth:`on_input_submitted`. The flag is named
        ``_busy`` (not ``_running``) to avoid colliding with Textual's internal
        ``App._running`` attribute.

        Args:
            busy: whether an agent turn is currently in progress.
        """
        self._busy = busy
        self.query_one("#toolbar", Static).update(self._toolbar_text())
        if not busy:
            self.query_one("#prompt", Input).focus()

    def clear_log(self) -> None:
        """Remove every rendered message, matching a cleared chat history."""
        self._reset_stream()
        self.query_one("#log", VerticalScroll).remove_children()

    def show_chatgpt_login(self, verification_url: str, user_code: str) -> None:
        """Show the active ChatGPT device-login link and code.

        Called on the Textual UI thread by :class:`UIController`; the agent
        worker remains free to wait for LiteLLM's authorization polling.

        Args:
            verification_url: browser address where the user enters the code.
            user_code: short code issued for the current login attempt.
        """
        self._mount(ChatGPTLoginNotice(verification_url, user_code))

    @property
    def working_directory(self) -> Path:
        """Return the app's working directory.

        Returns:
            The absolute :class:`~pathlib.Path` the app is working in.
        """
        return self._working_directory

    def set_working_directory(self, path: Path | str) -> None:
        """Move the working directory to ``path`` and mirror it to the process.

        The path is expanded and resolved, then validated as an existing
        directory. On success the app's working directory is updated and
        ``os.chdir`` is called so the process cwd follows, letting every
        consumer that reads ``Path.cwd()`` (or passes ``cwd=None`` to
        ``subprocess.run``) follow the change with no rewiring. The toolbar is
        redrawn so the displayed directory matches.

        Args:
            path: the new working directory, absolute or relative (resolved
                against the current process cwd), with ``~`` expansion.

        Raises:
            ValueError: when ``path`` does not exist, or exists but is not a
                directory; the two cases are reported differently, since a
                missing path and a path pointing at a file need different
                corrections.
        """
        resolved = Path(path).expanduser().resolve()
        if not resolved.exists():
            raise ValueError(f"{resolved} does not exist")
        if not resolved.is_dir():
            raise ValueError(
                f"{resolved} not a directory: did you mean {resolved.parent}?"
            )
        self._working_directory = resolved
        os.chdir(resolved)
        # before the app runs there is no toolbar to update; compose() renders
        # the current directory itself when it builds one
        if self.is_running:
            self.query_one("#toolbar", Static).update(self._toolbar_text())

    def set_displayed_model_id(self, model_id: str) -> None:
        """Update the model id shown in the toolbar.

        Args:
            model_id: the model id to display.
        """
        self._model_id = model_id
        self.query_one("#toolbar", Static).update(self._toolbar_text())

    def set_displayed_effort(self, effort: str | None) -> None:
        """Update the thinking effort shown in the toolbar.

        Args:
            effort: the thinking effort to display, or ``None`` when thinking
                is disabled.
        """
        self._effort = effort
        self.query_one("#toolbar", Static).update(self._toolbar_text())

    def _toolbar_text(self) -> str:
        """Build the toolbar text (placeholder content; fields TBD).

        Returns:
            The model id, thinking effort, working directory and the current
            running/ready status.
        """
        status = "working" if self._busy else "ready"
        effort = self._effort if self._effort is not None else "null"
        return (
            f"model: {self._model_id}  ·  effort: {effort}  ·  "
            f"{self._working_directory}  ·  {status}"
        )
