"""Textual terminal UI streaming the agent's output and gating its tool use."""

from textual import work
from textual.app import App as TextualApp
from textual.app import ComposeResult
from textual.containers import VerticalScroll
from textual.widget import Widget
from textual.widgets import Input, Markdown, Static

from arancio.core.agents import Agent
from arancio.core.types.messages import (
    AssistantChunkMessage,
    AssistantMessage,
    ErrorMessage,
    Message,
    ReasoningChunkMessage,
    ReasoningMessage,
    ToolCallMessage,
    ToolErrorMessage,
    ToolResultMessage,
    UserMessage,
)


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
    QuestionScreen { align: center middle; }
    #question-dialog {
        width: 70%; height: auto; padding: 1 2;
        background: $surface; border: round $primary;
    }
    #question-answers { height: auto; }
    """

    BINDINGS = [("ctrl+c", "quit", "Quit")]

    def __init__(self, agent: Agent, model_id: str) -> None:
        """Initialize the app with the agent it drives and the model label.

        Args:
            agent: the agent whose run loop the app streams.
            model_id: the model identifier shown in the toolbar.
        """
        super().__init__()
        self._agent = agent
        self._model_id = model_id
        self._busy = False
        self._streaming_kind: type | None = None
        self._stream_widget: Markdown | None = None
        self._stream_buffer = ""
        self._suppress: set[type] = set()

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
        """Focus the prompt input once the UI is mounted."""
        self.query_one("#prompt", Input).focus()

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
        self._suppress.clear()

    @work(thread=True)
    def _run_agent(self, text: str) -> None:
        """Run the agent loop on a worker thread, rendering each message.

        Args:
            text: the user message that starts the turn.
        """
        try:
            for message in self._agent.run(UserMessage(content=text)):
                self.call_from_thread(self._handle_message, message)
        finally:
            self.call_from_thread(self._set_busy, False)

    def _handle_message(self, message: Message) -> None:
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
            self._handle_chunk(kind, classes, message.display_text)
            return

        self._streaming_kind = None
        self._stream_widget = None
        if type(message) in self._suppress:
            self._suppress.discard(type(message))
            return
        self._mount(self._render(message))

    def _handle_chunk(self, kind: type, classes: str | None, delta: str) -> None:
        """Append a streaming delta to the current markdown span.

        Args:
            kind: the finalized message type the chunk belongs to.
            classes: CSS classes for the span's markdown widget.
            delta: the incremental text to append.
        """
        if self._streaming_kind is not kind:
            self._streaming_kind = kind
            self._suppress.add(kind)
            self._stream_buffer = delta
            self._stream_widget = Markdown(self._stream_buffer, classes=classes)
            self._mount(self._stream_widget)
        else:
            self._stream_buffer += delta
            self._stream_widget.update(self._stream_buffer)
            self.query_one("#log", VerticalScroll).scroll_end(animate=False)

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

    def _toolbar_text(self) -> str:
        """Build the toolbar text (placeholder content; fields TBD).

        Returns:
            The model id and the current running/ready status.
        """
        status = "working" if self._busy else "ready"
        return f"model: {self._model_id}  ·  {status}"
