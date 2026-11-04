"""Scrolling log rendering conversation messages and streaming Markdown chunks."""

from typing import Any

from textual.containers import VerticalScroll
from textual.widget import Widget
from textual.widgets import Markdown, Static
from textual.widgets.markdown import MarkdownStream

from arancio.core.messages.assistant import AssistantChunkMessage, AssistantMessage
from arancio.core.messages.base import ErrorMessage, Message, WarningMessage
from arancio.core.messages.reasoning import ReasoningChunkMessage, ReasoningMessage
from arancio.core.messages.tool import (
    ToolCallMessage,
    ToolErrorMessage,
    ToolResultMessage,
)
from arancio.ui.widgets.login import LoginPanel


class MessageLog(VerticalScroll):
    """Show messages one under the other, streaming chunks into Markdown spans.

    Assistant and reasoning chunks build inline in a single Markdown widget per span,
    and the finalized echo of a streamed span is suppressed so its text is not shown
    twice. The log renders what it is handed and records nothing.

    Every method must run on the UI thread; callers on another thread reach it through
    ``TUIApp.call_from_thread``.
    """

    def __init__(self, **kwargs: Any) -> None:
        """Initialize the stream state and forward the standard widget options.

        Args:
            **kwargs: the standard Textual widget options, such as ``id``.
        """
        super().__init__(**kwargs)
        self.__streaming_kind: type[Message] | None = None
        self.__stream_widget: Markdown | None = None
        self.__stream: MarkdownStream | None = None
        self.__suppress: set[type[Message]] = set()

    async def append(self, message: Message) -> None:
        """Stream a chunk into the current span, or show a finalized message.

        The finalized echo of a span already shown as chunks is skipped once.

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
            await self.__append_chunk(kind, classes, message.display_text)
            return

        await self.end_stream()
        if type(message) in self.__suppress:
            self.__suppress.discard(type(message))
            return
        await self.show(message)

    async def show(self, message: Message) -> None:
        """Mount the widget of a finalized message and scroll to it.

        Args:
            message: the finalized message to display.
        """
        await self.__mount_widget(self.__message_widget(message))
        self.scroll_end(animate=False)

    async def end_stream(self) -> None:
        """Stop the active stream, flushing buffered deltas, and clear its state.

        The next span renders into its own widget.
        """
        if self.__stream is not None:
            await self.__stream.stop()
            self.__stream = None
        self.__streaming_kind = None
        self.__stream_widget = None

    async def reset_stream(self) -> None:
        """End the active stream and forget which finalized echoes to suppress."""
        await self.end_stream()
        self.__suppress.clear()

    async def replace(self, messages: list[Message]) -> None:
        """Replace the whole conversation with the supplied messages.

        Args:
            messages: the messages to display in order, or an empty list to clear.
        """
        await self.reset_stream()
        await self.remove_children()
        for message in messages:
            await self.show(message)

    async def show_login(
        self, provider: str, verification_url: str, user_code: str
    ) -> None:
        """Mount a display-only login panel and scroll to it.

        Args:
            provider: display name of the provider asking for the sign-in.
            verification_url: browser address where the user enters the code.
            user_code: short code issued for the current login attempt.
        """
        await self.__mount_widget(LoginPanel(provider, verification_url, user_code))
        self.scroll_end(animate=False)

    async def __append_chunk(
        self, kind: type[Message], classes: str | None, delta: str
    ) -> None:
        """Stream a delta into the current Markdown span.

        A new span (different ``kind``) starts a fresh widget and
        :class:`MarkdownStream`; deltas are written to the stream, which coalesces
        bursts into as few renders as the UI can keep up with so a fast token stream
        never saturates the UI thread (which also handles input).

        Args:
            kind: the finalized message type the chunk belongs to.
            classes: CSS classes for the span's Markdown widget.
            delta: the incremental text to append.
        """
        if self.__streaming_kind is not kind:
            await self.end_stream()
            self.__streaming_kind = kind
            self.__suppress.add(kind)
            self.__stream_widget = Markdown(classes=classes)
            # await the mount so the widget's on-mount update() completes before
            # the stream's first append, otherwise it would wipe the first delta
            await self.__mount_widget(self.__stream_widget)
            self.__stream = Markdown.get_stream(self.__stream_widget)
        await self.__stream.write(delta)

    async def __mount_widget(self, widget: Widget) -> None:
        """Mount a widget at the bottom of the log and wait until it is mounted.

        Args:
            widget: the widget to append to the log.
        """
        await self.mount(widget)

    @staticmethod
    def __message_widget(message: Message) -> Widget:
        """Build the widget rendering a finalized message.

        Args:
            message: the finalized message to render.

        Returns:
            A Markdown widget for assistant/reasoning text, otherwise a styled
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
