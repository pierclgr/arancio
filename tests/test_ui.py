"""Tests for the Textual UI: the question screen and message rendering."""

import asyncio
from collections.abc import Iterator

from textual.widgets import Input, Markdown

from arancio.core.types.messages import (
    AssistantMessage,
    Message,
    ReasoningMessage,
    UserMessage,
)
from arancio.ui.app import App
from arancio.ui.widgets.question import QuestionScreen


class _DummyAgent:
    """Stand-in agent; the UI tests never run its loop."""


class _ScriptedAgent:
    """Agent stub whose ``run`` yields a fixed message list then stops."""

    def __init__(self, messages: list[Message]) -> None:
        """Store the messages each run yields.

        Args:
            messages: the messages produced for every ``run`` call.
        """
        self._messages = messages

    def run(self, message: Message) -> Iterator[Message]:
        """Yield the configured messages, ignoring the input.

        Args:
            message: the user message that starts the turn (unused).

        Yields:
            Each configured message in order.
        """
        yield from self._messages


class _SyncApp(App):
    """App whose agent loop runs inline (thread workers do not run under run_test)."""

    def _run_agent(self, text: str) -> None:
        """Run the agent loop synchronously (no worker thread).

        Args:
            text: the user message that starts the turn.
        """
        try:
            for message in self._agent.run(UserMessage(content=text)):
                self._handle_message(message)
        finally:
            self._set_busy(False)


def _app() -> App:
    """Build an app wired to a dummy agent for UI tests.

    Returns:
        An :class:`App` instance with a dummy agent.
    """
    return App(agent=_DummyAgent(), model_id="test-model")


def test_question_screen_dismisses_with_pressed_answer() -> None:
    """Pressing an answer button dismisses the screen with that answer's label."""

    async def _run() -> None:
        app = _app()
        async with app.run_test() as pilot:
            result: list[str] = []
            app.push_screen(QuestionScreen("ok?", ["yes", "no"]), result.append)
            await pilot.pause()
            await pilot.click("#answer-0")
            await pilot.pause()
            assert result == ["yes"]

    asyncio.run(_run())


def test_question_screen_dismisses_with_custom_answer() -> None:
    """Submitting the custom input dismisses the screen with the typed text."""

    async def _run() -> None:
        app = _app()
        async with app.run_test() as pilot:
            result: list[str] = []
            app.push_screen(QuestionScreen("ok?", ["yes", "no"]), result.append)
            await pilot.pause()
            custom = app.screen.query_one("#custom-answer")
            custom.focus()
            custom.value = "do it differently"
            await pilot.pause()
            await pilot.press("enter")
            await pilot.pause()
            assert result == ["do it differently"]

    asyncio.run(_run())


def test_finalized_assistant_message_renders_markdown() -> None:
    """A finalized assistant message mounts a markdown widget in the log."""

    async def _run() -> None:
        app = _app()
        async with app.run_test() as pilot:
            app._handle_message(AssistantMessage(content="# hi"))
            await pilot.pause()
            assert len(app.query(Markdown)) == 1

    asyncio.run(_run())


def test_finalized_reasoning_message_renders_dimmed_markdown() -> None:
    """A finalized reasoning message mounts a markdown widget tagged ``reasoning``."""

    async def _run() -> None:
        app = _app()
        async with app.run_test() as pilot:
            app._handle_message(ReasoningMessage(content="thinking", item={}))
            await pilot.pause()
            widgets = app.query(".reasoning")
            assert len(widgets) == 1
            assert isinstance(widgets.first(), Markdown)

    asyncio.run(_run())


def test_set_running_keeps_prompt_enabled_and_refocuses() -> None:
    """Toggling the running state never disables the prompt and refocuses it idle.

    Regression for the prompt becoming unusable after the first turn: it used to
    be disabled while running and re-enabled after, which corrupted its focus and
    render state.
    """

    async def _run() -> None:
        app = _app()
        async with app.run_test() as pilot:
            prompt = app.query_one("#prompt", Input)

            app._set_busy(True)
            await pilot.pause()
            assert prompt.disabled is False

            app._set_busy(False)
            await pilot.pause()
            assert prompt.disabled is False
            assert app.focused is prompt

    asyncio.run(_run())


def test_submit_ignored_while_running() -> None:
    """Submitting while a turn runs mounts nothing (the concurrency guard)."""

    async def _run() -> None:
        app = _app()
        async with app.run_test() as pilot:
            app._busy = True
            app.query_one("#prompt", Input).focus()
            await pilot.press(*"hello")
            await pilot.press("enter")
            await pilot.pause()
            assert len(app.query(".user")) == 0

    asyncio.run(_run())


def test_enter_sends_consecutive_messages() -> None:
    """Pressing Enter sends each message: both turns mount a user line + reply.

    Regression for Enter not sending: the busy flag was named ``_running``,
    colliding with Textual's internal ``App._running`` (True while the app runs),
    so the guard blocked every submit.
    """

    async def _run() -> None:
        app = _SyncApp(
            agent=_ScriptedAgent([AssistantMessage(content="reply")]), model_id="m"
        )
        async with app.run_test() as pilot:
            prompt = app.query_one("#prompt", Input)

            await pilot.press(*"first")
            await pilot.press("enter")
            await pilot.pause()
            assert len(app.query(".user")) == 1
            assert len(app.query(Markdown)) == 1
            assert prompt.value == ""
            assert app.focused is prompt

            await pilot.press(*"second")
            await pilot.press("enter")
            await pilot.pause()
            assert len(app.query(".user")) == 2
            assert len(app.query(Markdown)) == 2

    asyncio.run(_run())


def test_reset_stream_clears_cross_turn_state() -> None:
    """Resetting clears the suppress set so a new turn's output is not swallowed."""

    async def _run() -> None:
        app = _app()
        async with app.run_test() as pilot:
            app._suppress.add(AssistantMessage)
            app._reset_stream()
            app._handle_message(AssistantMessage(content="reply"))
            await pilot.pause()
            assert len(app.query(Markdown)) == 1

    asyncio.run(_run())
