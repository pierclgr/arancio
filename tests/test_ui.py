"""Tests for the Textual UI: the question screen and message rendering."""

import asyncio
from collections.abc import Iterator
from pathlib import Path

from textual.widgets import Input, Markdown, Static

from arancio.core.messages import (
    AssistantChunkMessage,
    AssistantMessage,
    ErrorMessage,
    Message,
    ReasoningMessage,
    UserMessage,
    WarningMessage,
)
from arancio.core.permissions.types import PermissionCategory, PermissionLevel
from arancio.settings.settings import Settings
from arancio.ui.app import App
from arancio.ui.widgets.question import QuestionScreen


class _DummyAgent:
    """Stand-in agent; the UI tests never run its loop."""


class _FakeSettingsManager:
    """Settings-manager stub recording ``apply``/``save`` calls without disk I/O."""

    def __init__(
        self, provider: str | None = "openai", model_name: str | None = "gpt-4o"
    ) -> None:
        """Build settings with the given provider and model name.

        Args:
            provider: the initial provider, or ``None`` to leave it unset.
            model_name: the initial model name, or ``None`` to leave it unset.
        """
        self.settings = Settings(
            permissions={c: PermissionLevel.ASK for c in PermissionCategory},
            provider=provider,
            model_name=model_name,
            thinking_effort="medium",
            thinking_summary=None,
            max_turns="inf",
            max_retries=3,
            turn_wait_time=1.0,
            turn_wait_time_multiplier=2.0,
        )
        self.applied = False
        self.saved = False

    def apply(self) -> None:
        """Record that the settings were applied to the live objects."""
        self.applied = True

    def save(self) -> None:
        """Record that the settings were persisted to disk."""
        self.saved = True


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
    """App running its agent loop as an async worker, not a thread worker."""

    def _run_agent(self, text: str) -> None:
        """Drive the agent loop via an async worker instead of a thread.

        Args:
            text: the user message that starts the turn.
        """

        async def _drive() -> None:
            """Iterate the agent run, awaiting each rendered message."""
            try:
                for message in self._agent.run(UserMessage(content=text)):
                    await self._handle_message(message)
            finally:
                await self._end_stream()
                self._set_busy(False)

        self.run_worker(_drive())


def _app() -> App:
    """Build an app wired to a dummy agent for UI tests.

    Returns:
        An :class:`App` instance with a dummy agent.
    """
    return App(
        agent=_DummyAgent(),
        model_id="test-model",
        settings_manager=_FakeSettingsManager(),
    )


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
            await app._handle_message(AssistantMessage(content="# hi"))
            await pilot.pause()
            assert len(app.query(Markdown)) == 1

    asyncio.run(_run())


def test_streamed_chunks_accumulate_into_one_widget() -> None:
    """Streamed chunks render into a single markdown widget without dropping text.

    Regression for the first streamed word being dropped and for the per-chunk
    render storm: deltas are written to a coalescing ``MarkdownStream`` and the
    finalized echo is suppressed.
    """

    async def _run() -> None:
        app = _app()
        async with app.run_test() as pilot:
            for delta in ("Async", " turn", " one"):
                await app._handle_message(AssistantChunkMessage(content=delta))
            await pilot.pause()
            await app._handle_message(AssistantMessage(content="Async turn one"))
            await pilot.pause()
            widgets = app.query("#log Markdown")
            assert len(widgets) == 1
            assert widgets.first().source == "Async turn one"

    asyncio.run(_run())


def test_finalized_reasoning_message_renders_dimmed_markdown() -> None:
    """A finalized reasoning message mounts a markdown widget tagged ``reasoning``."""

    async def _run() -> None:
        app = _app()
        async with app.run_test() as pilot:
            await app._handle_message(ReasoningMessage(content="thinking", item={}))
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
            agent=_ScriptedAgent([AssistantMessage(content="reply")]),
            model_id="m",
            settings_manager=_FakeSettingsManager(),
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


def test_unclosed_quote_in_a_command_renders_an_error() -> None:
    """A malformed prompt is reported, not left to kill the worker thread.

    ``PromptManager.resolve_prompt`` raises before the executor is reached, so the error
    cannot be caught by the executor's own wrapping.
    """

    async def _run() -> None:
        app = _app()
        async with app.run_test() as pilot:
            prompt = app.query_one("#prompt", Input)
            prompt.focus()
            prompt.value = '/cd "my folder'

            await pilot.press("enter")
            await app.workers.wait_for_complete()
            await pilot.pause()

            errors = app.query(".error")
            assert len(errors) == 1
            assert "unbalanced quote" in str(errors.first(Static).render())

    asyncio.run(_run())


def test_reset_stream_clears_cross_turn_state() -> None:
    """Resetting clears the suppress set so a new turn's output is not swallowed."""

    async def _run() -> None:
        app = _app()
        async with app.run_test() as pilot:
            app._suppress.add(AssistantMessage)
            app._reset_stream()
            await app._handle_message(AssistantMessage(content="reply"))
            await pilot.pause()
            assert len(app.query(Markdown)) == 1

    asyncio.run(_run())


def test_startup_messages_render_on_mount() -> None:
    """Startup messages (e.g. settings validation) render once when the app mounts."""

    async def _run() -> None:
        app = App(
            agent=_DummyAgent(),
            model_id="test-model",
            settings_manager=_FakeSettingsManager(),
            startup_messages=[
                WarningMessage(content="careful"),
                ErrorMessage(content="broken"),
            ],
        )
        async with app.run_test() as pilot:
            await pilot.pause()
            warnings = app.query(".warning")
            errors = app.query(".error")
            assert len(warnings) == 1
            assert str(warnings.first().render()) == "careful"
            assert len(errors) == 1
            assert str(errors.first().render()) == "broken"

    asyncio.run(_run())


def test_toolbar_shows_the_working_directory_after_the_effort() -> None:
    """The toolbar renders the working directory right after the effort field."""

    async def _run() -> None:
        app = _app()
        async with app.run_test() as pilot:
            await pilot.pause()

            text = str(app.query_one("#toolbar", Static).render())
            assert f"effort: medium  ·  {app.working_directory}" in text

    asyncio.run(_run())


def test_set_working_directory_updates_toolbar(tmp_path: Path) -> None:
    """Moving the working directory redraws the toolbar with the new path."""

    async def _run() -> None:
        app = _app()
        async with app.run_test() as pilot:
            app.set_working_directory(tmp_path)
            await pilot.pause()

            text = str(app.query_one("#toolbar", Static).render())
            assert f"·  {tmp_path.resolve()}  ·" in text

    asyncio.run(_run())


def test_set_displayed_model_id_updates_toolbar() -> None:
    """Updating the displayed model id refreshes the cached label and toolbar."""

    async def _run() -> None:
        app = _app()
        async with app.run_test() as pilot:
            app.set_displayed_model_id("openai/gpt-5")
            await pilot.pause()

            assert app._model_id == "openai/gpt-5"
            assert "openai/gpt-5" in str(app.query_one("#toolbar", Static).render())

    asyncio.run(_run())
