"""Tests for the layer that turns an action into messages and session writes.

The executor is where "core saves nothing" becomes true: the agent yields messages, and
this is the only place that decides which of them reach the log, and with what
visibility. Everything below runs against a real session on ``tmp_path``, because the
point is what actually lands in the file.
"""

import json
from pathlib import Path
from typing import List

import pytest
import yaml
from fakes import RecordingApp, ScriptedClient

import arancio.storage.manager as storage_module
from arancio.core.agents import Agent
from arancio.core.hooks.manager import HookManager
from arancio.core.messages import (
    AssistantMessage,
    ErrorMessage,
    Message,
    ToolCallMessage,
    ToolResultMessage,
)
from arancio.prompt.actions.executor import ActionExecutor
from arancio.prompt.actions.types import (
    CommandAction,
    PromptAction,
    ShellCommandAction,
)
from arancio.sessions.manager import SessionManager
from arancio.sessions.session import SessionConfiguration
from arancio.settings.manager import SettingsManager
from arancio.settings.settings import Settings
from arancio.storage.manager import StorageManager


@pytest.fixture
def app(tmp_path: Path) -> RecordingApp:
    """Return a fake app rooted in the test's temporary directory.

    Args:
        tmp_path: pytest's per-test temporary directory.

    Returns:
        A recording app.
    """
    return RecordingApp(working_directory=tmp_path)


@pytest.fixture
def executor(
    agent: Agent,
    app: RecordingApp,
    settings_manager: SettingsManager,
    session_manager: SessionManager,
    hook_manager: HookManager,
    tmp_path: Path,
) -> ActionExecutor:
    """Return an executor over a real, configured, open session.

    Args:
        agent: the agent prompts are sent to.
        app: the fake app commands act on.
        settings_manager: the manager holding a usable provider and model.
        session_manager: the manager owning the open chat.
        hook_manager: the hook manager the executor's own tools dispatch
            through.
        tmp_path: pytest's per-test temporary directory.

    Returns:
        An executor ready to run an action.
    """
    data = Settings.default().to_dict()
    data.update(provider="openai", model_name="gpt-4o")
    storage_module.ARANCIO_SETTINGS_FILE.write_text(yaml.safe_dump(data))
    settings_manager.load()
    session_manager.create(
        working_directory=tmp_path,
        configuration=SessionConfiguration.from_settings(settings_manager.settings),
    )
    return ActionExecutor(
        agent=agent,
        application=app,
        settings_manager=settings_manager,
        session_manager=session_manager,
        hook_manager=hook_manager,
    )


def test_the_first_action_is_what_puts_the_session_on_disk(
    executor: ActionExecutor,
    session_manager: SessionManager,
    storage_manager: StorageManager,
) -> None:
    """A session exists from startup, but an untouched one stays in memory."""
    assert session_manager.current.created_on_disk is False
    assert list(storage_manager.root.rglob("*.jsonl")) == []

    action = CommandAction(name="effort", args=["low"], raw_input="/effort low")
    list(executor.execute(action))

    assert session_manager.current.created_on_disk is True
    records = _records(session_manager)
    assert records[0]["type"] == "session_created"
    assert any(record.get("type") == "state_changed" for record in records)


@pytest.mark.parametrize(
    "name, args",
    [
        ("quit", []),
        ("exit", []),
        ("clear", []),
        ("new", []),
        ("fork", []),
        ("resume", ["nosuchsession"]),
    ],
)
def test_no_session_discarding_command_creates_the_log(
    executor: ActionExecutor,
    storage_manager: StorageManager,
    name: str,
    args: List[str],
) -> None:
    """A command that ends or switches an unused chat must leave nothing behind.

    Every alias is covered, and so is a ``/resume`` that fails: its error is shown but
    never saved, since there is no chat worth saving it into.
    """
    raw_input = " ".join([f"/{name}", *args])

    list(executor.execute(CommandAction(name=name, args=args, raw_input=raw_input)))

    assert list(storage_manager.root.rglob("*.jsonl")) == []


def test_forking_an_untouched_chat_switches_without_writing(
    executor: ActionExecutor,
    session_manager: SessionManager,
    storage_manager: StorageManager,
) -> None:
    """The fork still happens; a fork of nothing simply has no log to write."""
    source = session_manager.current

    produced = list(
        executor.execute(CommandAction(name="fork", args=[], raw_input="/fork"))
    )

    forked = session_manager.current
    assert forked is not source
    assert forked.forked_from == source.id
    assert produced[0].content == f"Session {source.id} forked to {forked.id}"
    assert list(storage_manager.root.rglob("*.jsonl")) == []


def test_a_forked_chat_records_the_confirmation_into_both_logs(
    executor: ActionExecutor,
    session_manager: SessionManager,
) -> None:
    """The fork's confirmation must be readable from either session's replay.

    ``ForkCommand`` writes its own copy into the source; the executor's generic post-
    command write is what puts the same text into the fork, since that chat is current
    by the time the executor records the result.
    """
    list(executor.execute(CommandAction(name="effort", args=["low"], raw_input="/e")))
    source = session_manager.current

    produced = list(
        executor.execute(CommandAction(name="fork", args=[], raw_input="/fork"))
    )

    forked = session_manager.current
    confirmation = produced[0].content
    assert confirmation in [event.record.get("content") for event in source.events]
    assert confirmation in [event.record.get("content") for event in forked.events]


@pytest.mark.parametrize("name", ["quit", "clear"])
def test_a_discarding_command_is_recorded_once_the_chat_has_a_log(
    executor: ActionExecutor, session_manager: SessionManager, name: str
) -> None:
    """The rule spares an unused chat only; a real one must keep the typed line."""
    list(executor.execute(CommandAction(name="effort", args=["low"], raw_input="/e")))
    chat = session_manager.current

    list(executor.execute(CommandAction(name=name, args=[], raw_input=f"/{name}")))

    records = [json.loads(line) for line in chat.path.read_text().splitlines() if line]
    assert any(
        record.get("type") == "command" and record.get("raw_input") == f"/{name}"
        for record in records
    )


def _records(session_manager: SessionManager) -> List[dict]:
    """Read every record written to the open session's log.

    Args:
        session_manager: the manager owning the open chat.

    Returns:
        One decoded record per line.
    """
    path = session_manager.current.path
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def test_a_shell_command_runs_without_asking_permission(
    executor: ActionExecutor,
) -> None:
    """Typing ``!`` is the consent, so the permission manager is not consulted."""
    produced = list(
        executor.execute(
            ShellCommandAction(
                command="echo hi", add_to_history=True, raw_input="!echo hi"
            )
        )
    )

    call = produced[0]
    result = produced[1]
    assert isinstance(call, ToolCallMessage)
    assert call.name == "ShellCommandTool"
    assert isinstance(result, ToolResultMessage)
    assert "hi" in result.display_text


def test_a_remembered_shell_command_is_attributed_to_the_user(
    executor: ActionExecutor, session_manager: SessionManager, agent: Agent
) -> None:
    """Without the attribution line the model reads the call as its own idea.

    The line is saved but hidden: the user never typed it, so replaying it into
    the log would show them something they did not write.
    """
    list(
        executor.execute(
            ShellCommandAction(
                command="echo hi", add_to_history=True, raw_input="!echo hi"
            )
        )
    )

    attribution = [
        record
        for record in _records(session_manager)
        if record.get("content") == "User explicitly ran the following command:"
    ]
    assert len(attribution) == 1
    assert attribution[0]["visible"] is False
    assert attribution[0]["in_history"] is True
    assert session_manager.model_history()[0].content == (
        "User explicitly ran the following command:"
    )


def test_a_hidden_shell_command_stays_out_of_model_history(
    executor: ActionExecutor, session_manager: SessionManager
) -> None:
    """``!!`` is for the user alone, so nothing about it reaches the model."""
    list(
        executor.execute(
            ShellCommandAction(
                command="echo hi", add_to_history=False, raw_input="!!echo hi"
            )
        )
    )

    assert session_manager.model_history() == []
    assert any(record["type"] == "command" for record in _records(session_manager))


def test_both_shell_forms_record_the_line_that_was_typed(
    executor: ActionExecutor, session_manager: SessionManager
) -> None:
    """The log replays what the user wrote, not the synthesized tool call."""
    list(
        executor.execute(
            ShellCommandAction(
                command="echo hi", add_to_history=False, raw_input="!!echo hi"
            )
        )
    )

    command = next(
        record for record in _records(session_manager) if record["type"] == "command"
    )
    assert command["raw_input"] == "!!echo hi"
    assert command["name"] == "shell"
    assert command["args"] == ["echo hi"]


def test_a_prompt_records_the_user_message_and_forwards_the_stream(
    executor: ActionExecutor, session_manager: SessionManager, client: ScriptedClient
) -> None:
    """The executor saves the prompt itself, then passes the agent through whole.

    The agent never yields the opening message back, so if the executor did not record
    it here it would be missing from the replayed chat.
    """
    client.turns = [[AssistantMessage(content="the answer")]]

    produced = list(
        executor.execute(PromptAction(prompt="a question", raw_input="a question"))
    )

    assert [m.content for m in produced] == ["the answer"]
    contents = [
        record.get("content")
        for record in _records(session_manager)
        if record["type"] in {"user", "assistant"}
    ]
    assert contents == ["a question", "the answer"]


def test_a_prompt_without_a_model_is_an_error_not_a_crash(
    agent: Agent,
    app: RecordingApp,
    settings_manager: SettingsManager,
    session_manager: SessionManager,
    hook_manager: HookManager,
    tmp_path: Path,
) -> None:
    """A first run has no provider yet, and typing must still be safe."""
    session_manager.create(
        working_directory=tmp_path,
        configuration=SessionConfiguration.from_settings(settings_manager.settings),
    )
    executor = ActionExecutor(
        agent=agent,
        application=app,
        settings_manager=settings_manager,
        session_manager=session_manager,
        hook_manager=hook_manager,
    )

    produced = list(executor.execute(PromptAction(prompt="hi", raw_input="hi")))

    assert isinstance(produced[0], ErrorMessage)
    assert produced[0].content.startswith("Error sending message:")


def test_an_unknown_command_is_an_error_message(executor: ActionExecutor) -> None:
    """A typo must not take the app down mid-session."""
    produced = list(
        executor.execute(CommandAction(name="nope", args=[], raw_input="/nope"))
    )

    assert isinstance(produced[0], ErrorMessage)
    assert produced[0].content == "Command not found: nope"


def test_a_command_that_raises_is_an_error_message(executor: ActionExecutor) -> None:
    """The failure is reported with the command's name and its own wording."""
    produced = list(
        executor.execute(
            CommandAction(name="cd", args=["nowhere"], raw_input="/cd nowhere")
        )
    )

    assert isinstance(produced[0], ErrorMessage)
    assert produced[0].content.startswith("Error while executing command cd:")


def test_a_command_result_is_shown_but_not_sent_to_the_model(
    executor: ActionExecutor,
) -> None:
    """Confirmation text is for the user; the model did not ask for it."""
    produced = list(
        executor.execute(
            CommandAction(
                name="hello-world", args=["Ada"], raw_input="/hello-world Ada"
            )
        )
    )

    message = produced[0]
    assert isinstance(message, AssistantMessage)
    assert message.content == "Hello World, Ada"
    assert message.in_history is False


def test_changing_the_directory_records_it_on_the_session(
    executor: ActionExecutor, session_manager: SessionManager, tmp_path: Path
) -> None:
    """A resumed chat has to reopen where the user left it."""
    (tmp_path / "sub").mkdir()

    list(executor.execute(CommandAction(name="cd", args=["sub"], raw_input="/cd sub")))

    moved = next(
        record
        for record in _records(session_manager)
        if record["type"] == "state_changed"
    )
    assert moved["working_directory"] == str((tmp_path / "sub").resolve())


@pytest.mark.parametrize(
    ("name", "args"),
    [
        ("model", ["gpt-5"]),
        ("provider", ["anthropic"]),
        ("effort", ["low"]),
        ("permissions", ["read", "auto"]),
    ],
)
def test_a_configuration_command_snapshots_the_session(
    executor: ActionExecutor,
    session_manager: SessionManager,
    name: str,
    args: List[str],
) -> None:
    """These four are what a resumed session restores, so each must be saved."""
    list(executor.execute(CommandAction(name=name, args=args, raw_input=f"/{name}")))

    types = [record["type"] for record in _records(session_manager)]
    assert "state_changed" in types


def test_reading_a_permission_changes_no_session_state(
    executor: ActionExecutor, session_manager: SessionManager
) -> None:
    """Only a mutating ``/permissions`` is a configuration change."""
    list(
        executor.execute(
            CommandAction(
                name="permissions", args=["read"], raw_input="/permissions read"
            )
        )
    )

    types = [record["type"] for record in _records(session_manager)]
    assert "state_changed" not in types


def test_only_declared_parameters_are_injected(executor: ActionExecutor) -> None:
    """A command receives the running objects it named, and nothing else.

    ``/permissions`` declares no ``application``, so passing one would be a
    ``TypeError`` from its own signature.
    """
    produced = list(
        executor.execute(
            CommandAction(
                name="permissions", args=["read"], raw_input="/permissions read"
            )
        )
    )

    assert produced[0].content == "read permission level: ask"


def test_extra_prompt_words_are_dropped(executor: ActionExecutor) -> None:
    """Binding is positional, so a stray word must not become an argument."""
    produced = list(
        executor.execute(
            CommandAction(
                name="hello-world", args=["Ada", "2", "extra"], raw_input="/hello-world"
            )
        )
    )

    assert produced[0].content == "Hello World, Ada\nHello World, Ada"


def test_a_joining_command_receives_every_word_as_one_argument(
    executor: ActionExecutor, session_manager: SessionManager
) -> None:
    """``/rename`` joins its words, so a spaced name is not cut to its first word."""
    produced = list(
        executor.execute(
            CommandAction(
                name="rename",
                args=["my", "new", "name"],
                raw_input="/rename my new name",
            )
        )
    )

    assert produced[0].content == "Session renamed to 'my new name'"
    assert session_manager.current.name == "my new name"


def test_a_joining_command_without_words_still_reports_the_missing_argument(
    executor: ActionExecutor,
) -> None:
    """Joining nothing must not bind an empty name in place of the error."""
    produced = list(
        executor.execute(CommandAction(name="rename", args=[], raw_input="/rename"))
    )

    assert isinstance(produced[0], ErrorMessage)


def test_an_unknown_action_type_is_refused(executor: ActionExecutor) -> None:
    """Dispatch is exhaustive by design, so a new action cannot pass silently."""

    class _Unknown(Message):
        """A message standing in for an action type nobody handles."""

    with pytest.raises(ValueError):
        executor.execute(_Unknown(content=""))
