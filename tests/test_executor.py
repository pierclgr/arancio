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
    tmp_path: Path,
) -> ActionExecutor:
    """Return an executor over a real, configured, open session.

    Args:
        agent: the agent prompts are sent to.
        app: the fake app commands act on.
        settings_manager: the manager holding a usable provider and model.
        session_manager: the manager owning the open chat.
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
    )


@pytest.fixture
def bare_executor(
    agent: Agent,
    app: RecordingApp,
    settings_manager: SettingsManager,
    session_manager: SessionManager,
) -> ActionExecutor:
    """Return an executor over a configured but session-less manager.

    Args:
        agent: the agent prompts are sent to.
        app: the fake app commands act on.
        settings_manager: the manager holding a usable provider and model.
        session_manager: the manager with no session created yet.

    Returns:
        An executor whose session doesn't exist until ``ensure_session`` or
        ``execute`` creates one.
    """
    data = Settings.default().to_dict()
    data.update(provider="openai", model_name="gpt-4o")
    storage_module.ARANCIO_SETTINGS_FILE.write_text(yaml.safe_dump(data))
    settings_manager.load()
    return ActionExecutor(
        agent=agent,
        application=app,
        settings_manager=settings_manager,
        session_manager=session_manager,
    )


def test_ensure_session_creates_one_only_on_first_call(
    bare_executor: ActionExecutor, session_manager: SessionManager
) -> None:
    """A no-op once a session exists, so a caller never has to check first."""
    assert session_manager.current is None

    bare_executor.ensure_session()
    first = session_manager.get_current_session()

    bare_executor.ensure_session()

    assert session_manager.get_current_session() is first


def test_running_an_action_from_a_session_less_executor_needs_ensure_session_first(
    bare_executor: ActionExecutor, session_manager: SessionManager
) -> None:
    """This is what ``App._run_agent`` does before resolving the prompt.

    ``execute`` alone still assumes a session exists (only ``_run_agent``'s ``except``
    branch and ``execute`` both need one, so the check has to run before either);
    calling ``ensure_session`` first is what makes both safe.
    """
    assert session_manager.current is None

    bare_executor.ensure_session()
    list(
        bare_executor.execute(
            CommandAction(name="effort", args=["low"], raw_input="/effort low")
        )
    )

    assert session_manager.current is not None
    assert any(
        record.get("type") == "state_changed" for record in _records(session_manager)
    )


def _records(session_manager: SessionManager) -> List[dict]:
    """Read every record written to the open session's log.

    Args:
        session_manager: the manager owning the open chat.

    Returns:
        One decoded record per line.
    """
    path = session_manager.get_current_session().path
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


def test_an_unknown_action_type_is_refused(executor: ActionExecutor) -> None:
    """Dispatch is exhaustive by design, so a new action cannot pass silently."""

    class _Unknown(Message):
        """A message standing in for an action type nobody handles."""

    with pytest.raises(ValueError):
        executor.execute(_Unknown(content=""))
