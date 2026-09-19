"""End-to-end walks with everything real except the model and the user.

These are the tests the old suite did not have. Each layer below is checked on its own
elsewhere; what these confirm is that a whole turn, written by the executor and read
back by a second session manager, reconstructs both what the user saw and what the model
saw.
"""

import json
from pathlib import Path
from typing import List

import pytest
import yaml
from fakes import RecordingApp, ScriptedClient, ScriptedController

import arancio.storage.manager as storage_module
from arancio.core.agents import Agent
from arancio.core.messages import (
    AssistantMessage,
    Message,
    ToolCallMessage,
    ToolErrorMessage,
    UserMessage,
)
from arancio.core.permissions.manager import PermissionManager
from arancio.core.tools.manager import ToolManager
from arancio.core.tools.session import ToolSession, shared_session
from arancio.prompt.actions.executor import ActionExecutor
from arancio.prompt.actions.types import CommandAction, PromptAction, ShellCommandAction
from arancio.sessions.manager import SessionManager
from arancio.sessions.session import SessionConfiguration
from arancio.settings.manager import SettingsManager
from arancio.storage.manager import StorageManager


@pytest.fixture
def wired(
    tmp_path: Path,
) -> tuple[ActionExecutor, SessionManager, StorageManager, ScriptedClient]:
    """Wire the real object graph the way ``main`` does, minus the TUI.

    Args:
        tmp_path: pytest's per-test temporary directory.

    Returns:
        The executor, the session manager, the storage manager and the
        scripted client standing in for the model.
    """
    storage_manager = StorageManager(root=tmp_path / "arancio")
    session_manager = SessionManager(storage_manager, root=storage_manager.root)
    controller = ScriptedController()
    client = ScriptedClient()
    tool_manager = ToolManager(web_summary_client=client)
    permission_manager = PermissionManager(
        tool_manager=tool_manager, controller=controller
    )
    agent = Agent(client=client, permission_manager=permission_manager)
    settings_manager = SettingsManager(
        storage_manager=storage_manager,
        client=client,
        summary_client=client,
        agent=agent,
    )
    data = {**yaml.safe_load(_default_settings()), "provider": "openai"}
    data["model_name"] = "gpt-4o"
    storage_module.ARANCIO_SETTINGS_FILE.write_text(yaml.safe_dump(data))
    settings_manager.load()
    session_manager.create(
        working_directory=tmp_path,
        configuration=SessionConfiguration.from_settings(settings_manager.settings),
    )
    executor = ActionExecutor(
        agent=agent,
        application=RecordingApp(working_directory=tmp_path),
        settings_manager=settings_manager,
        session_manager=session_manager,
    )
    return executor, session_manager, storage_manager, client


def _default_settings() -> str:
    """Return the default settings serialized as YAML.

    Returns:
        The YAML text of the default settings.
    """
    from arancio.settings.settings import Settings

    return yaml.safe_dump(Settings.default().to_dict())


def _reopen(
    session_manager: SessionManager, storage_manager: StorageManager
) -> SessionManager:
    """Reopen the current session from disk with a fresh manager.

    Args:
        session_manager: the manager whose open session is reopened.
        storage_manager: the storage manager holding the sessions root.

    Returns:
        A second manager that read the log back.
    """
    session_id = session_manager.require_current().id
    reopened = SessionManager(
        storage_manager, root=storage_manager.root, tool_session=ToolSession()
    )
    reopened.load(session_id)
    return reopened


def _contents(messages: List[Message]) -> List[str]:
    """Return the content of each message.

    Args:
        messages: the messages to read.

    Returns:
        One content string per message.
    """
    return [message.content for message in messages]


def test_a_whole_turn_replays_for_both_the_user_and_the_model(
    wired: tuple[ActionExecutor, SessionManager, StorageManager, ScriptedClient],
    tmp_path: Path,
) -> None:
    """A prompt, an approved tool call and a reply come back intact.

    This is the path everything else exists to support: the executor records
    the prompt, the agent's stream is recorded as it passes, the file the tool
    read is derived from the shared guard, and a second manager rebuilds all of
    it from the log alone.
    """
    executor, session_manager, storage_manager, client = wired
    target = tmp_path / "notes.txt"
    target.write_text("alpha\n")
    client.turns = [
        [
            ToolCallMessage(
                content="",
                id="c1",
                name="ReadFileTool",
                arguments={"file_path": str(target)},
            )
        ],
        [AssistantMessage(content="the file says alpha")],
    ]

    produced = list(
        executor.execute(PromptAction(prompt="what is in notes?", raw_input="@notes"))
    )

    assert isinstance(produced[0], ToolCallMessage)
    assert "alpha" in produced[1].display_text
    assert produced[2] == AssistantMessage(content="the file says alpha")

    reopened = _reopen(session_manager, storage_manager)

    assert _contents(reopened.model_history()) == [
        "what is in notes?",
        "",
        produced[1].content,
        "the file says alpha",
    ]
    assert reopened.require_current().file_states[str(target.resolve())] == (
        target.stat().st_mtime
    )


def test_the_read_guard_comes_back_only_for_files_that_did_not_change(
    wired: tuple[ActionExecutor, SessionManager, StorageManager, ScriptedClient],
    tmp_path: Path,
) -> None:
    """Resuming must not re-arm a read for a file edited while we were away."""
    executor, session_manager, storage_manager, client = wired
    target = tmp_path / "notes.txt"
    target.write_text("alpha\n")
    client.turns = [
        [
            ToolCallMessage(
                content="",
                id="c1",
                name="ReadFileTool",
                arguments={"file_path": str(target)},
            )
        ],
        [AssistantMessage(content="done")],
    ]
    list(executor.execute(PromptAction(prompt="read it", raw_input="read it")))

    reopened = _reopen(session_manager, storage_manager)
    reopened.restore_file_states()
    guard = reopened._tool_session
    assert guard.is_known(str(target.resolve())) is True

    target.write_text("edited by someone else\n")
    changed = _reopen(session_manager, storage_manager)
    changed.restore_file_states()

    assert changed._tool_session.is_known(str(target.resolve())) is False


def test_a_shell_command_and_a_slash_command_replay_differently(
    wired: tuple[ActionExecutor, SessionManager, StorageManager, ScriptedClient],
) -> None:
    """The two asymmetries in the log, checked together in one session.

    The ``!`` attribution line goes to the model but not the replayed log; the ``/``
    command line goes to the replayed log but not the model.
    """
    executor, session_manager, storage_manager, _ = wired

    list(
        executor.execute(
            ShellCommandAction(
                command="echo hi", add_to_history=True, raw_input="!echo hi"
            )
        )
    )
    list(
        executor.execute(
            CommandAction(name="effort", args=["low"], raw_input="/effort low")
        )
    )

    reopened = _reopen(session_manager, storage_manager)
    history = _contents(reopened.model_history())
    visible = _contents(reopened.visible_messages())

    assert "User explicitly ran the following command:" in history
    assert "User explicitly ran the following command:" not in visible
    assert "!echo hi" in visible
    assert "/effort low" in visible
    assert "/effort low" not in history


def test_a_session_that_stopped_mid_tool_call_is_closed_on_reopen(
    wired: tuple[ActionExecutor, SessionManager, StorageManager, ScriptedClient],
    tmp_path: Path,
) -> None:
    """Otherwise the resumed model waits forever on an answer that never comes."""
    executor, session_manager, storage_manager, _ = wired
    session_manager.session_recorder.message(UserMessage(content="do the thing"))
    session_manager.session_recorder.message(
        ToolCallMessage(
            content="",
            id="interrupted",
            name="ShellCommandTool",
            arguments={"command": "sleep 100"},
        )
    )

    reopened = _reopen(session_manager, storage_manager)
    history = reopened.model_history()

    closing = history[-1]
    assert isinstance(closing, ToolErrorMessage)
    assert closing.id == "interrupted"
    assert "may or may not have completed" in closing.content


def test_reopening_restores_the_last_state_changed_snapshot(
    wired: tuple[ActionExecutor, SessionManager, StorageManager, ScriptedClient],
    tmp_path: Path,
) -> None:
    """A later ``state_changed`` record supersedes an earlier one, atomically.

    ``/cd`` and ``/effort`` each snapshot both configuration and working directory
    together, so the session reopened from disk must reflect the directory moved by the
    first command and the effort set by the second, not a mix of the two records or the
    earlier one.
    """
    executor, session_manager, storage_manager, _ = wired
    (tmp_path / "sub").mkdir()

    list(executor.execute(CommandAction(name="cd", args=["sub"], raw_input="/cd sub")))
    list(
        executor.execute(
            CommandAction(name="effort", args=["low"], raw_input="/effort low")
        )
    )

    reopened = _reopen(session_manager, storage_manager)
    restored = reopened.require_current()

    assert restored.working_directory == (tmp_path / "sub").resolve()
    assert restored.configuration.thinking_effort == "low"


@pytest.mark.parametrize("command_name", ["clear", "new"])
def test_clearing_a_chat_starts_a_separate_log(
    command_name: str,
    wired: tuple[ActionExecutor, SessionManager, StorageManager, ScriptedClient],
) -> None:
    """A new chat must not inherit the old one's messages or its read guard.

    Runs once per registered name of the clear command, aliases included.
    """
    executor, session_manager, storage_manager, _ = wired
    first = session_manager.require_current()
    session_manager.session_recorder.message(UserMessage(content="old chat"))
    shared_session.record_read("/tmp/earlier.txt", 1.0)
    action = CommandAction(name=command_name, args=[], raw_input=f"/{command_name}")

    list(executor.execute(action))

    second = session_manager.require_current()
    assert second.id != first.id
    assert second.path != first.path
    assert shared_session.is_known("/tmp/earlier.txt") is False
    records = [json.loads(line) for line in first.path.read_text().splitlines()]
    assert any(record.get("content") == "old chat" for record in records)
    assert any(record.get("raw_input") == f"/{command_name}" for record in records)


@pytest.mark.parametrize("command_name", ["clear", "new"])
def test_a_cleared_chat_opens_empty(
    command_name: str,
    wired: tuple[ActionExecutor, SessionManager, StorageManager, ScriptedClient],
) -> None:
    """A new chat starts with nothing in it, including no trace of the command.

    Runs once per registered name of the clear command, aliases included.
    """
    executor, session_manager, _, _ = wired
    action = CommandAction(name=command_name, args=[], raw_input=f"/{command_name}")

    list(executor.execute(action))

    assert _contents(session_manager.visible_messages()) == []
