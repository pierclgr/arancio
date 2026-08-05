"""Tests for the action executor."""

from collections.abc import Iterator
from pathlib import Path

import pytest

from arancio.core.agents import Agent
from arancio.core.clients.base import BaseClient
from arancio.core.clients.litellm import LiteLLMClient
from arancio.core.controllers.requests import BaseControllerRequest
from arancio.core.controllers.responses import (
    BaseControllerResponse,
    Decision,
    PermissionResponse,
)
from arancio.core.messages import (
    AssistantMessage,
    ErrorMessage,
    Message,
    ToolCallMessage,
    ToolErrorMessage,
    ToolResultMessage,
    UserMessage,
)
from arancio.core.permissions.manager import PermissionManager
from arancio.core.permissions.types import PermissionCategory, PermissionLevel
from arancio.core.requests import BaseRequest
from arancio.core.tools.manager import ToolManager
from arancio.core.tools.session import default_session
from arancio.prompt.actions.executor import ActionExecutor
from arancio.prompt.actions.factory import ActionFactory
from arancio.prompt.actions.types import CommandAction, PromptAction, ShellCommandAction
from arancio.settings.settings import Settings


@pytest.fixture(autouse=True)
def _reset_default_tool_session() -> None:
    """Clear the shared tool session before each test for isolation."""
    default_session.clear()


class _DummyAgent:
    """Stand-in agent; command-action tests never run its loop.

    Also records whether its history was asked to be cleared, for the clear command
    action test.
    """

    def __init__(self) -> None:
        """Start with no recorded history clear."""
        self.clear_history_called = False

    def clear_history(self) -> None:
        """Record that the history was asked to be cleared."""
        self.clear_history_called = True


class _RecordingApplication:
    """Application stub recording exit/clear/model/effort display requests."""

    def __init__(self) -> None:
        """Start with no recorded exit, clear, displayed model id or effort."""
        self.exit_called = False
        self.clear_log_called = False
        self.displayed_model_id: str | None = None
        self.displayed_effort: str | None = None

    def exit(self) -> None:
        """Record that an exit was requested."""
        self.exit_called = True

    def clear_log(self) -> None:
        """Record that the log was asked to be cleared."""
        self.clear_log_called = True

    def set_displayed_model_id(self, model_id: str) -> None:
        """Record the model id the toolbar was asked to display.

        Args:
            model_id: the model id to display.
        """
        self.displayed_model_id = model_id

    def set_displayed_effort(self, effort: str | None) -> None:
        """Record the effort the toolbar was asked to display.

        Args:
            effort: the effort to display, or ``None`` when thinking is
                disabled.
        """
        self.displayed_effort = effort


class _RecordingSettingsManager:
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


class _RecordingAgent:
    """Agent stub recording the message/prelude it runs and yielding a fixed reply."""

    def __init__(self, reply: Message) -> None:
        """Store the reply each run yields.

        Args:
            reply: the message produced for every ``run`` call.
        """
        self._reply = reply
        self.received: Message | None = None
        self.received_prelude: list[Message] | None = None
        self.added_messages: list[Message] = []

    def add_message_to_history(self, message: Message) -> None:
        """Record a message explicitly appended outside ``run``.

        Args:
            message: the message added to agent history.
        """
        self.added_messages.append(message)

    def run(
        self, message: Message, prelude: list[Message] | None = None
    ) -> Iterator[Message]:
        """Record the input message/prelude and yield them then the reply.

        Args:
            message: the user message that starts the turn.
            prelude: messages to yield before the configured reply.

        Yields:
            Each ``prelude`` message, then the configured reply message.
        """
        self.received = message
        self.received_prelude = prelude
        yield from prelude or []
        yield self._reply


def test_factory_creates_command_action() -> None:
    """The factory builds a command action from a name and arguments."""
    action = ActionFactory.create_command_action("hello-world", ["Sam", "1"])

    assert action == CommandAction(name="hello-world", args=["Sam", "1"])


def test_factory_creates_hidden_shell_command_action() -> None:
    """The factory builds a non-history shell action for ``!!`` input."""
    action = ActionFactory.create_shell_command_action("echo hello", False)

    assert action == ShellCommandAction(command="echo hello", add_to_history=False)


def test_factory_creates_prompt_action() -> None:
    """The factory builds a prompt action from the raw prompt."""
    action = ActionFactory.create_prompt_action("normal prompt")

    assert action == PromptAction(prompt="normal prompt")


def test_factory_create_action_dispatches_to_command_action() -> None:
    """create_action builds a command action when given a command name."""
    action = ActionFactory.create_action(
        command_name="hello-world", command_args=["Sam", "1"]
    )

    assert action == CommandAction(name="hello-world", args=["Sam", "1"])


def test_factory_create_action_dispatches_to_shell_command_action() -> None:
    """create_action builds a shell action when given shell-command arguments."""
    action = ActionFactory.create_action(
        shell_command="echo hello", add_to_history=False
    )

    assert action == ShellCommandAction(command="echo hello", add_to_history=False)


def test_factory_create_action_dispatches_to_prompt_action() -> None:
    """create_action builds a prompt action when no name is given."""
    action = ActionFactory.create_action(prompt="normal prompt")

    assert action == PromptAction(prompt="normal prompt")


def test_execute_command_action_runs_command() -> None:
    """A command action runs its command and yields the result message."""
    executor = ActionExecutor(
        agent=_DummyAgent(),
        application=_RecordingApplication(),
        settings_manager=_RecordingSettingsManager(),
    )
    action = CommandAction(name="hello-world", args=["Sam", "1"])

    messages = list(executor.execute(action))

    assert messages == [AssistantMessage(content="Hello World, Sam")]


def test_execute_hidden_shell_command_yields_paired_call_and_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A hidden shell action yields a matching tool call and direct result pair."""
    agent = _RecordingAgent(AssistantMessage(content="unused"))
    executor = ActionExecutor(
        agent=agent,
        application=_RecordingApplication(),
        settings_manager=_RecordingSettingsManager(provider=None, model_name=None),
    )
    recorded: dict = {}

    def call_shell(call_id: str, **kwargs) -> ToolResultMessage:
        recorded.update(call_id=call_id, **kwargs)
        return ToolResultMessage(content="output", id=call_id)

    monkeypatch.setattr(executor._shell_command_tool, "call", call_shell)

    messages = list(
        executor.execute(
            ShellCommandAction(command="echo @notes.md", add_to_history=False)
        )
    )

    call, result = messages
    assert isinstance(call, ToolCallMessage)
    assert call.name == "ShellCommandTool"
    assert call.arguments == {"command": "echo @notes.md"}
    assert call.id == recorded["call_id"]
    assert recorded["command"] == "echo @notes.md"
    assert result == ToolResultMessage(content="output", id=call.id)
    assert agent.received is None
    assert agent.added_messages == []


def test_execute_history_shell_command_stores_paired_call_and_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A ``!`` action attributes the stored call and result to the user."""
    agent = _RecordingAgent(AssistantMessage(content="unused"))
    executor = ActionExecutor(
        agent=agent,
        application=_RecordingApplication(),
        settings_manager=_RecordingSettingsManager(provider=None, model_name=None),
    )

    def call_shell(call_id: str, **kwargs) -> ToolResultMessage:
        return ToolResultMessage(content={"stdout": "ok"}, id=call_id)

    monkeypatch.setattr(executor._shell_command_tool, "call", call_shell)

    messages = list(
        executor.execute(
            ShellCommandAction(command="echo /clear @notes.md", add_to_history=True)
        )
    )

    call, result = messages
    assert call.arguments == {"command": "echo /clear @notes.md"}
    assert result.id == call.id
    assert agent.added_messages == [
        UserMessage(content="User explicitly ran the following command:"),
        call,
        result,
    ]
    assert agent.received is None


def test_execute_history_shell_command_stores_tool_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed ``!`` command stores its paired tool error for the model."""
    agent = _RecordingAgent(AssistantMessage(content="unused"))
    executor = ActionExecutor(
        agent=agent,
        application=_RecordingApplication(),
        settings_manager=_RecordingSettingsManager(),
    )

    def call_shell(call_id: str, **kwargs) -> ToolErrorMessage:
        return ToolErrorMessage(content="failed", id=call_id)

    monkeypatch.setattr(executor._shell_command_tool, "call", call_shell)

    messages = list(
        executor.execute(ShellCommandAction(command="exit 1", add_to_history=True))
    )

    assert isinstance(messages[1], ToolErrorMessage)
    assert agent.added_messages == [
        UserMessage(content="User explicitly ran the following command:"),
        *messages,
    ]


def test_execute_command_action_ignores_extra_prompt_words() -> None:
    """A prompt word beyond the command's parameters is silently dropped."""
    executor = ActionExecutor(
        agent=_DummyAgent(),
        application=_RecordingApplication(),
        settings_manager=_RecordingSettingsManager(),
    )
    action = CommandAction(name="hello-world", args=["Sam", "1", "extra"])

    messages = list(executor.execute(action))

    assert messages == [AssistantMessage(content="Hello World, Sam")]


def test_execute_unknown_command_yields_error() -> None:
    """An unknown command name yields an error message."""
    executor = ActionExecutor(
        agent=_DummyAgent(),
        application=_RecordingApplication(),
        settings_manager=_RecordingSettingsManager(),
    )

    messages = list(executor.execute(CommandAction(name="nope", args=[])))

    assert messages == [ErrorMessage(content="Command not found: nope")]


def test_execute_command_with_uncoercible_argument_yields_error() -> None:
    """An argument that cannot match its type yields an error message."""
    executor = ActionExecutor(
        agent=_DummyAgent(),
        application=_RecordingApplication(),
        settings_manager=_RecordingSettingsManager(),
    )
    action = CommandAction(name="hello-world", args=["Sam", "three"])

    (message,) = list(executor.execute(action))

    assert isinstance(message, ErrorMessage)
    assert "hello-world" in message.content


def test_execute_command_missing_argument_yields_error() -> None:
    """A command missing a mandatory argument yields an error message."""
    executor = ActionExecutor(
        agent=_DummyAgent(),
        application=_RecordingApplication(),
        settings_manager=_RecordingSettingsManager(),
    )
    action = CommandAction(name="hello-world", args=[])

    (message,) = list(executor.execute(action))

    assert isinstance(message, ErrorMessage)
    assert "hello-world" in message.content


def test_execute_exit_command_quits_the_application() -> None:
    """The exit command action exits the application and yields no message."""
    application = _RecordingApplication()
    executor = ActionExecutor(
        agent=_DummyAgent(),
        application=application,
        settings_manager=_RecordingSettingsManager(),
    )

    messages = list(executor.execute(CommandAction(name="exit", args=[])))

    assert messages == []
    assert application.exit_called is True


def test_execute_clear_command_clears_history_and_log() -> None:
    """The clear command action empties the agent's history and the app's log."""
    agent = _DummyAgent()
    application = _RecordingApplication()
    executor = ActionExecutor(
        agent=agent,
        application=application,
        settings_manager=_RecordingSettingsManager(),
    )

    messages = list(executor.execute(CommandAction(name="clear", args=[])))

    assert messages == []
    assert agent.clear_history_called is True
    assert application.clear_log_called is True


def test_execute_model_command_injects_application_and_settings_manager() -> None:
    """The model command action applies, persists and confirms the new model id."""
    application = _RecordingApplication()
    settings_manager = _RecordingSettingsManager(provider="openai", model_name="gpt-4o")
    executor = ActionExecutor(
        agent=_DummyAgent(), application=application, settings_manager=settings_manager
    )
    action = CommandAction(name="model", args=["gpt-5"])

    messages = list(executor.execute(action))

    assert messages == [AssistantMessage(content="Model set to openai/gpt-5")]
    assert settings_manager.settings.model_name == "gpt-5"
    assert settings_manager.applied is True
    assert settings_manager.saved is True
    assert application.displayed_model_id == "openai/gpt-5"


def test_execute_model_command_without_provider_yields_error() -> None:
    """A missing provider surfaces as an error message, not a crash.

    ``ModelCommand`` sets ``model_name`` then reads ``settings.model_id``, which raises
    ``ValueError`` when the provider is unset, before ``apply`` or ``save`` ever run;
    the executor's generic exception handling turns it into an :class:`ErrorMessage`,
    the same as any other command failure. ``model_name`` is restored to its previous
    value rather than left dangling.
    """
    application = _RecordingApplication()
    settings_manager = _RecordingSettingsManager(provider=None, model_name="gpt-4o")
    executor = ActionExecutor(
        agent=_DummyAgent(), application=application, settings_manager=settings_manager
    )
    action = CommandAction(name="model", args=["gpt-5"])

    (message,) = list(executor.execute(action))

    assert isinstance(message, ErrorMessage)
    assert "model" in message.content
    assert settings_manager.applied is False
    assert settings_manager.saved is False
    assert settings_manager.settings.model_name == "gpt-4o"
    assert application.displayed_model_id is None


def test_execute_provider_command_injects_application_and_settings_manager() -> None:
    """The provider command action applies, persists and refreshes the toolbar."""
    application = _RecordingApplication()
    settings_manager = _RecordingSettingsManager(provider="openai", model_name="gpt-4o")
    executor = ActionExecutor(
        agent=_DummyAgent(), application=application, settings_manager=settings_manager
    )
    action = CommandAction(name="provider", args=["anthropic"])

    messages = list(executor.execute(action))

    assert messages == [AssistantMessage(content="Provider set to anthropic")]
    assert settings_manager.settings.provider == "anthropic"
    assert settings_manager.applied is True
    assert settings_manager.saved is True
    assert application.displayed_model_id == "anthropic/gpt-4o"


def test_execute_provider_command_without_model_name_skips_toolbar_update() -> None:
    """Setting the provider alone still confirms, without a model id to display."""
    application = _RecordingApplication()
    settings_manager = _RecordingSettingsManager(provider=None, model_name=None)
    executor = ActionExecutor(
        agent=_DummyAgent(), application=application, settings_manager=settings_manager
    )
    action = CommandAction(name="provider", args=["openai"])

    messages = list(executor.execute(action))

    assert messages == [AssistantMessage(content="Provider set to openai")]
    assert settings_manager.settings.provider == "openai"
    assert settings_manager.applied is True
    assert settings_manager.saved is True
    assert application.displayed_model_id is None


def test_execute_provider_command_rejects_invalid_provider() -> None:
    """An unrecognized provider surfaces as an error message, not a crash."""
    application = _RecordingApplication()
    settings_manager = _RecordingSettingsManager(provider="openai", model_name="gpt-4o")
    executor = ActionExecutor(
        agent=_DummyAgent(), application=application, settings_manager=settings_manager
    )
    action = CommandAction(name="provider", args=["not-a-real-provider"])

    (message,) = list(executor.execute(action))

    assert isinstance(message, ErrorMessage)
    assert "provider" in message.content.lower()
    assert settings_manager.applied is False
    assert settings_manager.saved is False
    assert settings_manager.settings.provider == "openai"
    assert application.displayed_model_id is None


def test_execute_effort_command_injects_application_and_settings_manager() -> None:
    """The effort command action applies, persists and confirms the new effort."""
    application = _RecordingApplication()
    settings_manager = _RecordingSettingsManager()
    executor = ActionExecutor(
        agent=_DummyAgent(), application=application, settings_manager=settings_manager
    )
    action = CommandAction(name="effort", args=["high"])

    messages = list(executor.execute(action))

    assert messages == [AssistantMessage(content="Thinking effort set to high")]
    assert settings_manager.settings.thinking_effort == "high"
    assert settings_manager.applied is True
    assert settings_manager.saved is True
    assert application.displayed_effort == "high"


def test_execute_effort_command_null_disables_thinking() -> None:
    """The effort command action accepts "null" to disable thinking entirely."""
    application = _RecordingApplication()
    settings_manager = _RecordingSettingsManager()
    executor = ActionExecutor(
        agent=_DummyAgent(), application=application, settings_manager=settings_manager
    )
    action = CommandAction(name="effort", args=["null"])

    messages = list(executor.execute(action))

    assert messages == [AssistantMessage(content="Thinking effort set to null")]
    assert settings_manager.settings.thinking_effort is None
    assert application.displayed_effort is None


def test_execute_effort_command_without_model_yields_error() -> None:
    """Setting the effort without a configured model surfaces as an error message."""
    application = _RecordingApplication()
    settings_manager = _RecordingSettingsManager(provider=None, model_name=None)
    executor = ActionExecutor(
        agent=_DummyAgent(), application=application, settings_manager=settings_manager
    )
    action = CommandAction(name="effort", args=["high"])

    (message,) = list(executor.execute(action))

    assert isinstance(message, ErrorMessage)
    assert "effort" in message.content
    assert settings_manager.applied is False
    assert settings_manager.saved is False
    assert settings_manager.settings.thinking_effort == "medium"
    assert application.displayed_effort is None


def test_execute_permissions_command_reports_current_level() -> None:
    """The permissions command action reports the current level, unpersisted."""
    settings_manager = _RecordingSettingsManager()
    executor = ActionExecutor(
        agent=_DummyAgent(),
        application=_RecordingApplication(),
        settings_manager=settings_manager,
    )
    action = CommandAction(name="permissions", args=["read"])

    messages = list(executor.execute(action))

    assert messages == [AssistantMessage(content="read permission level: ask")]
    assert settings_manager.applied is False
    assert settings_manager.saved is False


def test_execute_permissions_command_sets_level_and_confirms() -> None:
    """The permissions command action applies and persists the new level."""
    settings_manager = _RecordingSettingsManager()
    executor = ActionExecutor(
        agent=_DummyAgent(),
        application=_RecordingApplication(),
        settings_manager=settings_manager,
    )
    action = CommandAction(name="permissions", args=["read", "auto"])

    messages = list(executor.execute(action))

    assert messages == [AssistantMessage(content="read permission level set to auto")]
    assert (
        settings_manager.settings.permissions[PermissionCategory.READ]
        == PermissionLevel.AUTO
    )
    assert settings_manager.applied is True
    assert settings_manager.saved is True


def test_execute_permissions_command_rejects_unknown_permission() -> None:
    """An unknown permission category surfaces as an error message, unpersisted."""
    settings_manager = _RecordingSettingsManager()
    executor = ActionExecutor(
        agent=_DummyAgent(),
        application=_RecordingApplication(),
        settings_manager=settings_manager,
    )
    action = CommandAction(name="permissions", args=["nope"])

    (message,) = list(executor.execute(action))

    assert isinstance(message, ErrorMessage)
    assert "permission" in message.content.lower()
    assert settings_manager.applied is False
    assert settings_manager.saved is False


def test_execute_permissions_command_rejects_unknown_level() -> None:
    """An unknown permission level surfaces as an error message, unpersisted."""
    settings_manager = _RecordingSettingsManager()
    executor = ActionExecutor(
        agent=_DummyAgent(),
        application=_RecordingApplication(),
        settings_manager=settings_manager,
    )
    action = CommandAction(name="permissions", args=["read", "nope"])

    (message,) = list(executor.execute(action))

    assert isinstance(message, ErrorMessage)
    assert "level" in message.content.lower()
    assert settings_manager.applied is False
    assert settings_manager.saved is False


def test_execute_permissions_command_null_removes_the_grant() -> None:
    """The permissions command action accepts "null" to remove a grant."""
    settings_manager = _RecordingSettingsManager()
    executor = ActionExecutor(
        agent=_DummyAgent(),
        application=_RecordingApplication(),
        settings_manager=settings_manager,
    )
    action = CommandAction(name="permissions", args=["read", "null"])

    messages = list(executor.execute(action))

    assert messages == [AssistantMessage(content="read permission removed")]
    assert (
        settings_manager.settings.permissions[PermissionCategory.READ]
        is PermissionLevel.NONE
    )
    assert settings_manager.applied is True
    assert settings_manager.saved is True


def test_execute_prompt_action_delegates_to_agent() -> None:
    """A prompt action sends its text to the model through the agent."""
    reply = AssistantMessage(content="hi")
    agent = _RecordingAgent(reply)
    executor = ActionExecutor(
        agent=agent,
        application=_RecordingApplication(),
        settings_manager=_RecordingSettingsManager(),
    )

    messages = list(executor.execute(PromptAction(prompt="hello there")))

    assert messages == [reply]
    assert agent.received == UserMessage(content="hello there")


def test_execute_prompt_action_without_provider_yields_error() -> None:
    """Sending a message without a configured provider yields an error, not a call."""
    agent = _RecordingAgent(AssistantMessage(content="hi"))
    executor = ActionExecutor(
        agent=agent,
        application=_RecordingApplication(),
        settings_manager=_RecordingSettingsManager(provider=None, model_name=None),
    )

    (message,) = list(executor.execute(PromptAction(prompt="hello there")))

    assert isinstance(message, ErrorMessage)
    assert agent.received is None


def test_execute_prompt_action_without_model_name_yields_error() -> None:
    """Sending a message without a configured model name yields an error, not a call."""
    agent = _RecordingAgent(AssistantMessage(content="hi"))
    executor = ActionExecutor(
        agent=agent,
        application=_RecordingApplication(),
        settings_manager=_RecordingSettingsManager(provider="openai", model_name=None),
    )

    (message,) = list(executor.execute(PromptAction(prompt="hello there")))

    assert isinstance(message, ErrorMessage)
    assert agent.received is None


def test_execute_prompt_action_resolves_mention_and_injects_read_pair(
    tmp_path: Path,
) -> None:
    """A resolving @mention injects a ReadFileTool call/result pair before the reply."""
    target = tmp_path / "notes.txt"
    target.write_text("hello world\n")
    reply = AssistantMessage(content="ok")
    agent = _RecordingAgent(reply)
    executor = ActionExecutor(
        agent=agent,
        application=_RecordingApplication(),
        settings_manager=_RecordingSettingsManager(),
    )
    action = PromptAction(prompt=f"see @{target}", mentions=[target])

    messages = list(executor.execute(action))

    call, result, final = messages
    assert isinstance(call, ToolCallMessage)
    assert call.name == "ReadFileTool"
    assert call.arguments == {"file_path": str(target)}
    assert isinstance(result, ToolResultMessage)
    assert result.id == call.id
    assert final == reply
    assert agent.received == UserMessage(content=f"see @{target}")
    assert agent.received_prelude == [call, result]


def test_execute_prompt_action_directory_mention_lists_via_shell(
    tmp_path: Path,
) -> None:
    """A directory @mention injects a ShellCommandTool listing call/result pair.

    Unlike a glob-based file search, this must show subdirectory names too, not just
    files, since it's meant to mirror a plain directory listing.
    """
    directory = tmp_path / "adir"
    nested = directory / "subdir"
    nested.mkdir(parents=True)
    (directory / "top.txt").write_text("top")
    reply = AssistantMessage(content="ok")
    agent = _RecordingAgent(reply)
    executor = ActionExecutor(
        agent=agent,
        application=_RecordingApplication(),
        settings_manager=_RecordingSettingsManager(),
    )
    action = PromptAction(prompt=f"see @{directory}", mentions=[directory])

    messages = list(executor.execute(action))

    call, result, final = messages
    assert isinstance(call, ToolCallMessage)
    assert call.name == "ShellCommandTool"
    assert str(directory) in call.arguments["command"]
    assert isinstance(result, ToolResultMessage)
    assert result.id == call.id
    stdout = result.content["stdout"]
    assert "top.txt" in stdout
    assert "subdir" in stdout
    assert final == reply


def test_execute_prompt_action_md_mention_uses_dynamic_markdown(
    tmp_path: Path,
) -> None:
    """A .md mention's result content is dynamic_markdown-expanded, not raw."""
    (tmp_path / "other.md").write_text("included text")
    target = tmp_path / "doc.md"
    target.write_text("before <include>other.md</include> after")
    agent = _RecordingAgent(AssistantMessage(content="ok"))
    executor = ActionExecutor(
        agent=agent,
        application=_RecordingApplication(),
        settings_manager=_RecordingSettingsManager(),
    )
    action = PromptAction(prompt=f"see @{target}", mentions=[target])

    _, result, _ = list(executor.execute(action))

    assert "included text" in result.content["content"]
    assert "<include>" not in result.content["content"]


def test_execute_prompt_action_multiple_mentions_yield_ordered_pairs(
    tmp_path: Path,
) -> None:
    """Multiple resolved mentions yield ordered call/result pairs before the reply."""
    first = tmp_path / "first.txt"
    second = tmp_path / "second.txt"
    first.write_text("first")
    second.write_text("second")
    reply = AssistantMessage(content="ok")
    agent = _RecordingAgent(reply)
    executor = ActionExecutor(
        agent=agent,
        application=_RecordingApplication(),
        settings_manager=_RecordingSettingsManager(),
    )
    action = PromptAction(
        prompt=f"see @{first} and @{second}", mentions=[first, second]
    )

    messages = list(executor.execute(action))

    assert len(messages) == 5
    first_call, first_result, second_call, second_result, final = messages
    assert first_call.arguments == {"file_path": str(first)}
    assert second_call.arguments == {"file_path": str(second)}
    assert first_result.id == first_call.id
    assert second_result.id == second_call.id
    assert final == reply


class _FinalOnlyClient(BaseClient):
    """Client returning only a final assistant message, no tool calls."""

    _model_options = ["fake-model"]
    _thinking_options: list[str] = []
    _default_model_id = "fake-model"
    _default_thinking_effort = None
    request_schema = BaseRequest

    def __init__(self) -> None:
        """Initialize the fake client."""
        super().__init__(token="token")

    def send_request(self, request: BaseRequest) -> list[Message]:
        """Return a final assistant message, ignoring the request.

        Args:
            request: the request built by the agent.

        Returns:
            A single final assistant message.
        """
        return [AssistantMessage(content="done")]


class _NoOpController:
    """Controller stub never expected to be asked in these tests."""

    def request(self, request: BaseControllerRequest) -> BaseControllerResponse:
        """Approve any request, though none is expected to reach it.

        Args:
            request: the request to approve.

        Returns:
            An ALLOW permission response.
        """
        return PermissionResponse(decision=Decision.ALLOW)


def test_execute_prompt_action_mention_bypasses_read_permission_none(
    tmp_path: Path,
) -> None:
    """A mentioned file is read even when READ permission is set to NONE."""
    target = tmp_path / "secret.txt"
    target.write_text("top secret")
    permission_manager = PermissionManager(
        ToolManager(
            web_summary_client=LiteLLMClient(model_id="openai/gpt-4o", stream=False)
        ),
        _NoOpController(),
        {category: PermissionLevel.NONE for category in PermissionCategory},
    )
    agent = Agent(client=_FinalOnlyClient(), permission_manager=permission_manager)
    assert "ReadFileTool" not in agent._tools

    executor = ActionExecutor(
        agent=agent,
        application=_RecordingApplication(),
        settings_manager=_RecordingSettingsManager(),
    )
    action = PromptAction(prompt=f"see @{target}", mentions=[target])

    messages = list(executor.execute(action))

    call, result, final = messages
    assert isinstance(call, ToolCallMessage)
    assert call.name == "ReadFileTool"
    assert isinstance(result, ToolResultMessage)
    assert final == AssistantMessage(content="done")
