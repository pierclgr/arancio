"""Tests for the action executor."""

from collections.abc import Iterator

from arancio.core.messages import (
    AssistantMessage,
    ErrorMessage,
    Message,
    UserMessage,
)
from arancio.core.permissions.types import PermissionCategory, PermissionLevel
from arancio.prompt.actions.executor import ActionExecutor
from arancio.prompt.actions.factory import ActionFactory
from arancio.prompt.actions.types import CommandAction, PromptAction
from arancio.settings.settings import Settings


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
    """Agent stub recording the message it runs and yielding a fixed reply."""

    def __init__(self, reply: Message) -> None:
        """Store the reply each run yields.

        Args:
            reply: the message produced for every ``run`` call.
        """
        self._reply = reply
        self.received: Message | None = None

    def run(self, message: Message) -> Iterator[Message]:
        """Record the input message and yield the configured reply.

        Args:
            message: the user message that starts the turn.

        Yields:
            The configured reply message.
        """
        self.received = message
        yield self._reply


def test_factory_creates_command_action() -> None:
    """The factory builds a command action from a name and arguments."""
    action = ActionFactory.create_command_action("hello-world", ["Sam", "1"])

    assert action == CommandAction(name="hello-world", args=["Sam", "1"])


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
