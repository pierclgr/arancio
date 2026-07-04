"""Tests for the action executor."""

from collections.abc import Iterator

from arancio.core.messages import (
    AssistantMessage,
    ErrorMessage,
    Message,
    UserMessage,
)
from arancio.prompt.actions.executor import ActionExecutor
from arancio.prompt.actions.factory import ActionFactory
from arancio.prompt.actions.types import CommandAction, PromptAction


class _DummyAgent:
    """Stand-in agent; command-action tests never run its loop."""


class _RecordingApplication:
    """Application stub recording whether it was asked to exit."""

    def __init__(self) -> None:
        """Start with no recorded exit."""
        self.exit_called = False

    def exit(self) -> None:
        """Record that an exit was requested."""
        self.exit_called = True


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
    executor = ActionExecutor(agent=_DummyAgent(), application=_RecordingApplication())
    action = CommandAction(name="hello-world", args=["Sam", "1"])

    messages = list(executor.execute(action))

    assert messages == [AssistantMessage(content="Hello World, Sam")]


def test_execute_command_action_ignores_extra_prompt_words() -> None:
    """A prompt word beyond the command's parameters is silently dropped."""
    executor = ActionExecutor(agent=_DummyAgent(), application=_RecordingApplication())
    action = CommandAction(name="hello-world", args=["Sam", "1", "extra"])

    messages = list(executor.execute(action))

    assert messages == [AssistantMessage(content="Hello World, Sam")]


def test_execute_unknown_command_yields_error() -> None:
    """An unknown command name yields an error message."""
    executor = ActionExecutor(agent=_DummyAgent(), application=_RecordingApplication())

    messages = list(executor.execute(CommandAction(name="nope", args=[])))

    assert messages == [ErrorMessage(content="Command not found: nope")]


def test_execute_command_with_uncoercible_argument_yields_error() -> None:
    """An argument that cannot match its type yields an error message."""
    executor = ActionExecutor(agent=_DummyAgent(), application=_RecordingApplication())
    action = CommandAction(name="hello-world", args=["Sam", "three"])

    (message,) = list(executor.execute(action))

    assert isinstance(message, ErrorMessage)
    assert "hello-world" in message.content


def test_execute_command_missing_argument_yields_error() -> None:
    """A command missing a mandatory argument yields an error message."""
    executor = ActionExecutor(agent=_DummyAgent(), application=_RecordingApplication())
    action = CommandAction(name="hello-world", args=[])

    (message,) = list(executor.execute(action))

    assert isinstance(message, ErrorMessage)
    assert "hello-world" in message.content


def test_execute_exit_command_quits_the_application() -> None:
    """The exit command action exits the application and yields no message."""
    application = _RecordingApplication()
    executor = ActionExecutor(agent=_DummyAgent(), application=application)

    messages = list(executor.execute(CommandAction(name="exit", args=[])))

    assert messages == []
    assert application.exit_called is True


def test_execute_prompt_action_delegates_to_agent() -> None:
    """A prompt action sends its text to the model through the agent."""
    reply = AssistantMessage(content="hi")
    agent = _RecordingAgent(reply)
    executor = ActionExecutor(agent=agent, application=_RecordingApplication())

    messages = list(executor.execute(PromptAction(prompt="hello there")))

    assert messages == [reply]
    assert agent.received == UserMessage(content="hello there")
