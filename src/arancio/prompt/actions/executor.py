"""Executor running the action produced by the prompt manager."""

from collections.abc import Iterator

from arancio.commands.registry import COMMAND_REGISTRY
from arancio.core.agents import Agent
from arancio.core.messages import ErrorMessage, Message, UserMessage
from arancio.prompt.actions.types import BaseAction, CommandAction, PromptAction


class ActionExecutor:
    """Carries out the action produced by the prompt manager.

    A :class:`arancio.prompt.actions.types.CommandAction` is resolved against
    :data:`arancio.commands.registry.COMMAND_REGISTRY` and run locally; a
    :class:`arancio.prompt.actions.types.PromptAction` is sent to the model through the
    agent. Both produce the same message stream so callers render them the same
    way.
    """

    def __init__(self, agent: Agent) -> None:
        """Store the agent used to run a prompt action.

        Args:
            agent: the agent whose run loop handles a prompt action.
        """
        self._agent = agent

    def execute(self, action: BaseAction) -> Iterator[Message]:
        """Execute the action, producing its output messages.

        Args:
            action: the action produced by the prompt manager.

        Returns:
            An iterator over the messages the action produces.

        Raises:
            ValueError: when the action is of an unknown type.
        """
        if isinstance(action, CommandAction):
            return self._execute_command(action)
        elif isinstance(action, PromptAction):
            return self._execute_prompt(action)
        else:
            raise ValueError(f"Unknown action type: {type(action)}")

    def _execute_command(self, action: CommandAction) -> Iterator[Message]:
        """Run the action's command, yielding its output messages.

        Args:
            action: the command action to run.

        Yields:
            The command's result message, or an :class:`ErrorMessage` when the
            command is unknown or its arguments do not match its signature.
        """
        command = COMMAND_REGISTRY.get(action.name)
        if command is None:
            yield ErrorMessage(content=f"Command not found: {action.name}")
            return

        try:
            result = command.run(*action.args)
        except Exception as exc:
            yield ErrorMessage(
                content=f"Error while executing command {action.name}: {exc}"
            )
            return

        if result is not None:
            yield result

    def _execute_prompt(self, action: PromptAction) -> Iterator[Message]:
        """Send the action's text to the model through the agent.

        Args:
            action: the prompt action to send.

        Returns:
            The agent's message stream for the prompt.
        """
        return self._agent.run(UserMessage(content=action.prompt))
