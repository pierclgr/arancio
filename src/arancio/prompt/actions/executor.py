"""Executor running the action built from the prompt manager's arguments."""

from __future__ import annotations

import inspect
from collections.abc import Iterator
from typing import TYPE_CHECKING, Any, Type

from arancio.commands.base import BaseCommand
from arancio.commands.registry import COMMAND_REGISTRY
from arancio.core.agents import Agent
from arancio.core.messages import ErrorMessage, Message, UserMessage
from arancio.prompt.actions.types import BaseAction, CommandAction, PromptAction

if TYPE_CHECKING:
    from arancio.ui.app import App


class ActionExecutor:
    """Carries out an action built from the prompt manager's arguments.

    A :class:`arancio.prompt.actions.types.CommandAction` is resolved against
    :data:`arancio.commands.registry.COMMAND_REGISTRY`: its raw prompt words are
    bound to the command's parameter names and run locally. A
    :class:`arancio.prompt.actions.types.PromptAction` is sent to the model
    through the agent. Both produce the same message stream so callers render
    them the same way.
    """

    def __init__(self, agent: Agent, application: App) -> None:
        """Store the agent and application used to carry out actions.

        Args:
            agent: the agent whose run loop handles a prompt action.
            application: the running app a command acts on.
        """
        self._agent = agent
        self._application = application

    def execute(self, action: BaseAction) -> Iterator[Message]:
        """Execute the action, producing its output messages.

        Args:
            action: the action to execute.

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
        """Bind the action's words to the command's parameters and run it.

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
            kwargs = self._build_command_kwargs(command, action.args)
            result = command.run(**kwargs)
        except Exception as exc:
            yield ErrorMessage(
                content=f"Error while executing command {action.name}: {exc}"
            )
            return

        if result is not None:
            yield result

    def _build_command_kwargs(
        self, command: Type[BaseCommand], args: list[str]
    ) -> dict[str, Any]:
        """Match the prompt words to the command's parameters and add the application.

        Each word is matched, in order, to the next parameter the command
        declares (excluding ``application``); a word beyond the number of
        declared parameters is dropped rather than rejected.

        Args:
            command: the command whose ``execute`` parameters to match against.
            args: the raw prompt words.

        Returns:
            The keyword arguments for :meth:`command.run`: the prompt words
            keyed by their matching parameter name, plus the running application
            when the command declares an ``application`` parameter.
        """
        signature = inspect.signature(command.execute)
        prompt_parameter_names = [
            parameter_name
            for parameter_name, parameter in signature.parameters.items()
            if parameter_name != "application"
            and parameter.kind
            in (
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                inspect.Parameter.POSITIONAL_ONLY,
            )
        ]
        kwargs = dict(zip(prompt_parameter_names, args))
        if "application" in signature.parameters:
            kwargs["application"] = self._application
        return kwargs

    def _execute_prompt(self, action: PromptAction) -> Iterator[Message]:
        """Send the action's text to the model through the agent.

        Args:
            action: the prompt action to send.

        Returns:
            The agent's message stream for the prompt.
        """
        return self._agent.run(UserMessage(content=action.prompt))
