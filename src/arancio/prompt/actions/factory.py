"""Factory instantiating prompt actions."""

from pathlib import Path
from typing import Any

from arancio.prompt.actions.types import (
    BaseAction,
    CommandAction,
    PromptAction,
    ShellCommandAction,
)


class ActionFactory:
    """Instantiates the concrete actions handled by the prompt manager.

    Centralizes construction of the :class:`arancio.prompt.actions.types.BaseAction`
    subtypes so callers depend on the factory rather than on each action's constructor.
    """

    @classmethod
    def create_action(cls, raw_input: str, **kwargs: Any) -> BaseAction:
        """Create the action described by the given keyword arguments.

        Args:
            raw_input: the exact text the user submitted before parsing.
            **kwargs: the keyword arguments produced by
                :meth:`arancio.prompt.manager.PromptManager.resolve_prompt`:
                ``shell_command``/``add_to_history`` for a shell action,
                ``command_name``/``command_args`` for a slash command, or
                ``prompt``/``mentions`` for a prompt action.

        Returns:
            The concrete action described by ``kwargs``.
        """
        if "shell_command" in kwargs:
            return cls.create_shell_command_action(raw_input=raw_input, **kwargs)
        if "command_name" in kwargs:
            return cls.create_command_action(raw_input=raw_input, **kwargs)
        return cls.create_prompt_action(raw_input=raw_input, **kwargs)

    @classmethod
    def create_command_action(
        cls, command_name: str, command_args: list[str], raw_input: str
    ) -> CommandAction:
        """Create a command action.

        Args:
            command_name: the command name parsed from the prompt.
            command_args: the command arguments parsed from the prompt.
            raw_input: the exact text the user submitted.

        Returns:
            A command action for the given name and arguments.
        """
        return CommandAction(
            name=command_name,
            args=command_args,
            raw_input=raw_input,
        )

    @classmethod
    def create_shell_command_action(
        cls, shell_command: str, add_to_history: bool, raw_input: str
    ) -> ShellCommandAction:
        """Create a shell command action.

        Args:
            shell_command: the exact command text following the shell prefix.
            add_to_history: whether to store the tool call and result in agent
                history.
            raw_input: the exact text the user submitted.

        Returns:
            A shell command action for the given command and history policy.
        """
        return ShellCommandAction(
            command=shell_command,
            add_to_history=add_to_history,
            raw_input=raw_input,
        )

    @classmethod
    def create_prompt_action(
        cls,
        prompt: str,
        raw_input: str,
        mentions: list[Path] | None = None,
    ) -> PromptAction:
        """Create a prompt action.

        Args:
            prompt: the prompt to send to the model.
            raw_input: the exact text the user submitted.
            mentions: resolved absolute paths for the prompt's @mentions, in
                order. Defaults to none.

        Returns:
            A prompt action carrying the prompt text and its resolved
            mentions.
        """
        return PromptAction(
            prompt=prompt,
            mentions=list(mentions or []),
            raw_input=raw_input,
        )
