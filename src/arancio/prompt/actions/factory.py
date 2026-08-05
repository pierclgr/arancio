"""Factory instantiating prompt actions."""

from pathlib import Path

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
    def create_action(cls, **kwargs) -> BaseAction:
        """Create the action described by the given keyword arguments.

        Args:
            **kwargs: the keyword arguments produced by
                :meth:`arancio.prompt.manager.PromptManager.resolve_prompt`:
                ``shell_command``/``add_to_history`` for a shell action,
                ``command_name``/``command_args`` for a slash command, or
                ``prompt``/``mentions`` for a prompt action.

        Returns:
            The concrete action described by ``kwargs``.
        """
        if "shell_command" in kwargs:
            return cls.create_shell_command_action(**kwargs)
        if "command_name" in kwargs:
            return cls.create_command_action(**kwargs)
        return cls.create_prompt_action(**kwargs)

    @classmethod
    def create_command_action(
        cls, command_name: str, command_args: list[str]
    ) -> CommandAction:
        """Create a command action.

        Args:
            command_name: the command name parsed from the prompt.
            command_args: the command arguments parsed from the prompt.

        Returns:
            A command action for the given name and arguments.
        """
        return CommandAction(name=command_name, args=command_args)

    @classmethod
    def create_shell_command_action(
        cls, shell_command: str, add_to_history: bool
    ) -> ShellCommandAction:
        """Create a shell command action.

        Args:
            shell_command: the exact command text following the shell prefix.
            add_to_history: whether to store the tool call and result in agent
                history.

        Returns:
            A shell command action for the given command and history policy.
        """
        return ShellCommandAction(
            command=shell_command,
            add_to_history=add_to_history,
        )

    @classmethod
    def create_prompt_action(
        cls, prompt: str, mentions: list[Path] | None = None
    ) -> PromptAction:
        """Create a prompt action.

        Args:
            prompt: the prompt to send to the model.
            mentions: resolved absolute paths for the prompt's @mentions, in
                order. Defaults to none.

        Returns:
            A prompt action carrying the prompt text and its resolved
            mentions.
        """
        return PromptAction(prompt=prompt, mentions=list(mentions or []))
