"""Factory instantiating prompt actions."""

from arancio.prompt.actions.types import BaseAction, CommandAction, PromptAction


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
                ``command_name``/``command_args`` for a command action, or
                ``prompt`` for a prompt action.

        Returns:
            A command action when ``kwargs`` contains ``command_name``, otherwise a
            prompt
            action.
        """
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
    def create_prompt_action(cls, prompt: str) -> PromptAction:
        """Create a prompt action.

        Args:
            prompt: the raw prompt to send to the model.

        Returns:
            A prompt action carrying the prompt text.
        """
        return PromptAction(prompt=prompt)
