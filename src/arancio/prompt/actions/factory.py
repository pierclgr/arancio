"""Factory instantiating prompt actions."""

from arancio.prompt.actions.types import CommandAction, PromptAction


class ActionFactory:
    """Instantiates the concrete actions handled by the prompt manager.

    Centralizes construction of the :class:`arancio.prompt.actions.types.BaseAction`
    subtypes so callers depend on the factory rather than on each action's constructor.
    """

    @classmethod
    def create_command_action(cls, name: str, args: list[str]) -> CommandAction:
        """Create a command action.

        Args:
            name: the command name parsed from the prompt.
            args: the command arguments parsed from the prompt.

        Returns:
            A command action for the given name and arguments.
        """
        return CommandAction(name=name, args=args)

    @classmethod
    def create_prompt_action(cls, prompt: str) -> PromptAction:
        """Create a prompt action.

        Args:
            prompt: the raw prompt to send to the model.

        Returns:
            A prompt action carrying the prompt text.
        """
        return PromptAction(prompt=prompt)
