"""System prompt builder blueprint (TODO: concrete implementation)."""

from codo.builders.base import Builder


class SystemPromptBuilder(Builder):
    """Abstract builder that assembles the system prompt for an agent."""

    _base_prompt: str = (
        "You are an helpful assistant assisting the user Pier. Answer his questions."
    )

    @classmethod
    def build(cls) -> str:
        """Build and return the system prompt.

        Returns:
            The rendered system prompt string.
        """
        return cls._base_prompt
