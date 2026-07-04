"""Hello-world command."""

from arancio.commands.base import BaseCommand
from arancio.core.messages import AssistantMessage


class HelloWorldCommand(BaseCommand):
    """Command that greets the name passed as its argument."""

    name = "hello-world"
    description = "Greet the name given as the command's argument."

    @classmethod
    def execute(cls, name: str, times: int) -> AssistantMessage:
        """Return a greeting for the given name, repeated ``times`` times.

        Args:
            name: the name to greet, bound to the first prompt word.
            times: how many times to repeat the greeting, bound to the
                second prompt word.

        Returns:
            An assistant message greeting the name.
        """
        greeting = f"Hello World, {name}"
        return AssistantMessage(content="\n".join(greeting for _ in range(times)))
