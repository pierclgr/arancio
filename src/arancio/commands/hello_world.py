"""Hello-world command."""

from arancio.commands.base import BaseCommand


class HelloWorldCommand(BaseCommand):
    """Command that greets the name passed as its argument."""

    name = "hello-world"
    description = "Greet the name given as the command's argument."

    @classmethod
    def execute(cls, name: str, times: int = 1, **kwargs) -> str:
        """Return a greeting for the given name, repeated ``times`` times.

        Args:
            name: the name to greet, bound to the first prompt word.
            times: how many times to repeat the greeting, bound to the
                second prompt word; defaults to 1 when omitted.
            **kwargs: absorbs the ``application`` argument, which this command
                does not need.

        Returns:
            The greeting text.
        """
        greeting = f"Hello World, {name}"
        return "\n".join(greeting for _ in range(times))
