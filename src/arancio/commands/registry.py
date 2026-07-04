"""Registry of available slash commands."""

from typing import Dict, Type

from arancio.commands.base import BaseCommand
from arancio.commands.hello_world import HelloWorldCommand

# register a command by adding an entry keyed by its name
COMMAND_REGISTRY: Dict[str, Type[BaseCommand]] = {
    HelloWorldCommand.name: HelloWorldCommand,
}
