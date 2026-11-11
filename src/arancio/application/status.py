"""Display values Application gives to a frontend."""

from dataclasses import dataclass
from pathlib import Path

from arancio.core.clients.configs.model import ModelConfig


@dataclass(frozen=True)
class ApplicationStatus:
    """Immutable values a frontend can read without accessing managed services.

    Attributes:
        model: the active model section: provider, name and reasoning options.
        working_directory: the runtime working directory owned by Application.
    """

    model: ModelConfig
    working_directory: Path
