"""Turn and retry types of the agent loop, shared with the upper layers."""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Retry:
    """How the agent retries a failed turn; frozen so changes go through their owner.

    Attributes:
        max: the number of failed turns in a row before the run stops.
        delay: the seconds to wait before the first retry.
        multiplier: the factor the wait grows by on each further retry.
    """

    max: int
    delay: float
    multiplier: float


@dataclass(frozen=True)
class Turn:
    """How many turns a run may take and how failed turns are retried.

    Shared by Settings, Application and the Agent; frozen like
    :class:`Retry`, so a holder can share it without copying.

    Attributes:
        max: the number of turns per run, or ``"inf"`` for no limit.
        retry: the retry rules for failed turns.
    """

    max: int | str
    retry: Retry

    def to_dict(self) -> dict[str, Any]:
        """Render the configuration with built-in types only.

        Returns:
            The ``max`` value and a nested ``retry`` mapping.
        """
        return {
            "max": self.max,
            "retry": {
                "max": self.retry.max,
                "delay": self.retry.delay,
                "multiplier": self.retry.multiplier,
            },
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Turn":
        """Build the configuration from a complete, checked mapping.

        Args:
            data: the checked values, as produced by :meth:`to_dict`.

        Returns:
            A new turn configuration.
        """
        retry = data["retry"]
        return cls(
            max=data["max"],
            retry=Retry(
                max=retry["max"],
                delay=retry["delay"],
                multiplier=retry["multiplier"],
            ),
        )
