"""Model and reasoning types of a client, shared with the upper layers."""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Reasoning:
    """The reasoning configuration of a model; frozen so changes go through their owner.

    Attributes:
        effort: the reasoning effort tier (e.g. ``"low"``), or ``None`` to disable
            reasoning.
        summary: the reasoning summary level (e.g. ``"auto"``), or ``None`` to
            disable summaries.
    """

    effort: str | None = None
    summary: str | None = None


@dataclass(frozen=True)
class Model:
    """The model to use and its reasoning configuration, shared across layers.

    Frozen like :class:`Reasoning`, so a holder can share it without copying.

    Attributes:
        provider: the provider prefix (e.g. ``"openai"``), or ``None`` when not
            configured.
        name: the model name without the provider prefix, or ``None`` when not
            configured.
        reasoning: the reasoning effort and summary level; ``None`` values disable
            reasoning or summaries (e.g. no summary for Ollama models).
    """

    provider: str | None
    name: str | None
    reasoning: Reasoning

    def check_configured(self) -> None:
        """Require both the provider and the model name, as the agent needs them.

        Raises:
            ValueError: when the provider is not configured.
            ValueError: when the model name is not configured.
        """
        if not self.provider:
            raise ValueError("No provider configured.")
        if not self.name:
            raise ValueError("No model name configured.")

    def to_dict(self) -> dict[str, Any]:
        """Render the options with built-in types only.

        Returns:
            The provider, the name and a nested ``reasoning`` mapping.
        """
        return {
            "provider": self.provider,
            "name": self.name,
            "reasoning": {
                "effort": self.reasoning.effort,
                "summary": self.reasoning.summary,
            },
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Model":
        """Build the options from a complete, checked mapping.

        Args:
            data: the checked values, as produced by :meth:`to_dict`.

        Returns:
            New model options.
        """
        return cls(
            provider=data["provider"],
            name=data["name"],
            reasoning=Reasoning(
                effort=data["reasoning"]["effort"],
                summary=data["reasoning"]["summary"],
            ),
        )
