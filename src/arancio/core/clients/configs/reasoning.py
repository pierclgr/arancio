"""Reasoning configuration of a client, shared with the upper layers."""

from dataclasses import dataclass


@dataclass(frozen=True)
class ReasoningConfig:
    """The reasoning configuration of a model; frozen so changes go through their owner.

    Attributes:
        effort: the reasoning effort tier (e.g. ``"low"``), or ``None`` to disable
            reasoning.
        summary: the reasoning summary level (e.g. ``"auto"``), or ``None`` to
            disable summaries.
    """

    effort: str | None = None
    summary: str | None = None
