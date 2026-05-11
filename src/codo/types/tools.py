"""Tool definitions exposed to LLM clients."""

from dataclasses import dataclass


@dataclass(frozen=True)
class ToolSchema:
    """Definition of a tool exposable to a model.

    Attributes:
        name: unique tool identifier.
        description: human-readable tool purpose.
        input_schema: JSON Schema describing the tool arguments.
    """

    name: str
    description: str
    input_schema: dict
