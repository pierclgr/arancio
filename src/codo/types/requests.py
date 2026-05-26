"""Generic request models shared across client implementations."""

from dataclasses import dataclass, field
from typing import List

from codo.types.messages import Message
from codo.types.tools import ToolSchema


@dataclass(frozen=True)
class BaseRequest:
    """Canonical request passed to every client implementation."""

    model_id: str
    thinking_effort: str | None = None
    thinking_summary: str | None = None
    system_prompt: str = ""
    tool_list: List[ToolSchema] = field(default_factory=list)
    message_list: List[Message] = field(default_factory=list)


LiteLLMRequest = BaseRequest
