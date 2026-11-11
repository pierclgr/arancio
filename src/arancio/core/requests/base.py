"""Generic request models shared across client implementations."""

from dataclasses import dataclass, field
from typing import List

from arancio.core.clients.configs.reasoning import ReasoningConfig
from arancio.core.messages.base import Message
from arancio.core.tools.schema import ToolSchema


@dataclass
class Request:
    """Mutable request shared with hooks before being sent to the client."""

    model_id: str
    reasoning: ReasoningConfig = field(default_factory=ReasoningConfig)
    system_prompt: str = ""
    tool_list: List[ToolSchema] = field(default_factory=list)
    message_list: List[Message] = field(default_factory=list)
