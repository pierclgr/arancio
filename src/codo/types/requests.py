"""Generic request models shared across client implementations."""

from dataclasses import dataclass, field
from typing import List

from codo.constants.openai import (
    OPENAI_DEFAULT_MODEL_ID,
    OPENAI_DEFAULT_THINKING_EFFORT,
    OPENAI_DEFAULT_THINKING_SUMMARY,
)
from codo.types.messages import Message
from codo.types.tools import ToolSchema


@dataclass(frozen=True)
class BaseRequest:
    """Canonical request passed to every client implementation."""

    model_id: str
    thinking_effort: str | None = None
    system_prompt: str = ""
    tool_list: List[ToolSchema] = field(default_factory=list)
    message_list: List[Message] = field(default_factory=list)


@dataclass(frozen=True)
class OpenAIRequest(BaseRequest):
    """Canonical request passed to OpenAI client implementations."""

    model_id: str = OPENAI_DEFAULT_MODEL_ID
    thinking_effort: str = OPENAI_DEFAULT_THINKING_EFFORT
    thinking_summary: str = OPENAI_DEFAULT_THINKING_SUMMARY


LiteLLMRequest = BaseRequest
