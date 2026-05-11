"""OpenAI client defaults, option sets, env var name, and OAuth identifiers."""

from typing import List

OPENAI_MODEL_OPTIONS: List[str] = ["gpt-5.4", "gpt-5.5"]
OPENAI_THINKING_OPTIONS: List[str] = [
    "none",
    "minimal",
    "low",
    "medium",
    "high",
    "xhigh",
]
OPENAI_THINKING_SUMMARY_OPTIONS: List[str] = ["auto", "concise", "detailed"]

OPENAI_DEFAULT_MODEL_ID: str = "gpt-5.5"
OPENAI_DEFAULT_THINKING_EFFORT: str = "medium"
OPENAI_DEFAULT_THINKING_SUMMARY: str = "auto"

OPENAI_API_KEY_ENV_VAR: str = "OPENAI_API_KEY"
OPENAI_OAUTH_CLIENT_ID: str = "app_EMoamEEZ73f0CkXaXp7hrann"
OPENAI_OAUTH_SCOPE: str = "openid profile email offline_access"
