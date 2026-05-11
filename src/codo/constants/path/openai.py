"""OpenAI-specific filesystem path constants."""

from pathlib import Path

OPENAI_OAUTH_TOKEN_PATH = (
    Path.home() / ".config" / "codo" / "openai" / "oauth_token.json"
)
