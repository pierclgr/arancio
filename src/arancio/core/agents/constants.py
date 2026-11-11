"""Agent loop constants."""

from arancio.core.agents.types import Retry, Turn

# "inf" means unlimited: the loop runs until the model stops requesting
# tools; set an int only to cap unattended runs
UNLIMITED_MAX_TURNS: str = "inf"
# stop after 5 failed turns in a row, waiting 3, 6, 12, 24 seconds between the
# retries
DEFAULT_RETRY: Retry = Retry(max=5, delay=3.0, multiplier=2.0)
# no turn limit, with the default retry rules
DEFAULT_TURN: Turn = Turn(
    max=UNLIMITED_MAX_TURNS,
    retry=DEFAULT_RETRY,
)
# used by SystemPromptBuilder when the harness system_prompt.md file is missing
DEFAULT_SYSTEM_PROMPT: str = (
    "You are an helpful assistant. Answer the user's questions."
)
