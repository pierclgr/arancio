"""Agent loop constants."""

# "inf" means unlimited: the loop runs until the model stops requesting
# tools; set an int only to cap unattended runs
AGENT_UNLIMITED_MAX_TURNS: str = "inf"
AGENT_DEFAULT_MAX_TURNS: str = AGENT_UNLIMITED_MAX_TURNS
# consecutive failed turns before the run aborts
AGENT_DEFAULT_MAX_RETRIES: int = 5
AGENT_DEFAULT_TURN_WAIT_TIME: float = 3.0
AGENT_DEFAULT_TURN_WAIT_TIME_MULTIPLIER: float = 2.0
