"""Constants used to persist LiteLLM login data."""

from pathlib import Path

from arancio.core.constants.path import ARANCIO_DEFAULT_DIR

# LiteLLM login storage redirected into the arancio working directory
ARANCIO_LITELLM_DIR = ARANCIO_DEFAULT_DIR / "litellm"
LITELLM_CONFIG_DIR = Path.home() / ".config" / "litellm"
