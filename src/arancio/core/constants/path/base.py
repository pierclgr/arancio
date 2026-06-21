"""Provider-agnostic filesystem path constants."""

from pathlib import Path

# arancio working directory holding all persistent state (login info, harness, …)
ARANCIO_DEFAULT_DIR = Path.home() / ".arancio"

# harness the tools read from; the user places its files here manually
HARNESS_DIR_ROOT_PATH = ARANCIO_DEFAULT_DIR / "harness"
TOOLS_HARNESS_PATH = HARNESS_DIR_ROOT_PATH / "tools"

# litellm login storage redirected into the working directory: litellm hardcodes its
# config dir, so arancio symlinks that dir onto this arancio target
ARANCIO_LITELLM_DIR = ARANCIO_DEFAULT_DIR / "litellm"
LITELLM_CONFIG_DIR = Path.home() / ".config" / "litellm"

TOOL_DESCRIPTION_FILENAME = "description.md"
TOOL_INPUT_SCHEMA_FILENAME = "input_schema.yml"
