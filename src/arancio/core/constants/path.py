"""Provider-agnostic filesystem path constants."""

from pathlib import Path

# arancio working directory holding all persistent state (login info, harness, …)
ARANCIO_DEFAULT_DIR = Path.home() / ".arancio"

# harness resources; the user places their files here manually
HARNESS_DIR_ROOT_PATH = ARANCIO_DEFAULT_DIR / "harness"
TOOLS_HARNESS_PATH = HARNESS_DIR_ROOT_PATH / "tools"
SYSTEM_PROMPT_HARNESS_PATH = HARNESS_DIR_ROOT_PATH / "SYSTEM_PROMPT.md"

TOOL_DESCRIPTION_FILENAME = "description.md"
TOOL_INPUT_SCHEMA_FILENAME = "input_schema.yml"

# plugins; one folder per plugin, placed here by the user like the harness files
PLUGINS_PATH = ARANCIO_DEFAULT_DIR / "plugins"

PLUGIN_MANIFEST_FILENAME = "manifest.yml"
PLUGIN_MODULE_FILENAME = "module.py"
