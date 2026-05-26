"""Provider-agnostic filesystem path constants."""

from pathlib import Path

HARNESS_DIR_ROOT_PATH = Path("harness")
TOOLS_HARNESS_PATH = HARNESS_DIR_ROOT_PATH / "tools"

TOOL_DESCRIPTION_FILENAME = "description.md"
TOOL_INPUT_SCHEMA_FILENAME = "input_schema.yml"
