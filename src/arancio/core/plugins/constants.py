"""Path and import-name constants for the plugin system."""

from arancio.core.constants import ARANCIO_DEFAULT_DIR

PLUGINS_PATH = ARANCIO_DEFAULT_DIR / "plugins"
PLUGIN_MODULE_FILENAME = "plugin.py"

# synthetic package every plugin module is imported under, so a plugin's own
# modules are reachable by relative import without touching sys.path
PLUGIN_PACKAGE_ROOT = "arancio_plugins"
