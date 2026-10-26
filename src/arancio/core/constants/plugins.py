"""Import-name constants for the plugin system."""

# synthetic package every plugin module is imported under, so a plugin's own
# modules are reachable by relative import without touching sys.path
PLUGIN_PACKAGE_ROOT = "arancio_plugins"
