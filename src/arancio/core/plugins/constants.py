"""Import-name constants for the plugin system.

The plugin folder and filenames live with the other filesystem constants, in
:mod:`arancio.core.constants.path`, next to the harness paths they mirror.
"""

# synthetic package every plugin module is imported under, so a plugin's own
# modules are reachable by relative import without touching sys.path
PLUGIN_PACKAGE_ROOT = "arancio_plugins"
