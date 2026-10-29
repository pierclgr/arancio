"""Default permission levels for built-in categories."""

from arancio.core.permissions.types import PermissionCategory, PermissionLevel

DEFAULT_PERMISSION_LEVELS: dict[PermissionCategory, PermissionLevel] = {
    PermissionCategory.READ: PermissionLevel.ASK,
    PermissionCategory.WRITE: PermissionLevel.ASK,
    PermissionCategory.WEB: PermissionLevel.ASK,
    PermissionCategory.EXECUTE: PermissionLevel.ASK,
    PermissionCategory.PLUGIN: PermissionLevel.ASK,
}
