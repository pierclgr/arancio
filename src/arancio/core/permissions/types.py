"""Permission value types: autonomy levels, tool categories and decisions."""

from dataclasses import dataclass
from enum import Enum
from typing import ClassVar, Iterator


class PermissionLevel(Enum):
    """Autonomy level governing how a permitted tool category is executed.

    Attributes:
        NONE: the category is not granted; its tools are never created.
        ASK: prompt the user for confirmation before each tool call.
        AUTO: execute permitted tool calls without confirmation.
    """

    NONE = None
    ASK = "ask"
    AUTO = "auto"


class _PermissionCategoryMeta(type):
    """Metaclass giving PermissionCategory class-level iteration and lookup.

    Mirrors the surface :class:`enum.EnumMeta` gives ``for x in Enum``/
    ``Enum[name]``/``Enum.__members__``, so every existing call site keeps working
    unchanged even though categories are no longer enum members.
    """

    def __iter__(cls) -> Iterator["PermissionCategory"]:
        """Iterate every registered category.

        Returns:
            An iterator over the registered categories.
        """
        return iter(cls._REGISTRY.values())

    def __getitem__(cls, name: str) -> "PermissionCategory":
        """Look up a registered category by name, case-insensitively.

        Args:
            name: the category's name.

        Returns:
            The matching category.
        """
        return cls._REGISTRY[name.upper()]

    def __len__(cls) -> int:
        """Return how many categories are registered.

        Returns:
            The number of registered categories.
        """
        return len(cls._REGISTRY)

    @property
    def __members__(cls) -> dict[str, "PermissionCategory"]:
        """Return every registered category keyed by its uppercased name.

        Returns:
            A copy of the registry.
        """
        return dict(cls._REGISTRY)


class PermissionCategory(metaclass=_PermissionCategoryMeta):
    """Identity of a capability that a permission grant covers.

    A category does not own tools: each tool definition registered with
    :class:`~arancio.core.tools.manager.ToolManager` names its category. Not an
    ``enum.Enum``: a plugin can create a brand new category (via
    ``@tool(category="CUSTOM")``) after the built-in categories already exist,
    which a closed ``Enum`` cannot do. Every category, built-in or
    plugin-created, is this same type, registered by name (case-insensitively)
    in one place, so nothing downstream special-cases one kind against the
    other.

    Attributes:
        name: the category's display name, as given at creation (e.g.
            ``"READ"``, ``"PLUGIN"``, ``"PLUGIN:TestPlugin"``).
    """

    _REGISTRY: ClassVar[dict[str, "PermissionCategory"]] = {}

    READ: ClassVar["PermissionCategory"]
    WRITE: ClassVar["PermissionCategory"]
    WEB: ClassVar["PermissionCategory"]
    EXECUTE: ClassVar["PermissionCategory"]
    PLUGIN: ClassVar["PermissionCategory"]

    def __init__(self, name: str) -> None:
        """Create and register a new category.

        Args:
            name: the category's display name; registered case-insensitively.

        Raises:
            ValueError: when a category with this name (case-insensitively)
                is already registered.
        """
        key = name.upper()
        if key in PermissionCategory._REGISTRY:
            raise ValueError(f"Permission category {name!r} already exists.")
        self.name = name
        PermissionCategory._REGISTRY[key] = self

    def __repr__(self) -> str:
        """Return a developer-friendly representation of the category.

        Returns:
            The category's own name, qualified by the class name.
        """
        return f"PermissionCategory({self.name!r})"

    @classmethod
    def get(cls, name: str) -> "PermissionCategory | None":
        """Return the category registered under this name, if any.

        Args:
            name: the category name, matched case-insensitively.

        Returns:
            The matching category, or ``None`` when none is registered.
        """
        return cls._REGISTRY.get(name.upper())

    @classmethod
    def get_or_create(cls, name: str) -> "PermissionCategory":
        """Return the category registered under this name, creating it if needed.

        Args:
            name: the category name; a new, empty category is created and
                registered under it when none exists yet.

        Returns:
            The existing or newly created category.
        """
        return cls.get(name) or cls(name)


PermissionCategory.READ = PermissionCategory("READ")
PermissionCategory.WRITE = PermissionCategory("WRITE")
PermissionCategory.WEB = PermissionCategory("WEB")
PermissionCategory.EXECUTE = PermissionCategory("EXECUTE")
PermissionCategory.PLUGIN = PermissionCategory("PLUGIN")


class PermissionOutcome(Enum):
    """How a requested tool call was resolved.

    Attributes:
        ALLOWED: the call may run.
        DENIED: the user refused the call.
        UNAVAILABLE: the tool is unknown or its category is not granted.
    """

    ALLOWED = "allowed"
    DENIED = "denied"
    UNAVAILABLE = "unavailable"


@dataclass(frozen=True)
class PermissionDecision:
    """Resolved permission for one tool call, for the caller to act on.

    Attributes:
        outcome: how the call was resolved.
        note: the user's note when allowing, or the reason when denying.
    """

    outcome: PermissionOutcome
    note: str | None = None
