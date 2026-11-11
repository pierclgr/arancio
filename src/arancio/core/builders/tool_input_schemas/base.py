"""Builder deriving a tool's JSON input schema from its method signature."""

import inspect
from typing import Any, Callable, get_type_hints

from pydantic import Field, PydanticUserError, create_model

from arancio.core.builders.base import Builder
from arancio.core.parsers.docstrings.base import DocstringParser
from arancio.core.parsers.docstrings.google import GoogleDocstringParser
from arancio.core.parsers.docstrings.numpy import NumpyDocstringParser
from arancio.core.parsers.docstrings.sphinx import SphinxDocstringParser


class InputSchemaBuilder(Builder):
    """Build the raw JSON input schema of a tool from the method it runs.

    Property names, types and required-ness come from the signature, constraints from
    ``Annotated[..., Field(...)]`` hints, and descriptions from the parameter section of
    the docstring (Google, NumPy or Sphinx style). Optional ``X | None`` parameters are
    shown to the model as plain ``X``.
    """

    _null_type_schema: dict[str, str] = {"type": "null"}
    _docstring_parsers: tuple[type[DocstringParser], ...] = (
        GoogleDocstringParser,
        NumpyDocstringParser,
        SphinxDocstringParser,
    )

    @classmethod
    def build(cls, method: Callable) -> dict[str, Any]:
        """Build the input schema of a tool method.

        Args:
            method: a bound method, or an unbound one whose first parameter is the
                instance; the instance is not part of the schema.

        Returns:
            A mapping with the ``properties`` and ``required`` schema keys.

        Raises:
            ValueError: when a type hint cannot be resolved, a parameter is
                variadic or has no type hint, or a type has no JSON schema.
        """
        parameters: list[inspect.Parameter] = list(
            inspect.signature(method).parameters.values()
        )
        if not inspect.ismethod(method):
            parameters = parameters[1:]
        try:
            hints: dict[str, Any] = get_type_hints(method, include_extras=True)
        except (NameError, TypeError) as exc:
            raise ValueError(
                f"{method.__qualname__}: type hints cannot be resolved: {exc}"
            ) from exc
        descriptions: dict[str, str] = cls.__parse_arg_descriptions(
            inspect.getdoc(method) or ""
        )
        fields: dict[str, Any] = {}
        for parameter in parameters:
            if parameter.kind in (
                inspect.Parameter.VAR_POSITIONAL,
                inspect.Parameter.VAR_KEYWORD,
            ):
                raise ValueError(
                    f"{method.__qualname__}: variadic parameter {parameter.name!r} "
                    "cannot be described as a tool argument."
                )
            if parameter.name not in hints:
                raise ValueError(
                    f"{method.__qualname__}: parameter {parameter.name!r} "
                    "has no type hint."
                )
            is_required: bool = parameter.default is inspect.Parameter.empty
            fields[parameter.name] = (
                hints[parameter.name],
                Field(
                    ... if is_required else parameter.default,
                    description=descriptions.get(parameter.name),
                ),
            )
        try:
            schema: dict[str, Any] = create_model(
                method.__qualname__, **fields
            ).model_json_schema()
        except PydanticUserError as exc:
            raise ValueError(f"{method.__qualname__}: {exc}") from exc
        return cls.__simplify(schema)

    @classmethod
    def __parse_arg_descriptions(cls, docstring: str) -> dict[str, str]:
        """Read the parameter descriptions of a docstring in any supported style.

        Args:
            docstring: the dedented docstring.

        Returns:
            Each documented parameter mapped to its description on one line, as read
            by the first style that finds any; empty when none does.
        """
        for parser in cls._docstring_parsers:
            descriptions: dict[str, str] = parser.parse(docstring)
            if descriptions:
                return descriptions
        return {}

    @classmethod
    def __simplify(cls, schema: dict[str, Any]) -> dict[str, Any]:
        """Remove pydantic-specific noise from a generated model schema.

        Args:
            schema: the schema returned by ``model_json_schema``.

        Returns:
            The ``properties`` and ``required`` keys, without titles, with
            nullable unions reduced to their non-null type and null defaults dropped.
        """
        properties: dict[str, dict[str, Any]] = {}
        for name, prop in schema["properties"].items():
            prop = {key: value for key, value in prop.items() if key != "title"}
            branches: list[dict[str, Any]] = prop.get("anyOf", [])
            if cls._null_type_schema in branches and len(branches) == 2:
                (non_null,) = [b for b in branches if b != cls._null_type_schema]
                del prop["anyOf"]
                prop = {**non_null, **prop}
            if "default" in prop and prop["default"] is None:
                del prop["default"]
            properties[name] = prop
        return {"properties": properties, "required": schema.get("required", [])}
