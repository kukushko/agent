"""Discovery, validation, schema generation, and dispatch for tools."""

from __future__ import annotations

import importlib
import inspect
import json
import logging
import pkgutil
import re
import types
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Iterator, Union, get_args, get_origin, get_type_hints

from toolpolicy import render_reliability_turn_guidance

from .base import (
    TOOL_SUCCESS_MARKER,
    TOOL_RESOURCE_USAGE_MARKER,
    ToolContext,
    ToolEnvironment,
    ToolMetadata,
    ToolPackage,
    activate_tool_context,
    is_tool,
    tool_metadata,
)


NAME_RE = re.compile(r"^[a-z][a-z0-9_]*$")
LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class ToolDefinition:
    """Validated metadata and callable for one registered tool."""

    name: str
    description: str
    parameters: dict[str, Any]
    method: Any
    signature: inspect.Signature
    hints: dict[str, Any]
    metadata: ToolMetadata


@dataclass(frozen=True)
class ToolResult:
    """Normalized result returned by the tool runtime."""

    ok: bool
    name: str
    result: dict[str, Any]
    resource_usage: dict[str, int] = field(default_factory=dict)


class ToolRegistry:
    """Registry of validated, namespaced tool methods."""

    def __init__(self) -> None:
        self._tools: dict[str, ToolDefinition] = {}

    def register(self, package: ToolPackage) -> None:
        namespace = getattr(package, "namespace", "")
        if not NAME_RE.fullmatch(namespace):
            raise ValueError(f"invalid tool package namespace: {namespace!r}")

        for method_name, class_method in inspect.getmembers(
            type(package), predicate=inspect.isfunction
        ):
            if not is_tool(class_method):
                continue
            definition = build_definition(namespace, method_name, getattr(package, method_name))
            if definition.name in self._tools:
                raise ValueError(f"duplicate tool name: {definition.name}")
            self._tools[definition.name] = definition

    def execute(self, name: str, arguments: dict[str, Any]) -> ToolResult:
        definition = self._tools.get(name)
        if definition is None:
            return ToolResult(False, name, {"error": f"unknown tool: {name}"})
        try:
            bound = definition.signature.bind(**arguments)
            bound.apply_defaults()
            for parameter_name, value in bound.arguments.items():
                validate_value(value, definition.hints[parameter_name], parameter_name)
            value = definition.method(*bound.args, **bound.kwargs)
            if not isinstance(value, dict):
                raise TypeError("tool result must be a dictionary")
            normalized = dict(value)
            success = normalized.pop(TOOL_SUCCESS_MARKER, True)
            resource_usage = normalized.pop(TOOL_RESOURCE_USAGE_MARKER, {})
            if not isinstance(success, bool):
                raise TypeError("internal tool success marker must be a boolean")
            if not isinstance(resource_usage, dict) or any(
                not isinstance(resource, str)
                or not NAME_RE.fullmatch(resource)
                or not isinstance(amount, int)
                or isinstance(amount, bool)
                or amount < 0
                for resource, amount in resource_usage.items()
            ):
                raise TypeError("internal tool resource usage must contain non-negative integers")
            return ToolResult(success, name, normalized, dict(resource_usage))
        except (OSError, RuntimeError, UnicodeError, TypeError, ValueError) as exc:
            return ToolResult(False, name, {"error": str(exc)})

    def is_exposed(self, name: str, surface: str) -> bool:
        """Return whether a registered tool is visible on an execution surface."""
        definition = self._tools.get(name)
        if definition is None:
            return False
        if surface not in ("direct", "delegated"):
            raise ValueError(f"unknown tool exposure surface: {surface}")
        return bool(getattr(definition.metadata.exposure, surface))

    def prompt_instructions(self) -> str:
        lines = [
            "Local tools are available.",
            "Use a tool only when it is needed to satisfy the user's request.",
            "To call a tool, output exactly one tag containing exactly one JSON object:",
            '<tool_call>{"name":"package.tool","arguments":{"key":"value"}}</tool_call>',
            "Do not add prose, Markdown fences, or another JSON object to a tool call.",
            "Use JSON escaping inside strings, including \\n for newlines.",
            "After a tool result is provided, answer normally or call another tool.",
            "Do not claim that a tool succeeded until its result confirms it.",
            "",
            "Available tools:",
        ]
        for definition in self.definitions("direct"):
            schema = definition.parameters
            properties = schema["properties"]
            required = set(schema["required"])
            rendered_parameters = []
            for name, value in properties.items():
                status = "required" if name in required else "optional"
                rendered_parameters.append(f"{name}: {value['type']} ({status})")
            parameters = ", ".join(rendered_parameters) or "none"
            lines.append(f"- {definition.name}({parameters}): {definition.description}")
        prompt_instructions = list(
            dict.fromkeys(
                instruction
                for definition in self.definitions("direct")
                for instruction in definition.metadata.prompt_instructions
            )
        )
        if prompt_instructions:
            lines.extend(("", *prompt_instructions))
        return "\n".join(lines)

    def turn_guidance(self) -> str:
        """Build the same generic reliability reminder used by MCP clients."""
        roles = {
            role
            for definition in self.definitions("direct")
            for role in definition.metadata.epistemic_roles
        }
        guidance = {
            value
            for definition in self.definitions("direct")
            for value in definition.metadata.reliability_guidance
        }
        return render_reliability_turn_guidance(roles, guidance)

    def reliability_catalog(self) -> dict[str, list[str]]:
        """Return role-to-tool mappings for reliability assessment."""
        catalog: dict[str, list[str]] = {}
        for definition in self.definitions("direct"):
            for role in definition.metadata.epistemic_roles:
                catalog.setdefault(role, []).append(definition.name)
        return catalog

    def definitions(self, surface: str | None = None) -> tuple[ToolDefinition, ...]:
        """Return tools visible on a generic execution surface."""
        if surface not in (None, "direct", "delegated"):
            raise ValueError(f"unknown tool exposure surface: {surface}")
        definitions = tuple(self._tools[name] for name in sorted(self._tools))
        if surface is None:
            return definitions
        return tuple(
            definition
            for definition in definitions
            if getattr(definition.metadata.exposure, surface)
        )

    def api_definitions(self) -> list[dict[str, Any]]:
        """Return OpenAI-compatible function definitions for the active tools."""
        return [
            {
                "type": "function",
                "function": {
                    "name": definition.name,
                    "description": definition.description,
                    "parameters": {
                        **definition.parameters,
                        "additionalProperties": False,
                    },
                },
            }
            for definition in self.definitions("direct")
        ]

    @contextmanager
    def request_context(
        self, user_input: str, history: list[Any]
    ) -> Iterator[None]:
        """Expose the same request-scoped interface as an MCP tool collection."""
        with activate_tool_context(ToolContext(user_input, tuple(history))):
            yield


def discover_tool_packages(environment: ToolEnvironment) -> ToolRegistry:
    """Discover explicitly exported package classes in the tools package."""
    package = importlib.import_module("tools")
    registry = ToolRegistry()
    ignored = {"base", "runtime"}
    discovered: list[type[ToolPackage]] = []
    for module_info in sorted(pkgutil.iter_modules(package.__path__), key=lambda item: item.name):
        if module_info.name.startswith("_") or module_info.name in ignored:
            continue
        module = importlib.import_module(f"tools.{module_info.name}")
        package_classes = getattr(module, "TOOL_PACKAGES", ())
        for package_class in package_classes:
            if not inspect.isclass(package_class) or not issubclass(package_class, ToolPackage):
                raise TypeError(
                    f"tools.{module_info.name}.TOOL_PACKAGES must contain ToolPackage classes"
                )
            discovered.append(package_class)

    packages_by_namespace: dict[str, list[type[ToolPackage]]] = {}
    for package_class in discovered:
        packages_by_namespace.setdefault(package_class.namespace, []).append(package_class)
    for namespace, parameters in environment.tool_parameters.items():
        package_classes = packages_by_namespace.get(namespace)
        if package_classes is None:
            raise ValueError(f"unknown tool package in parameter override: {namespace}")
        unknown = sorted(
            name
            for name in parameters
            if not any(
                package_class.accepts_parameter(name)
                for package_class in package_classes
            )
        )
        if unknown:
            raise ValueError(
                f"unknown parameter for tool package {namespace}: {unknown[0]}"
            )

    environment.bind_tools(registry)
    for package_class in discovered:
        availability = package_class.check_availability(environment)
        if not availability.available:
            LOGGER.warning(
                'tool package "%s" (%s) is unavailable: %s',
                getattr(package_class, "namespace", package_class.__name__),
                package_class.__name__,
                availability.reason or "no reason provided",
            )
            continue
        registry.register(package_class(environment))
    return registry


def build_definition(namespace: str, method_name: str, method: Any) -> ToolDefinition:
    if not NAME_RE.fullmatch(method_name):
        raise ValueError(f"invalid tool method name: {method_name!r}")
    description = inspect.getdoc(method)
    if not description:
        raise TypeError(f"tool {namespace}.{method_name} must have a docstring")
    metadata = tool_metadata(method)
    validate_metadata(namespace, method_name, metadata)

    signature = inspect.signature(method)
    hints = get_type_hints(method)
    if "return" not in hints:
        raise TypeError(f"tool {namespace}.{method_name} must have a return type hint")
    if get_origin(hints["return"]) is not dict:
        raise TypeError(f"tool {namespace}.{method_name} must return a typed dictionary")

    properties: dict[str, Any] = {}
    required: list[str] = []
    for parameter in signature.parameters.values():
        if parameter.kind in (parameter.VAR_POSITIONAL, parameter.VAR_KEYWORD):
            raise TypeError(f"tool {namespace}.{method_name} cannot use *args or **kwargs")
        if parameter.name not in hints:
            raise TypeError(
                f"tool {namespace}.{method_name} parameter {parameter.name!r} "
                "must have a type hint"
            )
        property_schema = annotation_schema(hints[parameter.name])
        if parameter.default is inspect.Parameter.empty:
            required.append(parameter.name)
        else:
            validate_value(parameter.default, hints[parameter.name], parameter.name)
            try:
                json.dumps(parameter.default)
            except (TypeError, ValueError) as exc:
                raise TypeError(
                    f"tool {namespace}.{method_name} parameter {parameter.name!r} "
                    "must have a JSON-compatible default"
                ) from exc
            property_schema["default"] = parameter.default
        properties[parameter.name] = property_schema

    return ToolDefinition(
        name=f"{namespace}.{method_name}",
        description=description.split("\n\n", 1)[0].replace("\n", " "),
        parameters={"type": "object", "properties": properties, "required": required},
        method=method,
        signature=signature,
        hints=hints,
        metadata=metadata,
    )


def validate_metadata(
    namespace: str, method_name: str, metadata: ToolMetadata
) -> None:
    """Validate decorator metadata before it reaches transports or policies."""
    tool_name = f"{namespace}.{method_name}"
    for resource, amount in metadata.delegation.costs.items():
        if not isinstance(resource, str) or not NAME_RE.fullmatch(resource):
            raise TypeError(f"tool {tool_name} has an invalid resource name: {resource!r}")
        if not isinstance(amount, int) or isinstance(amount, bool) or amount < 1:
            raise TypeError(
                f"tool {tool_name} resource cost {resource!r} must be a positive integer"
            )
    for resource, limit in metadata.delegation.quota_defaults.items():
        if not isinstance(resource, str) or not NAME_RE.fullmatch(resource):
            raise TypeError(f"tool {tool_name} has an invalid quota name: {resource!r}")
        if not isinstance(limit, int) or isinstance(limit, bool) or limit < 1:
            raise TypeError(
                f"tool {tool_name} quota {resource!r} must be a positive integer"
            )
    if any(not value.strip() for value in metadata.delegation.instructions):
        raise TypeError(f"tool {tool_name} delegation instructions must not be empty")
    if metadata.delegation.reason is not None and not metadata.delegation.reason.strip():
        raise TypeError(f"tool {tool_name} delegation reason must not be empty")
    if any(not value.strip() for value in metadata.prompt_instructions):
        raise TypeError(f"tool {tool_name} prompt instructions must not be empty")
    for role in metadata.epistemic_roles:
        if not NAME_RE.fullmatch(role):
            raise TypeError(f"tool {tool_name} has an invalid epistemic role: {role!r}")
    if any(not value.strip() for value in metadata.reliability_guidance):
        raise TypeError(f"tool {tool_name} reliability guidance must not be empty")


def annotation_schema(annotation: Any) -> dict[str, Any]:
    origin = get_origin(annotation)
    if origin in (Union, types.UnionType):
        args = get_args(annotation)
        non_none = [value for value in args if value is not type(None)]
        if len(non_none) == 1 and len(non_none) != len(args):
            schema = annotation_schema(non_none[0])
            schema["nullable"] = True
            return schema
        raise TypeError(f"unsupported union type hint: {annotation!r}")
    if origin is list:
        args = get_args(annotation)
        return {"type": "array", "items": annotation_schema(args[0]) if args else {}}
    if origin is dict:
        return {"type": "object"}
    mapping = {str: "string", int: "integer", float: "number", bool: "boolean"}
    if annotation in mapping:
        return {"type": mapping[annotation]}
    raise TypeError(f"unsupported tool type hint: {annotation!r}")


def validate_value(value: Any, annotation: Any, parameter_name: str) -> None:
    origin = get_origin(annotation)
    if origin in (Union, types.UnionType):
        if value is None and type(None) in get_args(annotation):
            return
        candidates = [item for item in get_args(annotation) if item is not type(None)]
        if len(candidates) == 1:
            validate_value(value, candidates[0], parameter_name)
            return
    expected = origin or annotation
    if expected is int and isinstance(value, bool):
        raise TypeError(f"argument {parameter_name!r} must be an integer")
    if expected in (str, int, float, bool, list, dict) and not isinstance(value, expected):
        type_name = annotation_schema(annotation)["type"]
        raise TypeError(f"argument {parameter_name!r} must be a {type_name}")
    if origin is list and isinstance(value, list):
        args = get_args(annotation)
        if args:
            for index, item in enumerate(value):
                validate_value(item, args[0], f"{parameter_name}[{index}]")
