"""Parsing for package-owned tool configuration overrides."""

from __future__ import annotations

import re


TOOL_PARAMETER_RE = re.compile(
    r"^(?P<namespace>[a-z][a-z0-9_]*):(?P<name>[a-z][a-z0-9_]*)=(?P<value>.*)$"
)
MCP_SERVER_RE = re.compile(r"^(?P<name>[a-z][a-z0-9_-]*)=(?P<url>https?://.+)$")


def parse_tool_parameters(values: list[str]) -> dict[str, dict[str, str]]:
    """Parse repeated PACKAGE:NAME=VALUE tool configuration overrides."""
    parameters: dict[str, dict[str, str]] = {}
    for value in values:
        match = TOOL_PARAMETER_RE.fullmatch(value)
        if match is None:
            raise ValueError(
                f"invalid --tool-param {value!r}; expected PACKAGE:NAME=VALUE"
            )
        namespace = match.group("namespace")
        name = match.group("name")
        package_parameters = parameters.setdefault(namespace, {})
        if name in package_parameters:
            raise ValueError(f"duplicate --tool-param override: {namespace}:{name}")
        package_parameters[name] = match.group("value")
    return parameters


def parse_mcp_servers(values: list[str]) -> list[tuple[str, str]]:
    """Parse repeated NAME=URL external MCP server specifications."""
    servers: list[tuple[str, str]] = []
    names: set[str] = set()
    for value in values:
        match = MCP_SERVER_RE.fullmatch(value)
        if match is None:
            raise ValueError(f"invalid --add-mcp {value!r}; expected NAME=http[s]://URL")
        name, url = match.group("name"), match.group("url")
        if name == "embedded":
            raise ValueError("external MCP server name 'embedded' is reserved")
        if name in names:
            raise ValueError(f"duplicate --add-mcp server name: {name}")
        names.add(name)
        servers.append((name, url))
    return servers
