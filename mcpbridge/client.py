"""MCP transports and an aggregated tool facade for the terminal agent."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Iterator, Protocol
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from toolpolicy import render_reliability_turn_guidance

from .protocol import (
    LEGACY_PROTOCOL_VERSION,
    PROTOCOL_VERSION,
    McpError,
    decode_message,
    encode_message,
    notification_payload,
    request_payload,
)


@dataclass(frozen=True)
class McpTool:
    """One tool advertised by an MCP server."""

    name: str
    description: str
    input_schema: dict[str, Any]
    metadata: dict[str, Any] | None = None


@dataclass(frozen=True)
class McpToolResult:
    """Normalized result consumed by the agent completion loop."""

    ok: bool
    name: str
    result: dict[str, Any]
    resource_usage: dict[str, int] = field(default_factory=dict)


class McpConnection(Protocol):
    name: str
    trusted_context: bool

    def initialize(self) -> tuple[McpTool, ...]: ...

    def call_tool(
        self, name: str, arguments: dict[str, Any], context: dict[str, Any] | None
    ) -> McpToolResult: ...

    def close(self) -> None: ...


class JsonRpcMcpClient:
    """Common request sequencing and response validation for MCP transports."""

    def __init__(self, name: str, trusted_context: bool = False) -> None:
        self.name = name
        self.trusted_context = trusted_context
        self._next_id = 1
        self._lock = threading.Lock()
        self.protocol_version = PROTOCOL_VERSION
        self.delegated_tools: tuple[McpTool, ...] = ()

    def initialize(self) -> tuple[McpTool, ...]:
        try:
            discovered = self._request("server/discover", {})
            supported = discovered.get("supportedVersions")
            if not isinstance(supported, list) or PROTOCOL_VERSION not in supported:
                raise McpError(
                    f"MCP server {self.name!r} does not advertise {PROTOCOL_VERSION}"
                )
        except McpError:
            self.protocol_version = LEGACY_PROTOCOL_VERSION
            response = self._request(
                "initialize",
                {
                    "protocolVersion": LEGACY_PROTOCOL_VERSION,
                    "capabilities": {},
                    "clientInfo": {"name": "terminal-agent", "version": "1.0"},
                },
            )
            if not isinstance(response.get("protocolVersion"), str):
                raise McpError(f"MCP server {self.name!r} returned invalid initialization data")
            self._notify("notifications/initialized", {})
        tools: list[McpTool] = []
        delegated_tools: list[McpTool] = []
        cursor: str | None = None
        while True:
            listed = self._request(
                "tools/list", {} if cursor is None else {"cursor": cursor}
            )
            raw_tools = listed.get("tools")
            if not isinstance(raw_tools, list):
                raise McpError(f"MCP server {self.name!r} returned invalid tools/list data")
            for raw_tool in raw_tools:
                if not isinstance(raw_tool, dict):
                    raise McpError(f"MCP server {self.name!r} returned an invalid tool")
                name = raw_tool.get("name")
                description = raw_tool.get("description", "")
                schema = raw_tool.get("inputSchema")
                if not isinstance(name, str) or not isinstance(description, str) or not isinstance(schema, dict):
                    raise McpError(f"MCP server {self.name!r} returned an invalid tool definition")
                raw_meta = raw_tool.get("_meta", {})
                metadata = (
                    raw_meta.get("terminal-agent/tool", {})
                    if isinstance(raw_meta, dict)
                    else {}
                )
                if not isinstance(metadata, dict):
                    metadata = {}
                tools.append(McpTool(name, description, schema, metadata))
            raw_list_meta = listed.get("_meta", {})
            raw_delegated = (
                raw_list_meta.get("terminal-agent/delegatedTools", [])
                if isinstance(raw_list_meta, dict)
                else []
            )
            if isinstance(raw_delegated, list):
                delegated_tools.extend(parse_tool_descriptors(raw_delegated, self.name))
            next_cursor = listed.get("nextCursor")
            if next_cursor is None:
                break
            if not isinstance(next_cursor, str) or not next_cursor:
                raise McpError(f"MCP server {self.name!r} returned an invalid tools cursor")
            cursor = next_cursor
        self.delegated_tools = tuple(delegated_tools)
        return tuple(tools)

    def call_tool(
        self, name: str, arguments: dict[str, Any], context: dict[str, Any] | None
    ) -> McpToolResult:
        params: dict[str, Any] = {"name": name, "arguments": arguments}
        if self.trusted_context and context is not None:
            params["_meta"] = {"terminal-agent/context": context}
        response = self._request("tools/call", params)
        structured = response.get("structuredContent")
        if isinstance(structured, dict):
            envelope_ok = structured.get("ok")
            envelope_result = structured.get("result")
            returned_name = structured.get("name", name)
            resource_usage = structured.get("resource_usage", {})
            if (
                isinstance(envelope_ok, bool)
                and isinstance(envelope_result, dict)
                and isinstance(returned_name, str)
            ):
                return McpToolResult(
                    envelope_ok,
                    returned_name,
                    envelope_result,
                    resource_usage if isinstance(resource_usage, dict) else {},
                )
            return McpToolResult(not bool(response.get("isError")), name, structured)
        return McpToolResult(
            not bool(response.get("isError")), name, normalize_content_result(response)
        )

    def _request(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        params = self._with_request_meta(params)
        with self._lock:
            request_id = self._next_id
            self._next_id += 1
            message = self._exchange(request_payload(request_id, method, params), True)
        if message is None or message.get("id") != request_id:
            raise McpError(f"MCP server {self.name!r} returned a mismatched response")
        error = message.get("error")
        if isinstance(error, dict):
            raise McpError(f"MCP {method} failed on {self.name}: {error.get('message', error)}")
        result = message.get("result")
        if not isinstance(result, dict):
            raise McpError(f"MCP server {self.name!r} returned a response without an object result")
        return result

    def _with_request_meta(self, params: dict[str, Any]) -> dict[str, Any]:
        if self.protocol_version != PROTOCOL_VERSION:
            return params
        raw_meta = params.get("_meta", {})
        meta = dict(raw_meta) if isinstance(raw_meta, dict) else {}
        meta.update(
            {
                "io.modelcontextprotocol/protocolVersion": PROTOCOL_VERSION,
                "io.modelcontextprotocol/clientInfo": {
                    "name": "terminal-agent",
                    "version": "1.0",
                },
                "io.modelcontextprotocol/clientCapabilities": {},
            }
        )
        return {**params, "_meta": meta}

    def _notify(self, method: str, params: dict[str, Any]) -> None:
        with self._lock:
            self._exchange(notification_payload(method, params), False)

    def _exchange(
        self, payload: dict[str, Any], expect_response: bool
    ) -> dict[str, Any] | None:
        raise NotImplementedError


class StdioMcpClient(JsonRpcMcpClient):
    """MCP client connected to a managed child process over stdio."""

    def __init__(
        self,
        name: str,
        command: list[str],
        trusted_context: bool = False,
        environment: dict[str, str] | None = None,
    ) -> None:
        super().__init__(name, trusted_context)
        self.process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0,
            close_fds=True,
            env={**os.environ, **environment} if environment is not None else None,
        )
        self._stderr_thread = threading.Thread(target=self._forward_stderr, daemon=True)
        self._stderr_thread.start()

    def _exchange(
        self, payload: dict[str, Any], expect_response: bool
    ) -> dict[str, Any] | None:
        if self.process.stdin is None or self.process.stdout is None:
            raise McpError(f"MCP server {self.name!r} has no stdio transport")
        if self.process.poll() is not None:
            raise McpError(f"MCP server {self.name!r} exited with code {self.process.returncode}")
        try:
            self.process.stdin.write(encode_message(payload))
            self.process.stdin.flush()
            if not expect_response:
                return None
            line = self.process.stdout.readline()
        except (BrokenPipeError, OSError) as exc:
            raise McpError(f"MCP stdio transport {self.name!r} failed: {exc}") from exc
        if not line:
            raise McpError(f"MCP server {self.name!r} closed stdout")
        return decode_message(line)

    def close(self) -> None:
        if self.process.stdin is not None and not self.process.stdin.closed:
            try:
                self.process.stdin.close()
            except OSError:
                pass
        if self.process.poll() is None:
            try:
                self.process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self.process.terminate()
                try:
                    self.process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait()
        self._stderr_thread.join(timeout=1)
        for stream in (self.process.stdout, self.process.stderr):
            if stream is not None and not stream.closed:
                stream.close()

    def _forward_stderr(self) -> None:
        if self.process.stderr is None:
            return
        for line in iter(self.process.stderr.readline, b""):
            sys.stderr.write(f"[mcp:{self.name}] {line.decode('utf-8', errors='replace')}")
            sys.stderr.flush()


class HttpMcpClient(JsonRpcMcpClient):
    """MCP client using the Streamable HTTP request/response subset."""

    def __init__(self, name: str, url: str, timeout: float = 30.0) -> None:
        super().__init__(name, False)
        if not url.startswith(("http://", "https://")):
            raise ValueError("external MCP URL must use http:// or https://")
        self.url = url
        self.timeout = timeout
        self.session_id: str | None = None

    def _exchange(
        self, payload: dict[str, Any], expect_response: bool
    ) -> dict[str, Any] | None:
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
            "MCP-Protocol-Version": self.protocol_version,
        }
        if self.protocol_version == PROTOCOL_VERSION:
            method = payload.get("method")
            if isinstance(method, str):
                headers["Mcp-Method"] = method
            params = payload.get("params")
            if isinstance(params, dict) and isinstance(params.get("name"), str):
                headers["Mcp-Name"] = params["name"]
        if self.session_id is not None:
            headers["Mcp-Session-Id"] = self.session_id
        request = Request(
            self.url,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        try:
            with urlopen(request, timeout=self.timeout) as response:
                session_id = response.headers.get("Mcp-Session-Id")
                if session_id:
                    self.session_id = session_id
                body = response.read(4_194_305)
                content_type = response.headers.get_content_type()
        except HTTPError as exc:
            raise McpError(f"MCP server {self.name!r} returned HTTP {exc.code}") from exc
        except URLError as exc:
            raise McpError(f"MCP server {self.name!r} request failed: {exc.reason}") from exc
        if not expect_response or not body:
            return None
        if len(body) > 4_194_304:
            raise McpError(f"MCP server {self.name!r} response is too large")
        if content_type == "text/event-stream":
            body = extract_sse_data(body)
        return decode_message(body)

    def close(self) -> None:
        return None


class McpToolCollection:
    """Aggregate tools from embedded and external MCP servers."""

    def __init__(self, direct_mode: str = "hybrid") -> None:
        if direct_mode not in ("hybrid", "jobs"):
            raise ValueError(f"unknown direct tool mode: {direct_mode}")
        self.direct_mode = direct_mode
        self._connections: list[McpConnection] = []
        self._tools: dict[str, tuple[McpConnection, McpTool]] = {}
        self._delegated_tools: dict[str, tuple[McpConnection, McpTool]] = {}
        self._context: dict[str, Any] | None = None

    def add(self, connection: McpConnection, prefix: bool) -> None:
        try:
            tools = connection.initialize()
        except BaseException:
            connection.close()
            raise
        pending: dict[str, tuple[McpConnection, McpTool]] = {}
        for tool in tools:
            exposed_name = f"{connection.name}.{tool.name}" if prefix else tool.name
            if (
                exposed_name in self._tools
                or exposed_name in self._delegated_tools
                or exposed_name in pending
            ):
                connection.close()
                raise McpError(f"duplicate MCP tool name: {exposed_name}")
            pending[exposed_name] = (connection, tool)
        pending_delegated = dict(pending)
        for tool in getattr(connection, "delegated_tools", ()):
            exposed_name = f"{connection.name}.{tool.name}" if prefix else tool.name
            if exposed_name in pending_delegated or exposed_name in self._delegated_tools:
                connection.close()
                raise McpError(f"duplicate delegated MCP tool name: {exposed_name}")
            pending_delegated[exposed_name] = (connection, tool)
        self._connections.append(connection)
        self._tools.update(pending)
        self._delegated_tools.update(pending_delegated)

    @property
    def server_count(self) -> int:
        return len(self._connections)

    @property
    def tool_count(self) -> int:
        return len(self._active_tools())

    def execute(self, name: str, arguments: dict[str, Any]) -> McpToolResult:
        entry = self._active_tools().get(name)
        if entry is None:
            return McpToolResult(False, name, {"error": f"unknown tool: {name}"})
        connection, tool = entry
        try:
            result = connection.call_tool(tool.name, arguments, self._context)
            return McpToolResult(result.ok, name, result.result, result.resource_usage)
        except (McpError, OSError, RuntimeError, TypeError, ValueError) as exc:
            return McpToolResult(False, name, {"error": str(exc)})

    def api_definitions(self) -> list[dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": tool.description,
                    "parameters": tool.input_schema,
                },
            }
            for name, (_, tool) in sorted(self._active_tools().items())
        ]

    def tool_descriptors(self) -> tuple[McpTool, ...]:
        """Return exposed names and schemas for internal routing or job catalogs."""
        return tuple(
            McpTool(name, tool.description, tool.input_schema, tool.metadata)
            for name, (_, tool) in sorted(self._delegated_tools.items())
        )

    def prompt_instructions(self) -> str:
        active_tools = self._active_tools()
        lines = [
            "Tools are available through MCP servers.",
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
        if self.direct_mode == "jobs":
            lines.extend(
                (
                    "The direct catalog is in jobs mode: use an orchestration entrypoint "
                    "for tasks that need tools, and call delegated tools from its program.",
                    "Before calling an entrypoint, construct one end-to-end program that "
                    "performs all predictable tool steps and returns a compact final result. "
                    "Do not use separate entrypoint calls merely to inspect each intermediate result.",
                )
            )
        for name, (_, tool) in sorted(active_tools.items()):
            properties = tool.input_schema.get("properties", {})
            required = set(tool.input_schema.get("required", []))
            rendered = [
                f"{parameter}: {schema.get('type', 'value')} "
                f"({'required' if parameter in required else 'optional'})"
                for parameter, schema in properties.items()
                if isinstance(schema, dict)
            ]
            lines.append(f"- {name}({', '.join(rendered) or 'none'}): {tool.description}")
        tools_by_role = self.tools_by_epistemic_role()
        if tools_by_role:
            lines.extend(
                reliability_instructions(
                    tools_by_role, self.reliability_guidance()
                )
            )
        prompt_instructions: list[str] = []
        instruction_tools = {
            **active_tools,
            **self._delegated_tools,
        }
        for _, tool in sorted(
            instruction_tools.values(), key=lambda item: item[1].name
        ):
            metadata = tool.metadata or {}
            raw_instructions = metadata.get("prompt_instructions", [])
            if not isinstance(raw_instructions, list):
                continue
            for instruction in raw_instructions:
                if isinstance(instruction, str) and instruction not in prompt_instructions:
                    prompt_instructions.append(instruction)
        if prompt_instructions:
            lines.extend(("", *prompt_instructions))
            delegated = [
                name
                for name, (_, tool) in sorted(self._delegated_tools.items())
                if tool_is_delegable(tool)
            ]
            if delegated:
                lines.extend(("", "Tools available to delegated execution:"))
                lines.extend(
                    f"- {delegated_signature(name, self._delegated_tools[name][1])}"
                    for name in delegated
                )
        return "\n".join(lines)

    def tools_by_epistemic_role(self) -> dict[str, list[str]]:
        """Group exposed tools by portable reliability role metadata."""
        tools_by_role: dict[str, list[str]] = {}
        for name, (_, tool) in sorted(self._active_tools().items()):
            metadata = tool.metadata or {}
            raw_roles = metadata.get("epistemic_roles", [])
            if not isinstance(raw_roles, list):
                continue
            for role in raw_roles:
                if isinstance(role, str) and role:
                    tools_by_role.setdefault(role, []).append(name)
        return tools_by_role

    def turn_guidance(self) -> str:
        """Return a salient per-turn reminder when reliability roles exist."""
        return render_reliability_turn_guidance(
            self.tools_by_epistemic_role(), self.reliability_guidance()
        )

    def reliability_catalog(self) -> dict[str, list[str]]:
        """Return role-to-tool mappings for a stream-independent preflight."""
        return self.tools_by_epistemic_role()

    def reliability_guidance(self) -> set[str]:
        """Collect opaque reliability guidance owned by exposed tools."""
        guidance: set[str] = set()
        for _, tool in self._active_tools().values():
            metadata = tool.metadata or {}
            raw_guidance = metadata.get("reliability_guidance", [])
            if isinstance(raw_guidance, list):
                guidance.update(
                    value
                    for value in raw_guidance
                    if isinstance(value, str) and value
                )
        return guidance

    @contextmanager
    def request_context(
        self, user_input: str, history: list[Any]
    ) -> Iterator[None]:
        previous = self._context
        self._context = {
            "user_input": user_input,
            "history": [
                {"role": message.role, "content": message.content} for message in history
            ],
        }
        try:
            yield
        finally:
            self._context = previous

    def close(self) -> None:
        for connection in reversed(self._connections):
            connection.close()
        self._connections.clear()
        self._tools.clear()
        self._delegated_tools.clear()

    def _active_tools(self) -> dict[str, tuple[McpConnection, McpTool]]:
        if self.direct_mode == "hybrid":
            return self._tools
        return {
            name: value
            for name, value in self._tools.items()
            if tool_is_orchestration_entrypoint(value[1])
        }


def parse_tool_descriptors(raw_tools: list[Any], server_name: str) -> list[McpTool]:
    """Parse tool descriptors carried by a portable metadata extension."""
    parsed: list[McpTool] = []
    for raw_tool in raw_tools:
        if not isinstance(raw_tool, dict):
            raise McpError(f"MCP server {server_name!r} returned an invalid delegated tool")
        name = raw_tool.get("name")
        description = raw_tool.get("description", "")
        schema = raw_tool.get("inputSchema")
        raw_meta = raw_tool.get("_meta", {})
        metadata = raw_meta.get("terminal-agent/tool", {}) if isinstance(raw_meta, dict) else {}
        if (
            not isinstance(name, str)
            or not isinstance(description, str)
            or not isinstance(schema, dict)
            or not isinstance(metadata, dict)
        ):
            raise McpError(f"MCP server {server_name!r} returned an invalid delegated tool")
        parsed.append(McpTool(name, description, schema, metadata))
    return parsed


def normalize_content_result(response: dict[str, Any]) -> dict[str, Any]:
    content = response.get("content")
    if not isinstance(content, list):
        raise McpError("MCP tool result has no structuredContent or content")
    text_values: list[str] = []
    for item in content:
        if isinstance(item, dict) and item.get("type") == "text" and isinstance(item.get("text"), str):
            text_values.append(item["text"])
            try:
                parsed = json.loads(item["text"])
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict):
                return parsed
    if len(text_values) == 1:
        return {"content": text_values[0]}
    return {"content": content}


def tool_is_delegable(tool: McpTool) -> bool:
    """Read portable delegation metadata with compatible external defaults."""
    metadata = tool.metadata or {}
    delegation = metadata.get("delegation", {})
    return not isinstance(delegation, dict) or delegation.get("allowed", True) is True


def tool_is_orchestration_entrypoint(tool: McpTool) -> bool:
    """Read portable direct-mode metadata with conservative external defaults."""
    metadata = tool.metadata or {}
    exposure = metadata.get("exposure", {})
    return (
        isinstance(exposure, dict)
        and exposure.get("orchestration_entrypoint", False) is True
    )


def reliability_instructions(
    tools_by_role: dict[str, list[str]], guidance: set[str]
) -> list[str]:
    """Render soft evidence policy from tool-owned epistemic roles."""
    lines = [
        "",
        "Reliability and tool use:",
        "- Use tools to improve answer reliability, not only when they are strictly required.",
        "- Do not answer solely from memory when an available tool role directly covers a material fact or calculation. Tool use is the default in that case, even when you believe you know the answer.",
        *[f"- {value}" for value in sorted(guidance)],
        "- If an optional tool is unavailable or fails, continue with your best knowledge and identify material assumptions or approximations.",
        "- Do not repeat optional verification indefinitely, and do not use a tool when it would not meaningfully improve the answer.",
        "",
        "Available tool roles:",
    ]
    lines.extend(
        f"- {role}: {', '.join(names)}"
        for role, names in sorted(tools_by_role.items())
    )
    return lines


def delegated_signature(name: str, tool: McpTool) -> str:
    """Render a Python proxy signature from a portable MCP input schema."""
    properties = tool.input_schema.get("properties", {})
    required = set(tool.input_schema.get("required", []))
    parameters: list[str] = []
    if isinstance(properties, dict):
        for parameter, schema in properties.items():
            if not isinstance(schema, dict):
                continue
            rendered = f"{parameter}: {schema.get('type', 'value')}"
            if parameter not in required:
                rendered += f" = {schema.get('default', None)!r}"
            parameters.append(rendered)
    signature = f"tools.{name}({', '.join(parameters)})"
    metadata = tool.metadata or {}
    delegation = metadata.get("delegation", {})
    instructions = delegation.get("instructions", []) if isinstance(delegation, dict) else []
    if isinstance(instructions, list):
        details = " ".join(
            value for value in instructions if isinstance(value, str) and value
        )
        if details:
            return f"{signature}: {details}"
    return signature


def extract_sse_data(body: bytes) -> bytes:
    data_lines = [
        line[5:].lstrip()
        for line in body.splitlines()
        if line.startswith(b"data:")
    ]
    if not data_lines:
        raise McpError("MCP event stream contains no data event")
    return b"\n".join(data_lines)
