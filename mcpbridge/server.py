"""MCP request handling backed by the local tool registry."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable

from tools import ToolContext, ToolRegistry, activate_tool_context

from .protocol import (
    LEGACY_PROTOCOL_VERSION,
    PROTOCOL_VERSION,
    error_payload,
    success_payload,
)


@dataclass(frozen=True)
class ContextMessage:
    """Minimal history message reconstructed from trusted request metadata."""

    role: str
    content: str


class McpServer:
    """Expose a ToolRegistry through the MCP tools protocol."""

    def __init__(
        self,
        registry: ToolRegistry,
        close_callbacks: tuple[Callable[[], None], ...] = (),
    ) -> None:
        self.registry = registry
        self.close_callbacks = close_callbacks

    def close(self) -> None:
        for callback in reversed(self.close_callbacks):
            callback()

    def handle(self, message: dict[str, Any]) -> dict[str, Any] | None:
        request_id = message.get("id")
        method = message.get("method")
        if not isinstance(method, str):
            return error_payload(request_id, -32600, "invalid JSON-RPC request")
        if request_id is None:
            return None
        params = message.get("params", {})
        if not isinstance(params, dict):
            return error_payload(request_id, -32602, "params must be an object")
        try:
            modern = is_modern_request(params)
            if method == "server/discover":
                return success_payload(request_id, self._discover())
            if method == "initialize":
                return success_payload(request_id, self._initialize(params))
            if method == "ping":
                return success_payload(request_id, {})
            if method == "tools/list":
                return success_payload(
                    request_id, modern_result(self._list_tools(), cacheable=True) if modern else self._list_tools()
                )
            if method == "tools/call":
                result = self._call_tool(params)
                return success_payload(
                    request_id, modern_result(result) if modern else result
                )
            return error_payload(request_id, -32601, f"method not found: {method}")
        except (TypeError, ValueError) as exc:
            return error_payload(request_id, -32602, str(exc))
        except RuntimeError as exc:
            return error_payload(request_id, -32603, str(exc))

    def _initialize(self, params: dict[str, Any]) -> dict[str, Any]:
        requested = params.get("protocolVersion")
        selected = requested if isinstance(requested, str) else LEGACY_PROTOCOL_VERSION
        return {
            "protocolVersion": selected,
            "capabilities": {"tools": {"listChanged": False}},
            "serverInfo": {"name": "terminal-agent-tools", "version": "1.0"},
        }

    def _discover(self) -> dict[str, Any]:
        return {
            "resultType": "complete",
            "supportedVersions": [PROTOCOL_VERSION],
            "capabilities": {"tools": {}},
            "ttlMs": 0,
            "cacheScope": "private",
            "_meta": {
                "io.modelcontextprotocol/serverInfo": {
                    "name": "terminal-agent-tools",
                    "version": "1.0",
                }
            },
        }

    def _list_tools(self) -> dict[str, Any]:
        return {
            "tools": [
                tool_descriptor(definition)
                for definition in self.registry.definitions("direct")
            ],
            "_meta": {
                "terminal-agent/delegatedTools": [
                    tool_descriptor(definition)
                    for definition in self.registry.definitions("delegated")
                    if not definition.metadata.exposure.direct
                ]
            },
        }

    def _call_tool(self, params: dict[str, Any]) -> dict[str, Any]:
        name = params.get("name")
        arguments = params.get("arguments", {})
        if not isinstance(name, str):
            raise ValueError("tool name must be a string")
        if not isinstance(arguments, dict):
            raise ValueError("tool arguments must be an object")
        if not self.registry.is_exposed(name, "direct"):
            raise ValueError(f"tool is not available for direct calls: {name}")
        context = context_from_meta(params.get("_meta"))
        if context is None:
            result = self.registry.execute(name, arguments)
        else:
            with activate_tool_context(context):
                result = self.registry.execute(name, arguments)
        structured = {
            "ok": result.ok,
            "name": result.name,
            "result": result.result,
            "resource_usage": result.resource_usage,
        }
        return {
            "content": [
                {
                    "type": "text",
                    "text": json.dumps(structured, ensure_ascii=False),
                }
            ],
            "structuredContent": structured,
            "isError": not result.ok,
        }


def context_from_meta(raw_meta: Any) -> ToolContext | None:
    """Decode request context supplied only by the trusted embedded client."""
    if not isinstance(raw_meta, dict):
        return None
    raw_context = raw_meta.get("terminal-agent/context")
    if not isinstance(raw_context, dict):
        return None
    user_input = raw_context.get("user_input")
    raw_history = raw_context.get("history", [])
    if not isinstance(user_input, str) or not isinstance(raw_history, list):
        return None
    history: list[ContextMessage] = []
    for item in raw_history:
        if not isinstance(item, dict):
            continue
        role, content = item.get("role"), item.get("content")
        if isinstance(role, str) and isinstance(content, str):
            history.append(ContextMessage(role, content))
    return ToolContext(user_input=user_input, history=tuple(history))


def is_modern_request(params: dict[str, Any]) -> bool:
    meta = params.get("_meta")
    return isinstance(meta, dict) and meta.get(
        "io.modelcontextprotocol/protocolVersion"
    ) == PROTOCOL_VERSION


def modern_result(
    result: dict[str, Any], cacheable: bool = False
) -> dict[str, Any]:
    existing_meta = result.get("_meta", {})
    modern = {
        **result,
        "resultType": "complete",
        "_meta": {
            **(existing_meta if isinstance(existing_meta, dict) else {}),
            "io.modelcontextprotocol/serverInfo": {
                "name": "terminal-agent-tools",
                "version": "1.0",
            }
        },
    }
    if cacheable:
        modern.update({"ttlMs": 0, "cacheScope": "private"})
    return modern


def tool_descriptor(definition: Any) -> dict[str, Any]:
    """Serialize one tool declaration for direct or delegated discovery."""
    return {
        "name": definition.name,
        "description": definition.description,
        "inputSchema": {
            **definition.parameters,
            "additionalProperties": False,
        },
        "_meta": {
            "terminal-agent/tool": {
                "delegation": {
                    "allowed": definition.metadata.delegation.allowed,
                    "costs": dict(definition.metadata.delegation.costs),
                    "quota_defaults": dict(definition.metadata.delegation.quota_defaults),
                    "instructions": list(definition.metadata.delegation.instructions),
                    "reason": definition.metadata.delegation.reason,
                },
                "exposure": {
                    "direct": definition.metadata.exposure.direct,
                    "delegated": definition.metadata.exposure.delegated,
                    "orchestration_entrypoint": (
                        definition.metadata.exposure.orchestration_entrypoint
                    ),
                },
                "prompt_instructions": list(definition.metadata.prompt_instructions),
                "epistemic_roles": list(definition.metadata.epistemic_roles),
                "reliability_guidance": list(definition.metadata.reliability_guidance),
            }
        },
    }
