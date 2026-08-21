"""Shared MCP JSON-RPC helpers."""

from __future__ import annotations

import json
from typing import Any


PROTOCOL_VERSION = "2026-07-28"
LEGACY_PROTOCOL_VERSION = "2025-11-25"
JSONRPC_VERSION = "2.0"


class McpError(RuntimeError):
    """Protocol or transport failure while communicating with an MCP server."""


def request_payload(request_id: int, method: str, params: dict[str, Any]) -> dict[str, Any]:
    return {
        "jsonrpc": JSONRPC_VERSION,
        "id": request_id,
        "method": method,
        "params": params,
    }


def notification_payload(method: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
    payload: dict[str, Any] = {"jsonrpc": JSONRPC_VERSION, "method": method}
    if params is not None:
        payload["params"] = params
    return payload


def success_payload(request_id: Any, result: dict[str, Any]) -> dict[str, Any]:
    return {"jsonrpc": JSONRPC_VERSION, "id": request_id, "result": result}


def error_payload(request_id: Any, code: int, message: str) -> dict[str, Any]:
    return {
        "jsonrpc": JSONRPC_VERSION,
        "id": request_id,
        "error": {"code": code, "message": message},
    }


def encode_message(payload: dict[str, Any]) -> bytes:
    return (json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n").encode(
        "utf-8"
    )


def decode_message(payload: bytes | str) -> dict[str, Any]:
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise McpError(f"invalid MCP JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise McpError("MCP message must be a JSON object")
    return value
