"""Minimal MCP client and server support used by the terminal agent."""

from .client import HttpMcpClient, McpToolCollection, StdioMcpClient
from .protocol import McpError

__all__ = ["HttpMcpClient", "McpError", "McpToolCollection", "StdioMcpClient"]
