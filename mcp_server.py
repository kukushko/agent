#!/usr/bin/env python3.12
"""Standalone MCP server exposing the bundled terminal-agent tools."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from jobruntime import JobService
from mcpbridge import HttpMcpClient, McpToolCollection
from mcpbridge.protocol import decode_message, encode_message, error_payload
from mcpbridge.server import McpServer
from sandbox import DockerSandbox
from toolconfig import parse_mcp_servers, parse_tool_parameters
from tools import ServiceRegistry, ToolEnvironment, discover_tool_packages
from tools.jobs import JobRunner


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_SANDBOX_IMAGE = "terminal-agent-python312-sandbox:local"
DEFAULT_SANDBOX_CONTEXT = SCRIPT_DIR / "sandbox" / "python312"
DEFAULT_JOBS_DIR = SCRIPT_DIR / "tmp" / "jobs"
MAX_HTTP_REQUEST_BYTES = 4_194_304


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve bundled tools through MCP.")
    parser.add_argument(
        "--transport", choices=("stdio", "http"), default="stdio"
    )
    parser.add_argument(
        "--listen",
        default="127.0.0.1:9000",
        help="HTTP listen address, default: 127.0.0.1:9000",
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=Path("work"),
        help="directory exposed to bundled file tools, default: ./work",
    )
    parser.add_argument(
        "--tool-param",
        action="append",
        default=[],
        metavar="PACKAGE:NAME=VALUE",
        help="override a package-owned setting; may be repeated",
    )
    parser.add_argument(
        "--add-mcp",
        action="append",
        default=[],
        metavar="NAME=URL",
        help="add an external MCP dependency for jobs; may be repeated",
    )
    parser.add_argument(
        "--jobs-dir",
        type=Path,
        default=DEFAULT_JOBS_DIR,
        help="persistent job state directory, default: ./tmp/jobs beside this script",
    )
    args = parser.parse_args()
    try:
        args.tool_parameters = parse_tool_parameters(args.tool_param)
        args.mcp_servers = parse_mcp_servers(args.add_mcp)
        args.listen_address = parse_listen_address(args.listen)
    except ValueError as exc:
        parser.error(str(exc))
    return args


def parse_listen_address(value: str) -> tuple[str, int]:
    host, separator, raw_port = value.rpartition(":")
    if not separator or not host:
        raise ValueError("--listen must use HOST:PORT")
    try:
        port = int(raw_port)
    except ValueError as exc:
        raise ValueError("--listen port must be an integer") from exc
    if not 1 <= port <= 65535:
        raise ValueError("--listen port must be between 1 and 65535")
    return host, port


def create_server(args: argparse.Namespace) -> McpServer:
    sandbox = DockerSandbox(
        work_dir=args.work_dir,
        image=DEFAULT_SANDBOX_IMAGE,
        build_context=DEFAULT_SANDBOX_CONTEXT,
    )
    job_service = JobService(args.jobs_dir, sandbox)
    services = ServiceRegistry()
    services.register(JobRunner, job_service)
    environment = ToolEnvironment(
        work_dir=args.work_dir,
        sandbox=sandbox,
        tool_parameters=args.tool_parameters,
        services=services,
    )
    registry = discover_tool_packages(environment)
    external_tools = McpToolCollection()
    try:
        for name, url in args.mcp_servers:
            external_tools.add(HttpMcpClient(name, url), prefix=True)
        job_service.bind(registry, external_tools)
    except BaseException:
        external_tools.close()
        raise
    return McpServer(registry, (external_tools.close,))


def serve_stdio(server: McpServer) -> int:
    """Serve newline-delimited MCP JSON-RPC until the client closes stdin."""
    for raw_line in sys.stdin.buffer:
        if not raw_line.strip():
            continue
        try:
            request = decode_message(raw_line)
            response = server.handle(request)
        except Exception as exc:  # Keep malformed client input from killing the service.
            response = error_payload(None, -32700, str(exc))
        if response is not None:
            sys.stdout.buffer.write(encode_message(response))
            sys.stdout.buffer.flush()
    return 0


def handler_class(server: McpServer) -> type[BaseHTTPRequestHandler]:
    class McpHttpHandler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def do_POST(self) -> None:
            if self.path.rstrip("/") != "/mcp":
                self.send_error(404)
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                self.send_error(400, "invalid Content-Length")
                return
            if length < 1 or length > MAX_HTTP_REQUEST_BYTES:
                self.send_error(413 if length > MAX_HTTP_REQUEST_BYTES else 400)
                return
            try:
                request = decode_message(self.rfile.read(length))
                response = server.handle(request)
            except Exception as exc:
                response = error_payload(None, -32700, str(exc))
            if response is None:
                self.send_response(202)
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            body = json.dumps(response, ensure_ascii=False).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args: Any) -> None:
            logging.info("HTTP %s - %s", self.address_string(), format % args)

    return McpHttpHandler


def serve_http(server: McpServer, address: tuple[str, int]) -> int:
    httpd = ThreadingHTTPServer(address, handler_class(server))
    logging.info("MCP HTTP server listening on http://%s:%d/mcp", *address)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()
    return 0


def main() -> int:
    logging.basicConfig(level=logging.INFO, stream=sys.stderr, format="%(levelname)s: %(message)s")
    args = parse_args()
    server = create_server(args)
    try:
        if args.transport == "stdio":
            return serve_stdio(server)
        return serve_http(server, args.listen_address)
    except KeyboardInterrupt:
        return 130
    finally:
        server.close()


if __name__ == "__main__":
    raise SystemExit(main())
