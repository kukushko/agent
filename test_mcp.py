import socket
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

from agent import ChatMessage, parse_mcp_servers
from mcpbridge import HttpMcpClient, McpToolCollection, StdioMcpClient
from mcpbridge.client import JsonRpcMcpClient
from mcpbridge.protocol import McpError


PROJECT_DIR = Path(__file__).resolve().parent


class ScriptedMcpClient(JsonRpcMcpClient):
    def __init__(self, responses):
        super().__init__("scripted")
        self.responses = iter(responses)

    def _exchange(self, payload, expect_response):
        if not expect_response:
            return None
        response = next(self.responses)
        if isinstance(response, Exception):
            raise response
        return {"jsonrpc": "2.0", "id": payload["id"], "result": response}

    def close(self):
        return None


class McpProtocolTest(unittest.TestCase):
    def test_external_server_specs_are_named_and_unique(self) -> None:
        self.assertEqual(
            parse_mcp_servers(["docs=http://localhost:9000/mcp"]),
            [("docs", "http://localhost:9000/mcp")],
        )
        with self.assertRaisesRegex(ValueError, "duplicate"):
            parse_mcp_servers(
                ["docs=http://one/mcp", "docs=https://two/mcp"]
            )
        with self.assertRaisesRegex(ValueError, "reserved"):
            parse_mcp_servers(["embedded=http://localhost/mcp"])

    def test_client_accepts_standard_unstructured_tool_content(self) -> None:
        client = ScriptedMcpClient(
            [
                {
                    "supportedVersions": ["2026-07-28"],
                    "capabilities": {"tools": {}},
                    "resultType": "complete",
                },
                {
                    "tools": [
                        {
                            "name": "echo",
                            "description": "Echo text.",
                            "inputSchema": {"type": "object"},
                            "_meta": {
                                "terminal-agent/tool": {
                                    "delegation": {
                                        "allowed": False,
                                        "costs": {},
                                        "reason": "interactive only",
                                    }
                                }
                            },
                        }
                    ]
                },
                {"content": [{"type": "text", "text": "hello"}]},
            ]
        )
        tools = client.initialize()
        result = client.call_tool("echo", {}, None)

        self.assertEqual(tools[0].name, "echo")
        self.assertFalse(tools[0].metadata["delegation"]["allowed"])
        self.assertTrue(result.ok)
        self.assertEqual(result.result, {"content": "hello"})

    def test_client_falls_back_to_legacy_initialization(self) -> None:
        client = ScriptedMcpClient(
            [
                McpError("method not found"),
                {
                    "protocolVersion": "2025-11-25",
                    "capabilities": {"tools": {}},
                    "serverInfo": {"name": "legacy", "version": "1"},
                },
                {"tools": []},
            ]
        )

        self.assertEqual(client.initialize(), ())
        self.assertEqual(client.protocol_version, "2025-11-25")


class McpTransportIntegrationTest(unittest.TestCase):
    def test_embedded_stdio_server_lists_calls_and_exits_on_eof(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            command = [
                sys.executable,
                str(PROJECT_DIR / "mcp_server.py"),
                "--transport",
                "stdio",
                "--work-dir",
                tmp,
                "--jobs-dir",
                str(Path(tmp) / "jobs"),
            ]
            connection = StdioMcpClient(
                "embedded", command, trusted_context=True
            )
            collection = McpToolCollection()
            try:
                collection.add(connection, prefix=False)
                self.assertGreaterEqual(collection.tool_count, 6)
                written = collection.execute(
                    "files.write", {"path": "result.txt", "content": "hello"}
                )
                with collection.request_context(
                    "remember", [ChatMessage("user", "remember")]
                ):
                    history = collection.execute(
                        "files.write_history", {"path": "history.txt"}
                    )
                self.assertTrue(written.ok)
                self.assertTrue(history.ok)
                self.assertEqual(
                    (Path(tmp) / "result.txt").read_text(encoding="utf-8"), "hello"
                )
                self.assertIn(
                    "USER:\nremember",
                    (Path(tmp) / "history.txt").read_text(encoding="utf-8"),
                )
                names = {tool.name for tool in collection.tool_descriptors()}
                if "jobs.run" in names:
                    job = collection.execute(
                        "jobs.run",
                        {
                            "code": (
                                "import tools.files\n"
                                "tools.files.write(path='job.txt', content='one\\ntwo\\n')\n"
                                "data = tools.files.read('job.txt')\n"
                                "result = {'lines': len(data.content.splitlines())}"
                            )
                        },
                    )
                    self.assertTrue(job.ok, job.result)
                    self.assertEqual(job.result["result"], {"lines": 2})
                    self.assertIn("Answer the user", job.result["next_action"])
                    self.assertEqual(job.result["usage"]["tool_calls"], 2)
                    self.assertEqual(job.result["usage"]["work_file_reads"], 1)
                    self.assertIn("Python jobs:", collection.prompt_instructions())
                    self.assertIn(
                        "Reliability and tool use:", collection.prompt_instructions()
                    )
                    self.assertIn(
                        "computation: python.eval", collection.prompt_instructions()
                    )
                    self.assertIn(
                        "tools.files.read(path: string)",
                        collection.prompt_instructions(),
                    )
                    self.assertIn(
                        ".content.splitlines()", collection.prompt_instructions()
                    )
                    delegated_prompt = collection.prompt_instructions().split(
                        "Tools available to delegated execution:", 1
                    )[1]
                    self.assertNotIn("tools.python.run(", delegated_prompt)
            finally:
                collection.close()
            self.assertEqual(connection.process.returncode, 0)

    def test_http_server_can_be_added_with_a_prefix(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            listener = socket.socket()
            listener.bind(("127.0.0.1", 0))
            port = listener.getsockname()[1]
            listener.close()
            process = subprocess.Popen(
                [
                    sys.executable,
                    str(PROJECT_DIR / "mcp_server.py"),
                    "--transport",
                    "http",
                    "--listen",
                    f"127.0.0.1:{port}",
                    "--work-dir",
                    tmp,
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            collection = McpToolCollection()
            try:
                wait_for_port(process, port)
                collection.add(
                    HttpMcpClient("remote", f"http://127.0.0.1:{port}/mcp"),
                    prefix=True,
                )
                result = collection.execute(
                    "remote.files.write",
                    {"path": "remote.txt", "content": "remote"},
                )
                self.assertTrue(result.ok)
                self.assertEqual(
                    (Path(tmp) / "remote.txt").read_text(encoding="utf-8"),
                    "remote",
                )
            finally:
                collection.close()
                if process.poll() is None:
                    process.terminate()
                process.wait(timeout=5)


def wait_for_port(process: subprocess.Popen, port: int) -> None:
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError("MCP HTTP server exited during startup")
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.1):
                return
        except OSError:
            time.sleep(0.02)
    raise RuntimeError("MCP HTTP server did not start")


if __name__ == "__main__":
    unittest.main()
