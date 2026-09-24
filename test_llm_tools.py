import json
import tempfile
import unittest
from pathlib import Path

from mcpbridge.server import McpServer
from tools import ServiceRegistry, ToolContext, ToolEnvironment, ToolRegistry, activate_tool_context
from tools.llm import LanguageModelResponse, LanguageModelService, LlmTools


class FakeLanguageModelService:
    def __init__(self) -> None:
        self.calls = []

    def ask(self, instruction, data, response_schema, max_output_tokens):
        self.calls.append((instruction, data, response_schema, max_output_tokens))
        return LanguageModelResponse(
            value={"label": "relevant"},
            usage={"llm_input_tokens": 23, "llm_output_tokens": 7},
            duration_ms=12,
            attempts=1,
            raw_response={"choices": [{"message": {"content": "{}"}}]},
        )


class DelegatedLlmToolTest(unittest.TestCase):
    def test_tool_is_delegated_only_and_records_usage_and_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            services = ServiceRegistry()
            service = FakeLanguageModelService()
            services.register(LanguageModelService, service)
            environment = ToolEnvironment(Path(tmp), services=services)
            registry = ToolRegistry()
            environment.bind_tools(registry)
            registry.register(LlmTools(environment))

            self.assertEqual(registry.definitions("direct"), ())
            self.assertEqual(
                [item.name for item in registry.definitions("delegated")], ["llm.ask"]
            )
            listed = McpServer(registry).handle(
                {"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}}
            )
            self.assertEqual(listed["result"]["tools"], [])

            job_dir = Path(tmp) / "job"
            job_dir.mkdir()
            with activate_tool_context(
                ToolContext("", (), {"job_dir": job_dir, "execution_surface": "delegated"})
            ):
                result = registry.execute(
                    "llm.ask",
                    {
                        "instruction": "Classify the text.",
                        "data": {"text": "hello"},
                        "response_schema": {
                            "type": "object",
                            "properties": {"label": {"type": "string"}},
                            "required": ["label"],
                            "additionalProperties": False,
                        },
                    },
                )

            self.assertTrue(result.ok)
            self.assertEqual(
                result.resource_usage,
                {"llm_input_tokens": 23, "llm_output_tokens": 7},
            )
            artifact = job_dir / result.result["artifact"]
            self.assertTrue(artifact.is_file())
            self.assertEqual(json.loads(artifact.read_text())["value"]["label"], "relevant")

    def test_tool_rejects_direct_use_even_if_called_by_name(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            services = ServiceRegistry()
            services.register(LanguageModelService, FakeLanguageModelService())
            environment = ToolEnvironment(Path(tmp), services=services)
            registry = ToolRegistry()
            environment.bind_tools(registry)
            registry.register(LlmTools(environment))
            response = McpServer(registry).handle(
                {
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "tools/call",
                    "params": {"name": "llm.ask", "arguments": {}},
                }
            )
            self.assertIn("not available for direct calls", response["error"]["message"])


if __name__ == "__main__":
    unittest.main()
