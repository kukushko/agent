import json
import tempfile
import unittest
from pathlib import Path

from llmservice.openai import OpenAIJsonService
from mcpbridge.server import McpServer
from tools import ServiceRegistry, ToolContext, ToolEnvironment, ToolRegistry, activate_tool_context
from tools.llm import (
    LanguageModelFailure,
    LanguageModelResponse,
    LanguageModelService,
    LlmTools,
)


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
            raw_responses=({"choices": [{"message": {"content": "{}"}}]},),
        )


class FailingLanguageModelService:
    def ask(self, instruction, data, response_schema, max_output_tokens):
        raise LanguageModelFailure(
            "output budget exhausted",
            usage={"llm_input_tokens": 19, "llm_output_tokens": 4096},
            duration_ms=123,
            attempts=1,
            raw_responses=(
                {
                    "choices": [
                        {
                            "finish_reason": "length",
                            "message": {"content": "", "reasoning_content": "thinking"},
                        }
                    ]
                },
            ),
        )


class ScriptedOpenAIJsonService(OpenAIJsonService):
    def __init__(self, responses):
        super().__init__(
            base_url="http://unused",
            model="unused",
            api_key="unused",
            timeout=1,
        )
        self.responses = iter(responses)

    def _complete(self, messages, response_schema, max_output_tokens):
        return next(self.responses)


class DelegatedLlmToolTest(unittest.TestCase):
    def test_service_repairs_length_exhaustion_and_accumulates_usage(self) -> None:
        service = ScriptedOpenAIJsonService(
            [
                {
                    "choices": [
                        {"finish_reason": "length", "message": {"content": ""}}
                    ],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 20},
                },
                {
                    "choices": [
                        {
                            "finish_reason": "stop",
                            "message": {"content": '{"answer":"unknown"}'},
                        }
                    ],
                    "usage": {"prompt_tokens": 12, "completion_tokens": 4},
                },
            ]
        )
        response = service.ask(
            "Answer from supplied data.",
            {},
            {
                "type": "object",
                "properties": {"answer": {"type": "string"}},
                "required": ["answer"],
            },
            128,
        )
        self.assertEqual(response.value, {"answer": "unknown"})
        self.assertEqual(response.attempts, 2)
        self.assertEqual(
            response.usage, {"llm_input_tokens": 22, "llm_output_tokens": 24}
        )
        self.assertEqual(len(response.raw_responses), 2)

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

    def test_failed_call_preserves_usage_and_raw_response_in_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            services = ServiceRegistry()
            services.register(LanguageModelService, FailingLanguageModelService())
            environment = ToolEnvironment(Path(tmp), services=services)
            registry = ToolRegistry()
            environment.bind_tools(registry)
            registry.register(LlmTools(environment))
            job_dir = Path(tmp) / "job"
            job_dir.mkdir()
            with activate_tool_context(ToolContext("", (), {"job_dir": job_dir})):
                result = registry.execute(
                    "llm.ask",
                    {
                        "instruction": "Answer.",
                        "data": {},
                        "response_schema": {"type": "object"},
                    },
                )
            self.assertFalse(result.ok)
            self.assertEqual(result.resource_usage["llm_output_tokens"], 4096)
            artifact = json.loads(
                (job_dir / result.result["artifact"]).read_text(encoding="utf-8")
            )
            self.assertFalse(artifact["ok"])
            self.assertEqual(artifact["usage"]["llm_output_tokens"], 4096)
            self.assertEqual(
                artifact["raw_responses"][0]["choices"][0]["finish_reason"],
                "length",
            )

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
