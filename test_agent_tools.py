import json
import os
import tempfile
import unittest
from contextlib import nullcontext, redirect_stdout
from io import StringIO
from pathlib import Path
from unittest.mock import patch

import agent as agent_module
from agent import (
    AgentSession,
    ChatMessage,
    CompletionResult,
    OpenAIChatClient,
    PromptRenderer,
    TokenUsage,
    ToolEvent,
    assess_reliability_roles,
    complete_with_tools,
    complete_stop_terminated_tool_call,
    extract_tool_call,
    format_tool_result,
    format_token_usage,
    handle_command,
    parse_tool_call,
    parse_tool_parameters,
    parse_usage,
    print_tool_event,
    salvage_repeated_truncated_tool_call,
)
from debuglog import DebugLog
from tools import (
    Delegation,
    ToolContext,
    ToolEnvironment,
    ToolPackage,
    ToolRegistry,
    activate_tool_context,
    discover_tool_packages,
    tool,
)


def create_tools(work_dir: Path) -> ToolRegistry:
    return discover_tool_packages(ToolEnvironment(work_dir=work_dir))


class ToolRegistryTest(unittest.TestCase):
    def test_builds_namespaced_schema_from_signature_and_docstring(self) -> None:
        class ExampleTools(ToolPackage):
            namespace = "example"

            @tool
            def search(self, query: str, limit: int = 10) -> dict[str, object]:
                """Search for matching values."""
                return {"query": query, "limit": limit}

        with tempfile.TemporaryDirectory() as tmp:
            registry = ToolRegistry()
            registry.register(
                ExampleTools(ToolEnvironment(work_dir=Path(tmp) / "work"))
            )

        definition = registry.definitions()[0]
        self.assertEqual(definition.name, "example.search")
        self.assertEqual(definition.description, "Search for matching values.")
        self.assertEqual(definition.parameters["required"], ["query"])
        self.assertEqual(
            definition.parameters["properties"]["limit"],
            {"type": "integer", "default": 10},
        )

    def test_tool_decorator_attaches_delegation_metadata(self) -> None:
        class ExampleTools(ToolPackage):
            namespace = "example"

            @tool(
                delegation=Delegation(
                    allowed=False, costs={}, reason="interactive only"
                ),
                prompt_instructions=("Use this capability carefully.",),
            )
            def inspect(self, value: str) -> dict[str, object]:
                """Inspect a value."""
                return {"value": value}

        registry = ToolRegistry()
        registry.register(ExampleTools(ToolEnvironment(work_dir=Path("unused"))))
        metadata = registry.definitions()[0].metadata

        self.assertFalse(metadata.delegation.allowed)
        self.assertEqual(metadata.delegation.reason, "interactive only")
        self.assertEqual(
            metadata.prompt_instructions, ("Use this capability carefully.",)
        )
        self.assertIn("Use this capability carefully.", registry.prompt_instructions())

    def test_rejects_tool_parameter_without_type_hint(self) -> None:
        class InvalidTools(ToolPackage):
            namespace = "invalid"

            @tool
            def broken(self, value) -> dict[str, object]:
                """Return a value."""
                return {"value": value}

        with tempfile.TemporaryDirectory() as tmp:
            registry = ToolRegistry()
            with self.assertRaisesRegex(TypeError, "must have a type hint"):
                registry.register(
                    InvalidTools(ToolEnvironment(work_dir=Path(tmp) / "work"))
                )

    def test_request_context_is_scoped_and_reset(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            environment = ToolEnvironment(work_dir=Path(tmp) / "work")
            with self.assertRaisesRegex(RuntimeError, "outside an active user turn"):
                _ = environment.context

            context = ToolContext(user_input="hello", history=())
            with activate_tool_context(context):
                self.assertIs(environment.context, context)

            with self.assertRaisesRegex(RuntimeError, "outside an active user turn"):
                _ = environment.context

    def test_tool_parameter_overrides_are_package_scoped_and_validated(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            environment = ToolEnvironment(
                work_dir=Path(tmp) / "work",
                tool_parameters={"web": {"url": "http://search.test/search"}},
            )
            registry = discover_tool_packages(environment)
            web_package = registry._tools["web.search"].method.__self__
            self.assertEqual(web_package.provider.url, "http://search.test/search")

            invalid_environment = ToolEnvironment(
                work_dir=Path(tmp) / "other",
                tool_parameters={"web": {"typo": "value"}},
            )
            with self.assertRaisesRegex(ValueError, "unknown parameter"):
                discover_tool_packages(invalid_environment)

    def test_parses_repeated_tool_parameter_overrides(self) -> None:
        self.assertEqual(
            parse_tool_parameters(
                ["web:url=http://search.test/search", "web:timeout=4.5"]
            ),
            {
                "web": {
                    "url": "http://search.test/search",
                    "timeout": "4.5",
                }
            },
        )
        with self.assertRaisesRegex(ValueError, "expected PACKAGE:NAME=VALUE"):
            parse_tool_parameters(["web.url=x"])


class WebToolsTest(unittest.TestCase):
    def test_fetch_uses_package_owned_defaults_and_validates_batch(self) -> None:
        from tools.web import DEFAULT_FETCH_PROXY_URL, WebTools

        class FakeStore:
            def fetch_batch(self, url_list, max_chars_per_page):
                self.call = (url_list, max_chars_per_page)
                return {"manifest": "offline/fetch-test/manifest.json"}

        with tempfile.TemporaryDirectory() as tmp:
            package = WebTools(ToolEnvironment(work_dir=Path(tmp) / "work"))
            store = FakeStore()
            package.offline_store = store
            registry = ToolRegistry()
            registry.register(package)

            result = registry.execute(
                "web.fetch_as_markdown", {"url_list": ["https://example.com/"]}
            )
            empty = registry.execute("web.fetch_as_markdown", {"url_list": []})

            self.assertEqual(package.parameter("fetch_proxy_url"), DEFAULT_FETCH_PROXY_URL)
            self.assertTrue(result.ok)
            self.assertEqual(store.call, (["https://example.com/"], 100_000))
            self.assertFalse(empty.ok)

    def test_search_normalizes_provider_results_and_warnings(self) -> None:
        from tools.web import WebTools
        from websearch import SearchResponse, SearchResult

        class FakeProvider:
            def search(self, query, max_results, language, time_range):
                self.call = (query, max_results, language, time_range)
                return SearchResponse(
                    (
                        SearchResult(
                            "Python", "https://python.org/", "Language", "example"
                        ),
                    ),
                    ("startpage: CAPTCHA",),
                )

        with tempfile.TemporaryDirectory() as tmp:
            package = WebTools(ToolEnvironment(work_dir=Path(tmp) / "work"))
            provider = FakeProvider()
            package.provider = provider
            registry = ToolRegistry()
            registry.register(package)

            result = registry.execute(
                "web.search",
                {"query": " python ", "language": "en", "time_range": "month"},
            )

            self.assertTrue(result.ok)
            self.assertEqual(provider.call, ("python", 5, "en", "month"))
            self.assertEqual(result.result["returned"], 1)
            self.assertEqual(result.result["results"][0]["title"], "Python")
            self.assertEqual(result.result["warnings"], ["startpage: CAPTCHA"])
            self.assertIn("fetch promising", result.result["next_action"])
            prompt = registry.prompt_instructions()
            self.assertIn("one generic query rarely establishes", prompt)
            self.assertIn("time_range accepts only day, month, year, or null", prompt)
            self.assertIn("untrusted in the security sense", prompt)

    def test_search_rejects_invalid_limits_and_time_ranges(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            registry = create_tools(Path(tmp) / "work")
            too_many = registry.execute(
                "web.search", {"query": "python", "max_results": 21}
            )
            invalid_range = registry.execute(
                "web.search", {"query": "python", "time_range": "week"}
            )

            self.assertFalse(too_many.ok)
            self.assertFalse(invalid_range.ok)


class WorkFileToolsTest(unittest.TestCase):
    def test_search_returns_paths_lines_columns_and_glob_filter(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            work_dir = Path(tmp) / "work"
            (work_dir / "nested").mkdir(parents=True)
            (work_dir / "first.py").write_text(
                "alpha beta alpha\nMiXeD\n", encoding="utf-8"
            )
            (work_dir / "nested" / "second.py").write_text(
                "alpha\n", encoding="utf-8"
            )
            (work_dir / "notes.txt").write_text("alpha\n", encoding="utf-8")
            tools = create_tools(work_dir)

            result = tools.execute(
                "files.search",
                {"query": "alpha", "file_pattern": "*.py"},
            )

            self.assertTrue(result.ok)
            self.assertEqual(result.result["returned"], 3)
            self.assertEqual(
                [match["path"] for match in result.result["matches"]],
                ["first.py", "first.py", "nested/second.py"],
            )
            self.assertEqual(result.result["matches"][0]["line"], 1)
            self.assertEqual(result.result["matches"][0]["column"], 1)
            self.assertEqual(result.result["matches"][1]["column"], 12)

    def test_search_supports_case_insensitive_regex(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            work_dir = Path(tmp) / "work"
            work_dir.mkdir()
            (work_dir / "sample.txt").write_text("Value: Alpha42\n", encoding="utf-8")
            tools = create_tools(work_dir)

            result = tools.execute(
                "files.search",
                {
                    "query": "alpha[0-9]+",
                    "regex": True,
                    "case_sensitive": False,
                },
            )

            self.assertTrue(result.ok)
            self.assertEqual(result.result["returned"], 1)
            self.assertEqual(result.result["matches"][0]["column"], 8)

    def test_search_falls_back_to_python_and_enforces_result_limit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            work_dir = Path(tmp) / "work"
            work_dir.mkdir()
            (work_dir / "sample.txt").write_text(
                "alpha alpha\nalpha\n", encoding="utf-8"
            )
            tools = create_tools(work_dir)

            with patch("tools.filesystem.shutil.which", return_value=None):
                result = tools.execute(
                    "files.search",
                    {"query": "ALPHA", "case_sensitive": False, "max_results": 2},
                )

            self.assertTrue(result.ok)
            self.assertEqual(result.result["engine"], "python")
            self.assertEqual(result.result["returned"], 2)
            self.assertTrue(result.result["truncated"])

    def test_regex_search_reports_missing_ripgrep_without_breaking_package(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tools = create_tools(Path(tmp) / "work")

            with patch("tools.filesystem.shutil.which", return_value=None):
                result = tools.execute(
                    "files.search", {"query": "a.*b", "regex": True}
                )

            self.assertFalse(result.ok)
            self.assertIn("requires ripgrep", result.result["error"])

    def test_write_read_and_list_stay_inside_work_dir(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tools = create_tools(Path(tmp) / "work")

            write_result = tools.execute(
                "files.write", {"path": "notes/today.txt", "content": "hello"}
            )
            self.assertTrue(write_result.ok)
            self.assertEqual(write_result.result["path"], "notes/today.txt")

            read_result = tools.execute("files.read", {"path": "notes/today.txt"})
            self.assertTrue(read_result.ok)
            self.assertEqual(read_result.result["content"], "hello")

            list_result = tools.execute("files.list", {"path": "."})
            self.assertTrue(list_result.ok)
            self.assertEqual(list_result.result["files"], ["notes/today.txt"])

    def test_rejects_absolute_and_parent_escape_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tools = create_tools(Path(tmp) / "work")

            absolute_result = tools.execute("files.read", {"path": "/etc/passwd"})
            self.assertFalse(absolute_result.ok)
            self.assertIn("absolute paths", absolute_result.result["error"])

            parent_result = tools.execute(
                "files.write", {"path": "../outside.txt", "content": "x"}
            )
            self.assertFalse(parent_result.ok)
            self.assertIn("escapes the work directory", parent_result.result["error"])
            self.assertFalse((Path(tmp) / "outside.txt").exists())

    def test_rejects_symlink_escape_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            outside = tmp_path / "outside"
            outside.mkdir()
            tools = create_tools(tmp_path / "work")
            (tmp_path / "work" / "link").symlink_to(outside, target_is_directory=True)

            result = tools.execute(
                "files.write", {"path": "link/secret.txt", "content": "x"}
            )
            self.assertFalse(result.ok)
            self.assertIn("escapes the work directory", result.result["error"])
            self.assertFalse((outside / "secret.txt").exists())

    def test_write_history_file_uses_runtime_history(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            work_dir = Path(tmp) / "work"
            tools = create_tools(work_dir)
            history = [
                ChatMessage("user", "hello"),
                ChatMessage("assistant", "hi"),
            ]

            with activate_tool_context(
                ToolContext(user_input="save history", history=tuple(history))
            ):
                result = tools.execute("files.write_history", {"path": "history.txt"})

            self.assertTrue(result.ok)
            self.assertEqual(result.result["path"], "history.txt")
            self.assertEqual(
                (work_dir / "history.txt").read_text(encoding="utf-8"),
                "USER:\nhello\n\nASSISTANT:\nhi",
            )


class ToolCallParsingTest(unittest.TestCase):
    def test_extracts_tagged_tool_call(self) -> None:
        call = extract_tool_call(
            '<tool_call>{"name":"files.read","arguments":{"path":"a.txt"}}</tool_call>'
        )
        self.assertIsNotNone(call)
        assert call is not None
        self.assertEqual(call.name, "files.read")
        self.assertEqual(call.arguments, {"path": "a.txt"})

    def test_extracts_fenced_json_tool_call(self) -> None:
        call = extract_tool_call(
            '```json\n{"name":"files.list","arguments":{"path":"."}}\n```'
        )
        self.assertIsNotNone(call)
        assert call is not None
        self.assertEqual(call.name, "files.list")

    def test_reports_invalid_tool_call_json(self) -> None:
        result = parse_tool_call(
            '<tool_call>{"name":"files.write","arguments":{"path":"history.txt","content":"\\n".join(["x"])}}</tool_call>'
        )

        self.assertTrue(result.attempted)
        self.assertIsNone(result.call)
        self.assertIn("invalid tool call JSON", result.error)

    def test_rejects_multiple_json_objects_in_one_tool_tag(self) -> None:
        result = parse_tool_call(
            '<tool_call>{"name":"files.list","arguments":{}}\n'
            '{"name":"files.list","arguments":{}}</tool_call>'
        )

        self.assertTrue(result.attempted)
        self.assertIsNone(result.call)
        self.assertIn("exactly one strict JSON object", result.error)

    def test_rejects_text_surrounding_a_tool_tag(self) -> None:
        result = parse_tool_call(
            'I will write the file.\n'
            '<tool_call>{"name":"files.write","arguments":{}}</tool_call>'
        )

        self.assertTrue(result.attempted)
        self.assertIsNone(result.call)
        self.assertIn("no surrounding text", result.error)


class ChatCompletionFlowTest(unittest.TestCase):
    def test_tool_result_keeps_argument_provenance(self) -> None:
        result = type(
            "Result", (),
            {"name": "python.eval", "ok": True, "result": {"value": "0.5"}},
        )()

        rendered = json.loads(
            format_tool_result(result, {"expression": "sin(radians(30))"})
        )

        self.assertEqual(
            rendered["arguments"], {"expression": "sin(radians(30))"}
        )

    def test_completes_a_tool_call_terminated_by_the_required_call_stop(self) -> None:
        incomplete = '<tool_call>{"name":"python.eval","arguments":{"expression":"6*7"}}'

        completed = complete_stop_terminated_tool_call(incomplete)

        parsed = parse_tool_call(completed or "")
        self.assertIsNotNone(parsed.call)
        assert parsed.call is not None
        self.assertEqual(parsed.call.arguments, {"expression": "6*7"})

    def test_required_tool_request_stops_after_one_textual_call(self) -> None:
        captured_payload = {}

        def fake_urlopen(request, timeout):
            nonlocal captured_payload
            captured_payload = json.loads(request.data.decode("utf-8"))
            return FakeResponse(
                {
                    "choices": [
                        {
                            "finish_reason": "stop",
                            "message": {
                                "content": '<tool_call>{"name":"python.eval","arguments":{"expression":"6*7"}}'
                            },
                        }
                    ]
                }
            )

        with patch.object(agent_module, "urlopen", fake_urlopen):
            client = OpenAIChatClient(
                "http://example.test/v1", "qwen", "local", 1, 0.2, 16, True
            )
            completion = client.complete(
                "system",
                [ChatMessage("user", "calculate")],
                [{"type": "function", "function": {"name": "python.eval"}}],
                "required",
            )

        self.assertEqual(captured_payload["stop"], ["</tool_call>"])
        self.assertFalse(captured_payload["parallel_tool_calls"])
        self.assertIsNotNone(parse_tool_call(completion.content).call)

    def test_salvages_identical_tool_calls_from_length_truncated_content(self) -> None:
        call = (
            '<tool_call>{"name":"python.eval","arguments":'
            '{"expression":"6 * 7"}}</tool_call>'
        )

        salvaged = salvage_repeated_truncated_tool_call(call * 3 + call[:31])

        self.assertEqual(salvaged, call)
        parsed = parse_tool_call(salvaged or "")
        self.assertIsNotNone(parsed.call)
        assert parsed.call is not None
        self.assertEqual(parsed.call.name, "python.eval")

    def test_does_not_salvage_different_or_trailing_tool_output(self) -> None:
        first = '<tool_call>{"name":"python.eval","arguments":{"expression":"1"}}</tool_call>'
        second = '<tool_call>{"name":"python.eval","arguments":{"expression":"2"}}</tool_call>'

        self.assertIsNone(salvage_repeated_truncated_tool_call(first + second))
        self.assertIsNone(salvage_repeated_truncated_tool_call(first + " explanation"))

    def test_client_accepts_repeated_tool_call_despite_length_finish(self) -> None:
        call = (
            '<tool_call>{"name":"python.eval","arguments":'
            '{"expression":"6 * 7"}}</tool_call>'
        )

        def fake_urlopen(request, timeout):
            return FakeResponse(
                {
                    "choices": [
                        {
                            "finish_reason": "length",
                            "message": {"content": call * 2 + call[:20]},
                        }
                    ],
                    "usage": {"completion_tokens": 100},
                }
            )

        with tempfile.TemporaryDirectory() as tmp:
            debug_log = DebugLog(Path(tmp))
            with patch.object(agent_module, "urlopen", fake_urlopen):
                client = OpenAIChatClient(
                    "http://example.test/v1", "qwen", "local", 1, 0.2, 16,
                    True, debug_log,
                )
                completion = client.complete("system", [ChatMessage("user", "calculate")])
            events = [
                json.loads(line)["event"]
                for line in debug_log.path.read_text(encoding="utf-8").splitlines()
            ]

        self.assertEqual(completion.content, call)
        self.assertIn("llm.truncated_tool_call_salvaged", events)
        self.assertNotIn("llm.retry", events)

    def test_client_records_full_request_and_response(self) -> None:
        def fake_urlopen(request, timeout):
            return FakeResponse({"choices": [{"message": {"content": "ok"}}]})

        with tempfile.TemporaryDirectory() as tmp:
            debug_log = DebugLog(Path(tmp))
            with patch.object(agent_module, "urlopen", fake_urlopen):
                client = OpenAIChatClient(
                    "http://example.test/v1", "qwen", "secret", 1, 0.2, 16,
                    False, debug_log,
                )
                client.complete("system text", [ChatMessage("user", "hello")])

            raw_log = debug_log.path.read_text(encoding="utf-8")
            entries = [json.loads(line) for line in raw_log.splitlines()]

        self.assertEqual([entry["event"] for entry in entries], ["llm.request", "llm.response"])
        self.assertEqual(entries[0]["payload"]["messages"][0]["content"], "system text")
        self.assertNotIn("secret", raw_log)

    def test_terminal_tool_event_includes_arguments(self) -> None:
        output = StringIO()
        with redirect_stdout(output), patch.object(agent_module, "color_enabled", return_value=True):
            print_tool_event(
                ToolEvent(
                    "web.search",
                    None,
                    "calling",
                    {"query": "lead density"},
                    phase="started",
                )
            )
            print_tool_event(
                ToolEvent(
                    "web.search",
                    True,
                    "completed",
                    {"query": "lead density"},
                    usage=TokenUsage(7, 3, 10),
                )
            )

        rendered = output.getvalue()
        self.assertEqual(agent_module.ANSI_ACTIVITY, "\x1b[38;5;25m")
        self.assertIn(agent_module.ANSI_ACTIVITY, rendered)
        self.assertIn('[tool:web.search] calling {"query":"lead density"}', rendered)
        self.assertIn(
            "[tool:web.search] success; selection tokens: in 7, out 3, total 10; completed",
            rendered,
        )

    def test_client_retries_truncated_nonempty_content(self) -> None:
        payloads = []

        def fake_urlopen(request, timeout):
            payloads.append(json.loads(request.data))
            if len(payloads) == 1:
                return FakeResponse(
                    {
                        "choices": [
                            {
                                "message": {"content": "partial answer"},
                                "finish_reason": "length",
                            }
                        ],
                        "usage": {"prompt_tokens": 4, "completion_tokens": 8},
                    }
                )
            return FakeResponse(
                {
                    "choices": [
                        {
                            "message": {"content": "complete answer"},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {"prompt_tokens": 5, "completion_tokens": 2},
                }
            )

        with patch.object(agent_module, "urlopen", fake_urlopen):
            client = OpenAIChatClient(
                "http://example.test/v1", "qwen", "local", 1, 0.2, 16, True
            )
            result = client.complete("system", [ChatMessage("user", "answer")])

        self.assertEqual(result.content, "complete answer")
        self.assertEqual(result.usage, TokenUsage(9, 10, 19))
        self.assertFalse(payloads[1]["chat_template_kwargs"]["enable_thinking"])

    def test_thinking_commands_toggle_client_state(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            prompt_path = Path(tmp) / "SYSTEM_PROMPT.txt"
            prompt_path.write_text("System.", encoding="utf-8")
            client = OpenAIChatClient(
                base_url="http://example.test/v1",
                model="qwen",
                api_key="local",
                timeout=1,
                temperature=0.2,
                max_tokens=16,
                enable_thinking=True,
            )
            renderer = PromptRenderer(prompt_path)
            tools = create_tools(Path(tmp) / "work")

            self.assertTrue(
                handle_command("/disable_thinking", [], renderer, 4096, client, tools)
            )
            self.assertFalse(client.enable_thinking)
            self.assertTrue(
                handle_command("/enable_thinking", [], renderer, 4096, client, tools)
            )
            self.assertTrue(client.enable_thinking)

    def test_client_sends_thinking_chat_template_kwarg(self) -> None:
        captured_payload = {}
        original_urlopen = agent_module.urlopen

        def fake_urlopen(request, timeout):
            nonlocal captured_payload
            captured_payload = json.loads(request.data.decode("utf-8"))
            return FakeResponse(
                {
                    "choices": [{"message": {"content": "ok"}}],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3},
                }
            )

        agent_module.urlopen = fake_urlopen
        try:
            client = OpenAIChatClient(
                base_url="http://example.test/v1",
                model="qwen",
                api_key="local",
                timeout=1,
                temperature=0.2,
                max_tokens=16,
                enable_thinking=False,
            )
            result = client.complete("system", [ChatMessage("user", "hello")])
        finally:
            agent_module.urlopen = original_urlopen

        self.assertEqual(result.content, "ok")
        self.assertEqual(result.usage, TokenUsage(1, 2, 3))
        self.assertEqual(captured_payload["chat_template_kwargs"], {"enable_thinking": False})

    def test_client_reports_null_content_as_runtime_error(self) -> None:
        original_urlopen = agent_module.urlopen

        def fake_urlopen(request, timeout):
            return FakeResponse({"choices": [{"message": {"content": None}}]})

        agent_module.urlopen = fake_urlopen
        try:
            client = OpenAIChatClient(
                base_url="http://example.test/v1",
                model="qwen",
                api_key="local",
                timeout=1,
                temperature=0.2,
                max_tokens=16,
                enable_thinking=True,
            )
            with self.assertRaisesRegex(RuntimeError, "neither text content nor structured"):
                client.complete("system", [ChatMessage("user", "hello")])
        finally:
            agent_module.urlopen = original_urlopen

    def test_client_normalizes_structured_tool_call_with_null_content(self) -> None:
        original_urlopen = agent_module.urlopen
        arguments = {
            "path": "result.py",
            "content": "from transformers import AutoModelForCausalLM\n",
        }

        def fake_urlopen(request, timeout):
            return FakeResponse(
                {
                    "choices": [
                        {
                            "finish_reason": "tool_calls",
                            "message": {
                                "content": None,
                                "tool_calls": [
                                    {
                                        "id": "call_1",
                                        "type": "function",
                                        "function": {
                                            "name": "files.write",
                                            "arguments": json.dumps(arguments),
                                        },
                                    }
                                ],
                            },
                        }
                    ]
                }
            )

        agent_module.urlopen = fake_urlopen
        try:
            client = OpenAIChatClient(
                base_url="http://example.test/v1",
                model="qwen",
                api_key="local",
                timeout=1,
                temperature=0.2,
                max_tokens=16,
                enable_thinking=True,
            )
            completion = client.complete(
                "system", [ChatMessage("user", "write result.py")]
            )
        finally:
            agent_module.urlopen = original_urlopen

        parsed = parse_tool_call(completion.content)
        self.assertIsNotNone(parsed.call)
        assert parsed.call is not None
        self.assertEqual(parsed.call.name, "files.write")
        self.assertEqual(parsed.call.arguments, arguments)

    def test_client_retries_length_completion_without_thinking(self) -> None:
        original_urlopen = agent_module.urlopen
        payloads = []
        responses = [
            {
                "choices": [
                    {
                        "finish_reason": "length",
                        "message": {"content": None, "reasoning_content": "thinking"},
                    }
                ],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 16,
                    "total_tokens": 26,
                },
            },
            {
                "choices": [
                    {
                        "finish_reason": "tool_calls",
                        "message": {
                            "content": None,
                            "tool_calls": [
                                {
                                    "type": "function",
                                    "function": {
                                        "name": "files.write",
                                        "arguments": '{"path":"result.py","content":"pass\\n"}',
                                    },
                                }
                            ],
                        },
                    }
                ],
                "usage": {
                    "prompt_tokens": 12,
                    "completion_tokens": 8,
                    "total_tokens": 20,
                },
            },
        ]

        def fake_urlopen(request, timeout):
            payloads.append(json.loads(request.data.decode("utf-8")))
            return FakeResponse(responses[len(payloads) - 1])

        agent_module.urlopen = fake_urlopen
        try:
            client = OpenAIChatClient(
                base_url="http://example.test/v1",
                model="qwen",
                api_key="local",
                timeout=1,
                temperature=0.2,
                max_tokens=16,
                enable_thinking=True,
            )
            with tempfile.TemporaryDirectory() as tmp:
                tools = create_tools(Path(tmp) / "work")
                completion = client.complete(
                    "system",
                    [ChatMessage("user", "write result.py")],
                    tools.api_definitions(),
                )
        finally:
            agent_module.urlopen = original_urlopen

        parsed = parse_tool_call(completion.content)
        self.assertIsNotNone(parsed.call)
        self.assertEqual(completion.usage, TokenUsage(22, 24, 46))
        self.assertEqual(payloads[0]["tool_choice"], "auto")
        self.assertEqual(payloads[0]["tools"][0]["type"], "function")
        self.assertEqual(
            payloads[0]["chat_template_kwargs"], {"enable_thinking": True}
        )
        self.assertEqual(
            payloads[1]["chat_template_kwargs"], {"enable_thinking": False}
        )
        self.assertEqual(payloads[1]["max_tokens"], 16_384)
        self.assertIn("Do not deliberate", payloads[1]["messages"][0]["content"])

    def test_sends_history_as_chat_messages_and_returns_usage(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            prompt_path = Path(tmp) / "SYSTEM_PROMPT.txt"
            prompt_path.write_text(
                "System.\nHistory: {{HISTORY}}\nInput: {{USER_INPUT}}",
                encoding="utf-8",
            )
            client = FakeClient("answer", TokenUsage(10, 3, 13))
            history = [
                ChatMessage("user", "first"),
                ChatMessage("assistant", "second"),
            ]

            result = complete_with_tools(
                client=client,
                renderer=PromptRenderer(prompt_path),
                tools=create_tools(Path(tmp) / "work"),
                history=history,
                history_limit=24,
                max_prompt_chars=4096,
                user_input="third",
                max_tool_calls=0,
            )

            self.assertEqual(result.answer, "answer")
            self.assertEqual(result.usage, TokenUsage(10, 3, 13))
            self.assertEqual(client.chat_messages[:2], history)
            self.assertTrue(client.chat_messages[2].content.startswith("third\n\n"))
            self.assertIn(
                "Automatic reliability guidance",
                client.chat_messages[2].content,
            )

    def test_agent_session_runs_without_terminal_streams(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            prompt_path = Path(tmp) / "SYSTEM_PROMPT.txt"
            prompt_path.write_text("System.", encoding="utf-8")
            client = SequenceClient(
                [
                    CompletionResult(
                        '<tool_call>{"name":"files.write","arguments":{"path":"value.txt","content":"42"}}</tool_call>',
                        TokenUsage(4, 3, 7),
                    ),
                    CompletionResult("Saved.", TokenUsage(5, 2, 7)),
                ]
            )
            events: list[ToolEvent] = []
            session = AgentSession(
                client=client,
                renderer=PromptRenderer(prompt_path),
                tools=create_tools(Path(tmp) / "work"),
                history_limit=4,
                max_tool_calls=1,
                reliability_preflight=False,
            )

            result = session.run("save 42", events.append)

            self.assertEqual(result.answer, "Saved.")
            tool_events = [event for event in events if event.category == "tool"]
            self.assertEqual([event.phase for event in tool_events], ["started", "finished"])
            self.assertEqual(tool_events[0].name, "files.write")
            self.assertTrue(tool_events[1].ok)
            self.assertEqual(tool_events[1].usage, TokenUsage(4, 3, 7))
            self.assertEqual(len(session.history), 2)
            self.assertEqual(session.total_usage, TokenUsage(9, 5, 14))

    def test_reliability_preflight_selects_only_available_roles(self) -> None:
        class Catalog:
            def reliability_catalog(self):
                return {"lookup": ["source.find"], "computation": ["calc.eval"]}

            def turn_guidance(self):
                return ""

        client = FakeClient(
            '{"roles":["lookup","computation","unknown"],"reason":"needed"}',
            TokenUsage(3, 2, 5),
        )
        events: list[ToolEvent] = []
        guidance, usage, roles = assess_reliability_roles(
            client, Catalog(), "calculate from a reference value", events.append
        )

        self.assertEqual(roles, ("computation", "lookup"))
        self.assertIn("computation, lookup", guidance)
        self.assertEqual(usage, TokenUsage(3, 2, 5))
        self.assertTrue(client.enable_thinking)
        self.assertEqual(
            [(event.name, event.phase, event.ok) for event in events],
            [("preflight", "started", None), ("preflight", "finished", True)],
        )
        self.assertEqual(events[-1].usage, TokenUsage(3, 2, 5))

    def test_preflight_distinguishes_computable_math_from_reference_facts(self) -> None:
        class Catalog:
            def reliability_catalog(self):
                return {
                    "reference_lookup": ["web.search"],
                    "computation": ["python.eval"],
                }

            def turn_guidance(self):
                return "Do not search for deterministic mathematics."

        client = FakeClient(
            '{"roles":["computation"],"reason":"deterministic math"}',
            TokenUsage(),
        )

        _, _, roles = assess_reliability_roles(
            client, Catalog(), "calculate sin(52 degrees)"
        )

        self.assertEqual(roles, ("computation",))
        self.assertIn("Do not select lookup for deterministic mathematical", client.system_prompt)
        self.assertIn("degree/radian conversion", client.system_prompt)
        self.assertIn("exact and approximate values", client.system_prompt)
        self.assertIn("Tool-owned selection guidance", client.system_prompt)

    def test_reliability_gate_requires_one_attempt_per_selected_role(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp) / "work"
            work.mkdir()
            (work / "value.txt").write_text("42", encoding="utf-8")
            prompt = Path(tmp) / "SYSTEM_PROMPT.txt"
            prompt.write_text("System.", encoding="utf-8")
            client = SequenceClient(
                [
                    CompletionResult("42", TokenUsage(1, 1, 2)),
                    CompletionResult(
                        '<tool_call>{"name":"files.read","arguments":{"path":"value.txt"}}</tool_call>',
                        TokenUsage(1, 1, 2),
                    ),
                    CompletionResult("The value is 42.", TokenUsage(1, 1, 2)),
                ]
            )
            events: list[ToolEvent] = []

            result = complete_with_tools(
                client=client,
                renderer=PromptRenderer(prompt),
                tools=create_tools(work),
                history=[],
                history_limit=4,
                max_prompt_chars=4096,
                user_input="read value",
                max_tool_calls=2,
                event_handler=events.append,
                required_roles=("workspace",),
            )

            self.assertEqual(result.answer, "The value is 42.")
            self.assertEqual(
                [
                    (event.name, event.phase)
                    for event in events
                    if event.category == "tool"
                ],
                [
                    ("reliability_preflight", "finished"),
                    ("files.read", "started"),
                    ("files.read", "finished"),
                ],
            )

    def test_remaining_role_is_forced_before_a_discarded_draft_answer(self) -> None:
        class RoleTools:
            def prompt_instructions(self):
                return "Tools."

            def turn_guidance(self):
                return ""

            def reliability_catalog(self):
                return {
                    "reference_lookup": ["web.search"],
                    "computation": ["python.eval"],
                }

            def api_definitions(self):
                return [
                    {"type": "function", "function": {"name": name}}
                    for name in ("web.search", "python.eval")
                ]

            def execute(self, name, arguments):
                return type(
                    "Result", (),
                    {"name": name, "ok": True, "result": {"value": "42"}},
                )()

            def request_context(self, user_input, history):
                return nullcontext()

        with tempfile.TemporaryDirectory() as tmp:
            prompt = Path(tmp) / "SYSTEM_PROMPT.txt"
            prompt.write_text("System.", encoding="utf-8")
            client = SequenceClient(
                [
                    CompletionResult(
                        '<tool_call>{"name":"web.search","arguments":{"query":"value"}}</tool_call>',
                        TokenUsage(),
                    ),
                    CompletionResult(
                        '<tool_call>{"name":"python.eval","arguments":{"expression":"6*7"}}</tool_call>',
                        TokenUsage(),
                    ),
                    CompletionResult("42", TokenUsage()),
                ]
            )

            result = complete_with_tools(
                client=client,
                renderer=PromptRenderer(prompt),
                tools=RoleTools(),
                history=[],
                history_limit=4,
                max_prompt_chars=4096,
                user_input="look up and calculate",
                max_tool_calls=2,
                required_roles=("computation", "reference_lookup"),
            )

        self.assertEqual(result.answer, "42")
        self.assertEqual(client.tool_choices, ["auto", "required", "auto"])
        self.assertEqual(client.tool_names[1], ["python.eval"])

    def test_failed_role_attempt_allows_fallback_answer(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            prompt = Path(tmp) / "SYSTEM_PROMPT.txt"
            prompt.write_text("System.", encoding="utf-8")
            client = SequenceClient(
                [
                    CompletionResult(
                        '<tool_call>{"name":"files.read","arguments":{"path":"missing.txt"}}</tool_call>',
                        TokenUsage(1, 1, 2),
                    ),
                    CompletionResult(
                        "Using fallback knowledge.", TokenUsage(1, 1, 2)
                    ),
                ]
            )

            result = complete_with_tools(
                client=client,
                renderer=PromptRenderer(prompt),
                tools=create_tools(Path(tmp) / "work"),
                history=[],
                history_limit=4,
                max_prompt_chars=4096,
                user_input="inspect missing state",
                max_tool_calls=1,
                required_roles=("workspace",),
            )

            self.assertEqual(result.answer, "Using fallback knowledge.")

    def test_retries_invalid_tool_call_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            prompt_path = Path(tmp) / "SYSTEM_PROMPT.txt"
            prompt_path.write_text("System.", encoding="utf-8")
            client = SequenceClient(
                [
                    CompletionResult(
                        '<tool_call>{"name":"files.write","arguments":{"path":"history.txt","content":"\\n".join(["x"])}}</tool_call>',
                        TokenUsage(10, 5, 15),
                    ),
                    CompletionResult(
                        '<tool_call>{"name":"files.write_history","arguments":{"path":"history.txt"}}</tool_call>',
                        TokenUsage(12, 4, 16),
                    ),
                    CompletionResult("Saved history to history.txt.", TokenUsage(8, 6, 14)),
                ]
            )
            history = [ChatMessage("user", "hello"), ChatMessage("assistant", "hi")]
            work_dir = Path(tmp) / "work"
            tools = create_tools(work_dir)

            context_history = [*history, ChatMessage("user", "save history")]
            with activate_tool_context(
                ToolContext(user_input="save history", history=tuple(context_history))
            ):
                result = complete_with_tools(
                    client=client,
                    renderer=PromptRenderer(prompt_path),
                    tools=tools,
                    history=history,
                    history_limit=24,
                    max_prompt_chars=4096,
                    user_input="save history",
                    max_tool_calls=2,
                )

            self.assertEqual(result.answer, "Saved history to history.txt.")
            self.assertEqual(result.usage, TokenUsage(30, 15, 45))
            self.assertEqual(
                (work_dir / "history.txt").read_text(encoding="utf-8"),
                "USER:\nhello\n\nASSISTANT:\nhi\n\nUSER:\nsave history",
            )

    def test_usage_helpers(self) -> None:
        usage = parse_usage({"prompt_tokens": 7, "completion_tokens": 5, "total_tokens": 12})
        self.assertEqual(usage, TokenUsage(7, 5, 12))
        self.assertEqual(
            format_token_usage(usage, TokenUsage(70, 50, 120)),
            "tokens: in 7, out 5, total 12 (vllm total in 70, out 50)",
        )


class RawInputHelpersTest(unittest.TestCase):
    def setUp(self) -> None:
        agent_module.PENDING_STDIN_BYTES.clear()

    def tearDown(self) -> None:
        agent_module.PENDING_STDIN_BYTES.clear()

    def test_drains_paired_newline(self) -> None:
        read_fd, write_fd = os.pipe()
        try:
            os.write(write_fd, b"\n")

            agent_module.drain_paired_newline(read_fd)

            self.assertEqual(agent_module.PENDING_STDIN_BYTES, bytearray())
        finally:
            os.close(read_fd)
            os.close(write_fd)

    def test_preserves_non_newline_after_enter(self) -> None:
        read_fd, write_fd = os.pipe()
        try:
            os.write(write_fd, b"A")

            agent_module.drain_paired_newline(read_fd)

            self.assertEqual(agent_module.read_raw_stdin_byte(read_fd), b"A")
        finally:
            os.close(read_fd)
            os.close(write_fd)


class FakeClient:
    def __init__(self, content: str, usage: TokenUsage) -> None:
        self.content = content
        self.usage = usage
        self.chat_messages: list[ChatMessage] = []
        self.system_prompt = ""
        self.max_tokens = 16
        self.enable_thinking = True

    def complete(
        self,
        system_prompt: str,
        chat_messages: list[ChatMessage],
        tool_definitions=None,
        tool_choice="auto",
    ) -> CompletionResult:
        self.system_prompt = system_prompt
        self.chat_messages = chat_messages
        return CompletionResult(self.content, self.usage)


class SequenceClient:
    def __init__(self, results: list[CompletionResult]) -> None:
        self.results = results
        self.calls = 0
        self.max_tokens = 16
        self.enable_thinking = True
        self.tool_choices = []
        self.tool_names = []

    def complete(
        self,
        system_prompt: str,
        chat_messages: list[ChatMessage],
        tool_definitions=None,
        tool_choice="auto",
    ) -> CompletionResult:
        self.tool_choices.append(tool_choice)
        self.tool_names.append(
            [item.get("function", {}).get("name") for item in (tool_definitions or [])]
        )
        result = self.results[self.calls]
        self.calls += 1
        return result


class FakeResponse:
    def __init__(self, data) -> None:
        self.data = data

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        return None

    def read(self) -> bytes:
        return json.dumps(self.data).encode("utf-8")


if __name__ == "__main__":
    unittest.main()
