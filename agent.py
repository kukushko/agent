#!/usr/bin/env python3.12
"""Small terminal chat agent for an OpenAI-compatible chat endpoint."""

from __future__ import annotations

import argparse
import codecs
import json
import os
import re
import select
import shutil
import sys
import termios
import tty
import unicodedata
from enum import Enum
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, ContextManager, Protocol
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from debuglog import DebugLog, default_debug_log
from mcpbridge import HttpMcpClient, McpError, McpToolCollection, StdioMcpClient
from toolconfig import parse_mcp_servers, parse_tool_parameters


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_BASE_URL = "http://192.168.0.108:8000/v1"
DEFAULT_MODEL = "/root/models/Qwen3-30B-A3B-Thinking-2507-FP4"
DEFAULT_SYSTEM_PROMPT = SCRIPT_DIR / "SYSTEM_PROMPT.txt"
CONTEXT_WINDOW_TOKENS = 131_072
APPROX_CHARS_PER_TOKEN = 4
CONTEXT_SAFETY_TOKENS = 2_048
DEFAULT_MAX_TOKENS = 8_192
DEFAULT_HISTORY_MESSAGES = 24
DEFAULT_MAX_TOOL_CALLS = 4
DEFAULT_ENABLE_THINKING = True
LENGTH_RETRY_MAX_TOKENS = 16_384
HISTORY_CONTEXT_SHARE = 0.45
FILE_MACRO_RE = re.compile(r"{{\s*FILE\s*:\s*([^}]+?)\s*}}")
THINK_BLOCK_RE = re.compile(r"<think\b[^>]*>.*?</think\s*>", re.IGNORECASE | re.DOTALL)
UNCLOSED_THINK_BLOCK_RE = re.compile(r"<think\b[^>]*>.*\Z", re.IGNORECASE | re.DOTALL)
TOOL_CALL_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.IGNORECASE | re.DOTALL)
SURROGATE_RE = re.compile(r"[\udc80-\udcff]")
ANSI_RESET = "\x1b[0m"
ANSI_USER = "\x1b[97m"
ANSI_AGENT = "\x1b[90m"
ANSI_ACTIVITY = "\x1b[38;5;25m"
PENDING_STDIN_BYTES = bytearray()


@dataclass
class ChatMessage:
    role: str
    content: str


class ReadStatus(Enum):
    OK = "ok"
    EOF = "eof"
    INVALID = "invalid"


@dataclass
class ReadResult:
    status: ReadStatus
    text: str = ""
    error: str = ""


@dataclass
class ToolCall:
    name: str
    arguments: dict[str, Any]


@dataclass
class ToolCallParseResult:
    call: ToolCall | None = None
    error: str = ""
    attempted: bool = False


class ToolRuntime(Protocol):
    """Tool operations required by the model completion loop."""

    def prompt_instructions(self) -> str: ...

    def turn_guidance(self) -> str: ...

    def reliability_catalog(self) -> dict[str, list[str]]: ...

    def api_definitions(self) -> list[dict[str, Any]]: ...

    def execute(self, name: str, arguments: dict[str, Any]) -> Any: ...

    def request_context(
        self, user_input: str, history: list[Any]
    ) -> ContextManager[None]: ...


@dataclass
class TokenUsage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    def add(self, other: "TokenUsage") -> None:
        self.prompt_tokens += other.prompt_tokens
        self.completion_tokens += other.completion_tokens
        self.total_tokens += other.total_tokens


@dataclass
class CompletionResult:
    content: str
    usage: TokenUsage


@dataclass
class TurnResult:
    answer: str
    usage: TokenUsage


@dataclass(frozen=True)
class ToolEvent:
    """Structured progress event emitted during one agent turn."""

    name: str
    ok: bool | None
    summary: str
    arguments: dict[str, Any] | None = None
    category: str = "tool"
    phase: str = "finished"
    usage: TokenUsage | None = None


ToolEventHandler = Callable[[ToolEvent], None]


class CompletionLengthError(RuntimeError):
    """Raised when a completion exhausts its budget before producing a result."""


class PromptRenderer:
    def __init__(self, prompt_path: Path) -> None:
        self.prompt_path = prompt_path
        self.base_dir = prompt_path.resolve().parent

    def render(self, history: list[ChatMessage], user_input: str, max_chars: int) -> str:
        template = self.prompt_path.read_text(encoding="utf-8")
        template, file_values = self.collect_file_macros(template)
        template = template.replace("{{USER_INPUT}}", user_input)
        return self.fit_prompt(template, file_values, format_history(history), max_chars)

    def render_system(self, max_chars: int) -> str:
        template = self.prompt_path.read_text(encoding="utf-8")
        template, file_values = self.collect_file_macros(template)
        template = template.replace("{{HISTORY}}", "Use the prior chat messages.")
        template = template.replace("{{USER_INPUT}}", "Use the latest user message.")
        return self.fit_prompt(template, file_values, "", max_chars)

    def collect_file_macros(self, template: str) -> tuple[str, list[str]]:
        file_values: list[str] = []

        def replace(match: re.Match[str]) -> str:
            raw_path = match.group(1).strip()
            marker = f"{{{{__FILE_{len(file_values)}__}}}}"
            path = Path(raw_path)
            if not path.is_absolute():
                path = self.base_dir / path
            try:
                resolved = path.resolve()
                if not self.is_allowed_file(resolved):
                    file_values.append(f"[FILE access denied: {raw_path}]")
                    return marker
                file_values.append(resolved.read_text(encoding="utf-8"))
            except OSError as exc:
                file_values.append(f"[FILE read error: {raw_path}: {exc}]")
            return marker

        return FILE_MACRO_RE.sub(replace, template), file_values

    def fit_prompt(
        self,
        template: str,
        file_values: list[str],
        history_text: str,
        max_chars: int,
    ) -> str:
        empty_prompt = self.fill_prompt(template, [""] * len(file_values), "")
        elastic_budget = max(0, max_chars - len(empty_prompt))
        history_budget, file_budget = split_elastic_budget(
            elastic_budget,
            len(history_text),
            sum(len(value) for value in file_values),
        )
        fitted_history = truncate_head(history_text, history_budget, "history")
        fitted_files = fit_file_values(file_values, file_budget)
        prompt = self.fill_prompt(template, fitted_files, fitted_history)
        if len(prompt) > max_chars:
            prompt = truncate_tail(prompt, max_chars, "prompt")
        return prompt

    def fill_prompt(self, template: str, file_values: list[str], history_text: str) -> str:
        result = template.replace("{{HISTORY}}", history_text)
        for index, value in enumerate(file_values):
            result = result.replace(f"{{{{__FILE_{index}__}}}}", value)
        return result

    def is_allowed_file(self, path: Path) -> bool:
        try:
            path.relative_to(self.base_dir)
            return True
        except ValueError:
            return False


class OpenAIChatClient:
    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str,
        timeout: float,
        temperature: float,
        max_tokens: int,
        enable_thinking: bool,
        debug_log: DebugLog | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.timeout = timeout
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.enable_thinking = enable_thinking
        self.debug_log = debug_log or default_debug_log()

    def complete(
        self,
        system_prompt: str,
        chat_messages: list[ChatMessage],
        tool_definitions: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] = "auto",
    ) -> CompletionResult:
        messages: list[dict[str, str]] = [{"role": "system", "content": system_prompt}]
        messages.extend({"role": message.role, "content": message.content} for message in chat_messages)

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stream": False,
            "chat_template_kwargs": {"enable_thinking": self.enable_thinking},
        }
        if tool_definitions:
            payload["tools"] = tool_definitions
            payload["tool_choice"] = tool_choice
            payload["parallel_tool_calls"] = False
            if tool_choice == "required":
                payload["stop"] = ["</tool_call>"]

        total_usage = TokenUsage()
        for attempt in range(2):
            self.debug_log.record("llm.request", attempt=attempt + 1, payload=payload)
            try:
                data = self._send_payload(payload)
            except BaseException as exc:
                self.debug_log.record(
                    "llm.error", attempt=attempt + 1,
                    error_type=type(exc).__name__, error=str(exc),
                )
                raise
            self.debug_log.record("llm.response", attempt=attempt + 1, response=data)
            total_usage.add(parse_usage(data.get("usage")))
            try:
                choice = data["choices"][0]
                message = choice["message"]
            except (KeyError, IndexError, TypeError) as exc:
                raise RuntimeError(f"unexpected response shape: {data!r}") from exc
            try:
                content = normalize_assistant_message(
                    message, choice.get("finish_reason")
                )
                raw_content = message.get("content") if isinstance(message, dict) else None
                if (
                    choice.get("finish_reason") == "stop"
                    and isinstance(raw_content, str)
                    and content != raw_content.strip()
                ):
                    self.debug_log.record(
                        "llm.stop_terminated_tool_call_completed",
                        attempt=attempt + 1,
                        tool_call=content,
                    )
                if choice.get("finish_reason") == "length":
                    self.debug_log.record(
                        "llm.truncated_tool_call_salvaged",
                        attempt=attempt + 1,
                        tool_call=content,
                    )
                return CompletionResult(content=content, usage=total_usage)
            except CompletionLengthError:
                if attempt > 0:
                    self.debug_log.record("llm.retry_exhausted", reason="output_length")
                    raise RuntimeError(
                        "API exhausted the output budget twice without producing "
                        "text or a tool call"
                    )
                payload = build_length_retry_payload(payload)
                self.debug_log.record(
                    "llm.retry", reason="output_length",
                    max_tokens=payload["max_tokens"], enable_thinking=False,
                )

        raise RuntimeError("completion retry loop ended unexpectedly")

    def _send_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        request = Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urlopen(request, timeout=self.timeout) as response:
                data = json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"HTTP {exc.code}: {body}") from exc
        except URLError as exc:
            raise RuntimeError(f"connection failed: {exc}") from exc
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"invalid JSON response from API: {exc}") from exc
        if not isinstance(data, dict):
            raise RuntimeError("unexpected response shape: top-level value must be an object")
        return data


def normalize_assistant_message(message: Any, finish_reason: Any = None) -> str:
    """Normalize text and OpenAI-compatible structured tool-call messages."""
    if not isinstance(message, dict):
        raise RuntimeError("unexpected response shape: assistant message must be an object")

    tool_calls = message.get("tool_calls")
    if tool_calls:
        return normalize_structured_tool_calls(tool_calls)

    content = message.get("content")
    if finish_reason == "stop":
        completed = complete_stop_terminated_tool_call(content)
        if completed is not None:
            return completed
    if finish_reason == "length":
        salvaged = salvage_repeated_truncated_tool_call(content)
        if salvaged is not None:
            return salvaged
        raise CompletionLengthError(
            "API exhausted the output budget before completing text or a tool call"
        )
    if not isinstance(content, str):
        suffix = f" (finish_reason={finish_reason!r})" if finish_reason is not None else ""
        raise RuntimeError(
            "API returned neither text content nor structured tool calls" + suffix
        )
    content = content.strip()
    if not content:
        raise RuntimeError("API returned an empty assistant message")
    return content


def complete_stop_terminated_tool_call(content: Any) -> str | None:
    """Restore a tool-call closing tag omitted by an API stop sequence."""
    if not isinstance(content, str):
        return None
    stripped = content.strip()
    if not stripped.lower().startswith("<tool_call>"):
        return None
    if re.search(r"</tool_call>\s*$", stripped, re.IGNORECASE):
        return None
    candidate = f"{stripped}</tool_call>"
    if parse_tool_call(candidate).call is None:
        return None
    return candidate


def salvage_repeated_truncated_tool_call(content: Any) -> str | None:
    """Recover one valid call from a length-truncated run of identical calls."""
    if not isinstance(content, str):
        return None
    match = TOOL_CALL_RE.search(content)
    if match is None or content[: match.start()].strip():
        return None

    segment = match.group(0)
    parsed = parse_tool_call(segment)
    if parsed.call is None:
        return None

    position = match.end()
    while content.startswith(segment, position):
        position += len(segment)
    remainder = content[position:]
    if remainder.strip() and not segment.startswith(remainder):
        return None
    return segment


def build_length_retry_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Build a concise retry with a best-effort thinking-disable hint."""
    retry = dict(payload)
    retry["messages"] = [dict(message) for message in payload["messages"]]
    retry["messages"][0]["content"] += (
        "\n\nThe previous attempt exhausted its output budget during reasoning. "
        "Do not deliberate or repeat output. Respond exactly once and immediately "
        "with the required tool call or "
        "a concise final answer."
    )
    retry["chat_template_kwargs"] = {"enable_thinking": False}
    retry["max_tokens"] = max(int(payload["max_tokens"]), LENGTH_RETRY_MAX_TOKENS)
    return retry


def normalize_structured_tool_calls(raw_tool_calls: Any) -> str:
    """Convert one OpenAI-compatible function call to the internal text protocol."""
    if not isinstance(raw_tool_calls, list):
        raise RuntimeError("message.tool_calls must be a list")
    if len(raw_tool_calls) != 1:
        raise RuntimeError(
            "the agent supports exactly one structured tool call per completion, "
            f"got {len(raw_tool_calls)}"
        )

    raw_call = raw_tool_calls[0]
    if not isinstance(raw_call, dict) or not isinstance(raw_call.get("function"), dict):
        raise RuntimeError("structured tool call must contain a function object")
    function = raw_call["function"]
    name = function.get("name")
    if not isinstance(name, str) or not name:
        raise RuntimeError("structured tool call function name must be a string")

    arguments = function.get("arguments", {})
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"structured tool call arguments are invalid JSON: {exc}") from exc
    if not isinstance(arguments, dict):
        raise RuntimeError("structured tool call arguments must be a JSON object")

    payload = json.dumps(
        {"name": name, "arguments": arguments}, ensure_ascii=False, separators=(",", ":")
    )
    return f"<tool_call>{payload}</tool_call>"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a small terminal LLM agent.")
    try:
        enable_thinking_default = parse_bool_env("AGENT_ENABLE_THINKING", DEFAULT_ENABLE_THINKING)
        reliability_preflight_default = parse_bool_env(
            "AGENT_RELIABILITY_PREFLIGHT", True
        )
    except ValueError as exc:
        parser.error(str(exc))
    parser.add_argument("--base-url", default=os.environ.get("AGENT_BASE_URL", DEFAULT_BASE_URL))
    parser.add_argument("--model", default=os.environ.get("AGENT_MODEL", DEFAULT_MODEL))
    parser.add_argument("--api-key", default=os.environ.get("AGENT_API_KEY", "local"))
    parser.add_argument(
        "--system",
        dest="system_prompt",
        type=Path,
        default=Path(os.environ.get("AGENT_SYSTEM_PROMPT", DEFAULT_SYSTEM_PROMPT)),
        help="system prompt file path, default: SYSTEM_PROMPT.txt",
    )
    parser.add_argument(
        "--system-prompt",
        dest="system_prompt",
        type=Path,
        help="alias for --system",
    )
    parser.add_argument("--timeout", type=float, default=float(os.environ.get("AGENT_TIMEOUT", "120")))
    parser.add_argument("--temperature", type=float, default=float(os.environ.get("AGENT_TEMPERATURE", "0.2")))
    parser.add_argument("--max-tokens", type=int, default=int(os.environ.get("AGENT_MAX_TOKENS", str(DEFAULT_MAX_TOKENS))))
    parser.add_argument(
        "--disable-thinking",
        dest="enable_thinking",
        action="store_false",
        default=enable_thinking_default,
        help="send chat_template_kwargs.enable_thinking=false, default: enabled",
    )
    parser.add_argument(
        "--enable-thinking",
        dest="enable_thinking",
        action="store_true",
        help="send chat_template_kwargs.enable_thinking=true",
    )
    parser.add_argument(
        "--disable-reliability-preflight",
        dest="reliability_preflight",
        action="store_false",
        default=reliability_preflight_default,
        help="skip the short tool-role assessment before each user turn",
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=Path(os.environ.get("AGENT_WORK_DIR", "work")),
        help="directory exposed to file tools, default: ./work",
    )
    parser.add_argument(
        "--tool-param",
        action="append",
        default=[],
        metavar="PACKAGE:NAME=VALUE",
        help="override a package-owned tool setting; may be repeated",
    )
    parser.add_argument(
        "--disable-embedded-mcp",
        action="store_true",
        help="do not start the bundled MCP tool server",
    )
    parser.add_argument(
        "--add-mcp",
        action="append",
        default=[],
        metavar="NAME=URL",
        help="add an external Streamable HTTP MCP server; may be repeated",
    )
    parser.add_argument(
        "--max-tool-calls",
        type=int,
        default=int(os.environ.get("AGENT_MAX_TOOL_CALLS", str(DEFAULT_MAX_TOOL_CALLS))),
        help=f"maximum sequential tool calls per user turn, default: {DEFAULT_MAX_TOOL_CALLS}",
    )
    parser.add_argument(
        "--history-limit",
        type=int,
        default=int(os.environ.get("AGENT_HISTORY_LIMIT", str(DEFAULT_HISTORY_MESSAGES))),
        help=f"maximum number of prior messages to include, default: {DEFAULT_HISTORY_MESSAGES}",
    )
    args = parser.parse_args()
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    if args.max_tokens < 1:
        parser.error("--max-tokens must be at least 1")
    if args.max_tool_calls < 0:
        parser.error("--max-tool-calls must be non-negative")
    if args.history_limit < 0:
        parser.error("--history-limit must be non-negative")
    if not args.system_prompt.exists():
        parser.error(f"--system-prompt does not exist: {args.system_prompt}")
    try:
        args.tool_parameters = parse_tool_parameters(args.tool_param)
        args.mcp_servers = parse_mcp_servers(args.add_mcp)
    except ValueError as exc:
        parser.error(str(exc))
    return args


def prompt_char_budget(max_tokens: int) -> int:
    available_tokens = CONTEXT_WINDOW_TOKENS - max_tokens - CONTEXT_SAFETY_TOKENS
    return max(1_000, available_tokens * APPROX_CHARS_PER_TOKEN)


def format_history(history: list[ChatMessage]) -> str:
    if not history:
        return "(empty)"
    return "\n\n".join(f"{message.role.upper()}:\n{message.content}" for message in history)


def strip_think_blocks(value: str) -> str:
    """Remove Qwen-style hidden reasoning blocks from visible output/history."""
    without_closed_blocks = THINK_BLOCK_RE.sub("", value)
    without_dangling_block = UNCLOSED_THINK_BLOCK_RE.sub("", without_closed_blocks)
    return without_dangling_block.strip()


def parse_usage(raw_usage: Any) -> TokenUsage:
    if not isinstance(raw_usage, dict):
        return TokenUsage()
    prompt_tokens = int_value(raw_usage.get("prompt_tokens"))
    completion_tokens = int_value(raw_usage.get("completion_tokens"))
    total_tokens = int_value(raw_usage.get("total_tokens"))
    if total_tokens == 0 and (prompt_tokens or completion_tokens):
        total_tokens = prompt_tokens + completion_tokens
    return TokenUsage(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
    )


def int_value(value: Any) -> int:
    if isinstance(value, bool):
        return 0
    if isinstance(value, int):
        return value
    return 0


def parse_bool_env(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean value")


def append_tool_instructions(system_prompt: str, tools: ToolRuntime) -> str:
    return f"{system_prompt}\n\n{tools.prompt_instructions()}"


def parse_tool_call(answer: str) -> ToolCallParseResult:
    stripped_answer = answer.strip()
    match = TOOL_CALL_RE.fullmatch(stripped_answer)
    if match is not None:
        raw_json = match.group(1)
        attempted = True
    elif re.search(r"</?tool_call\b", stripped_answer, re.IGNORECASE):
        return ToolCallParseResult(
            attempted=True,
            error=(
                "a tool call must contain exactly one <tool_call> tag with one JSON "
                "object and no surrounding text or additional tool calls"
            ),
        )
    else:
        stripped = strip_markdown_json_fence(stripped_answer)
        if not stripped.startswith("{"):
            return ToolCallParseResult()
        raw_json = stripped
        attempted = True

    try:
        data = json.loads(raw_json)
    except json.JSONDecodeError as exc:
        return ToolCallParseResult(
            attempted=attempted,
            error=(
                f"invalid tool call JSON at line {exc.lineno}, column {exc.colno}: {exc.msg}. "
                "Retry with exactly one strict JSON object; do not add prose, another "
                "object, or Python expressions such as joins."
            ),
        )
    if not isinstance(data, dict):
        return ToolCallParseResult(attempted=attempted, error="tool call JSON must be an object")

    name = data.get("name", data.get("tool"))
    arguments = data.get("arguments", data.get("args", {}))
    if not isinstance(name, str):
        return ToolCallParseResult(attempted=attempted, error="tool call name must be a string")
    if not isinstance(arguments, dict):
        return ToolCallParseResult(attempted=attempted, error="tool call arguments must be an object")
    return ToolCallParseResult(call=ToolCall(name=name, arguments=arguments), attempted=attempted)


def extract_tool_call(answer: str) -> ToolCall | None:
    return parse_tool_call(answer).call


def strip_markdown_json_fence(value: str) -> str:
    stripped = value.strip()
    if not stripped.startswith("```"):
        return stripped
    lines = stripped.splitlines()
    if len(lines) >= 3 and lines[0].startswith("```") and lines[-1].strip() == "```":
        return "\n".join(lines[1:-1]).strip()
    return stripped


def format_tool_result(result: Any, arguments: dict[str, Any]) -> str:
    return json.dumps(
        {
            "tool": result.name,
            "arguments": arguments,
            "ok": result.ok,
            "result": result.result,
        },
        ensure_ascii=False,
        indent=2,
    )


def format_tool_parse_error(error: str) -> str:
    return json.dumps(
        {
            "tool": "tool_call_parser",
            "ok": False,
            "result": {
                "error": error,
                "action": "Retry the tool call using exactly the required format.",
                "required_format": (
                    '<tool_call>{"name":"package.tool",'
                    '"arguments":{"key":"value"}}</tool_call>'
                ),
            },
        },
        ensure_ascii=False,
        indent=2,
    )


def format_reliability_gate_result(missing_roles: list[str]) -> str:
    """Explain why a final answer needs another best-effort tool attempt."""
    return json.dumps(
        {
            "tool": "reliability_preflight",
            "ok": False,
            "result": {
                "missing_roles": missing_roles,
                "action": (
                    "Before answering, make one tool attempt for every missing "
                    "role. A failed attempt is sufficient; then use fallback knowledge."
                ),
            },
        },
        ensure_ascii=False,
        indent=2,
    )


def summarize_tool_result(result: Any) -> str:
    detail = result.result.get("path") or result.result.get("root") or result.result.get("error", "")
    if detail:
        return str(detail)
    return "completed"


def render_tool_turn_input(
    user_input: str, tool_results: list[str], guidance: str = ""
) -> str:
    if not tool_results:
        return f"{user_input}\n\n{guidance}" if guidance else user_input
    transcript = "\n\n".join(tool_results)
    rendered = (
        f"Original user request:\n{user_input}\n\n"
        f"Tool results so far:\n{transcript}\n\n"
        "Use these tool results to continue. Answer normally if no more tools are needed."
    )
    return f"{rendered}\n\n{guidance}" if guidance else rendered


def complete_with_tools(
    client: OpenAIChatClient,
    renderer: PromptRenderer,
    tools: ToolRuntime,
    history: list[ChatMessage],
    history_limit: int,
    max_prompt_chars: int,
    user_input: str,
    max_tool_calls: int,
    event_handler: ToolEventHandler | None = None,
    additional_guidance: str = "",
    required_roles: tuple[str, ...] = (),
    debug_log: DebugLog | None = None,
) -> TurnResult:
    trace = debug_log or default_debug_log()
    tool_results: list[str] = []
    turn_usage = TokenUsage()
    turn_guidance = "\n\n".join(
        value for value in (tools.turn_guidance(), additional_guidance) if value
    )
    role_catalog = tools.reliability_catalog()
    attempted_roles: set[str] = set()
    forced_role: str | None = None

    for _ in range(max_tool_calls):
        turn_input = render_tool_turn_input(user_input, tool_results, turn_guidance)
        context_history = trim_history(history, history_limit)
        system_prompt = append_tool_instructions(
            renderer.render_system(max_prompt_chars), tools
        )
        chat_messages = [*context_history, ChatMessage("user", turn_input)]
        definitions = tools.api_definitions()
        if forced_role is not None:
            allowed_names = set(role_catalog.get(forced_role, []))
            definitions = [
                definition
                for definition in definitions
                if definition.get("function", {}).get("name") in allowed_names
            ]
        activity_name = "required-tool" if forced_role is not None else (
            "decision" if not tool_results else "continuation"
        )
        completion = complete_with_activity(
            client,
            activity_name,
            event_handler,
            system_prompt,
            chat_messages,
            definitions,
            "required" if forced_role is not None else "auto",
        )
        turn_usage.add(completion.usage)
        answer = strip_think_blocks(completion.content)
        tool_parse = parse_tool_call(answer)
        if tool_parse.call is None and not tool_parse.attempted:
            missing_roles = sorted(set(required_roles) - attempted_roles)
            if not missing_roles:
                trace.record("agent.answer", answer=answer)
                return TurnResult(answer=answer, usage=turn_usage)
            tool_results.append(format_reliability_gate_result(missing_roles))
            forced_role = missing_roles[0]
            trace.record(
                "reliability.gate", missing_roles=missing_roles,
                forced_role=forced_role,
            )
            emit_tool_event(
                event_handler,
                ToolEvent(
                    "reliability_preflight",
                    False,
                    f"missing tool attempts for roles: {', '.join(missing_roles)}",
                ),
            )
            continue
        if tool_parse.call is None:
            trace.record("tool.parse_error", error=tool_parse.error, raw=answer)
            tool_results.append(format_tool_parse_error(tool_parse.error))
            emit_tool_event(
                event_handler,
                ToolEvent("parse_error", False, tool_parse.error),
            )
            continue

        tool_call = tool_parse.call
        attempted_roles.update(
            role
            for role, names in role_catalog.items()
            if tool_call.name in names
        )
        forced_role = None
        trace.record(
            "tool.call", name=tool_call.name, arguments=tool_call.arguments,
        )
        emit_tool_event(
            event_handler,
            ToolEvent(
                tool_call.name,
                None,
                "calling",
                tool_call.arguments,
                phase="started",
            ),
        )
        try:
            tool_result = tools.execute(tool_call.name, tool_call.arguments)
        except BaseException as exc:
            trace.record(
                "tool.error", name=tool_call.name,
                arguments=tool_call.arguments,
                error_type=type(exc).__name__, error=str(exc),
            )
            emit_tool_event(
                event_handler,
                ToolEvent(
                    tool_call.name,
                    False,
                    f"{type(exc).__name__}: {exc}",
                    tool_call.arguments,
                    usage=completion.usage,
                ),
            )
            raise
        trace.record(
            "tool.result", name=tool_call.name, ok=bool(tool_result.ok),
            result=tool_result.result,
        )
        formatted_result = format_tool_result(tool_result, tool_call.arguments)
        tool_results.append(formatted_result)
        emit_tool_event(
            event_handler,
            ToolEvent(
                tool_call.name,
                bool(tool_result.ok),
                summarize_tool_result(tool_result),
                tool_call.arguments,
                usage=completion.usage,
            ),
        )
        missing_roles = sorted(set(required_roles) - attempted_roles)
        if missing_roles:
            forced_role = missing_roles[0]
            trace.record(
                "reliability.advance",
                missing_roles=missing_roles,
                forced_role=forced_role,
            )

    turn_input = render_tool_turn_input(user_input, tool_results, turn_guidance)
    context_history = trim_history(history, history_limit)
    system_prompt = append_tool_instructions(
        renderer.render_system(max_prompt_chars), tools
    )
    chat_messages = [*context_history, ChatMessage("user", turn_input)]
    completion = complete_with_activity(
        client,
        "final",
        event_handler,
        system_prompt,
        chat_messages,
        tools.api_definitions(),
    )
    turn_usage.add(completion.usage)
    answer = strip_think_blocks(completion.content)
    tool_parse = parse_tool_call(answer)
    if tool_parse.call is None and not tool_parse.attempted:
        trace.record("agent.answer", answer=answer)
        return TurnResult(answer=answer, usage=turn_usage)
    if tool_parse.call is None:
        return TurnResult(
            answer=f"Tool call was invalid and could not be retried: {tool_parse.error}",
            usage=turn_usage,
        )

    return TurnResult(
        answer="Tool call limit reached. Please narrow the request or ask me to continue.",
        usage=turn_usage,
    )


def emit_tool_event(
    handler: ToolEventHandler | None, event: ToolEvent
) -> None:
    if handler is not None:
        handler(event)


def complete_with_activity(
    client: OpenAIChatClient,
    name: str,
    handler: ToolEventHandler | None,
    *args: Any,
) -> CompletionResult:
    """Run one logical LLM operation with paired progress events."""
    emit_tool_event(
        handler,
        ToolEvent(name, None, "calling LLM", category="llm", phase="started"),
    )
    try:
        completion = client.complete(*args)
    except BaseException as exc:
        emit_tool_event(
            handler,
            ToolEvent(
                name,
                False,
                f"{type(exc).__name__}: {exc}",
                category="llm",
            ),
        )
        raise
    emit_tool_event(
        handler,
        ToolEvent(
            name,
            True,
            "completed",
            category="llm",
            usage=completion.usage,
        ),
    )
    return completion


def print_tool_event(event: ToolEvent) -> None:
    """Render a structured tool event for the interactive terminal."""
    label = "preflight" if event.name == "preflight" else f"{event.category}:{event.name}"
    if event.phase == "started":
        detail = event.summary
        if event.arguments is not None:
            rendered = render_event_arguments(event.arguments)
            detail = f"{detail} {rendered}"
        line = f"[{label}] {detail}"
    else:
        status = "success" if event.ok else "failed"
        usage = ""
        if event.usage is not None:
            prefix = "selection tokens" if event.category == "tool" else "tokens"
            usage = f"; {prefix}: {format_compact_usage(event.usage)}"
        detail = f"; {event.summary}" if event.summary else ""
        line = f"[{label}] {status}{usage}{detail}"
    print(f"\n{colorize(line, ANSI_ACTIVITY)}", flush=True)


def render_event_arguments(arguments: dict[str, Any]) -> str:
    """Render bounded tool arguments for terminal progress output."""
    rendered = json.dumps(arguments, ensure_ascii=False, separators=(",", ":"))
    if len(rendered) > 2_000:
        rendered = rendered[:2_000] + "… [truncated; full value is in debug log]"
    return rendered


def format_compact_usage(usage: TokenUsage) -> str:
    """Render usage belonging to one logical LLM operation."""
    return (
        f"in {usage.prompt_tokens}, out {usage.completion_tokens}, "
        f"total {usage.total_tokens}"
    )


class AgentSession:
    """Stateful, stream-independent API for interactive and e2e agent turns."""

    def __init__(
        self,
        client: OpenAIChatClient,
        renderer: PromptRenderer,
        tools: ToolRuntime,
        history_limit: int,
        max_tool_calls: int,
        reliability_preflight: bool = True,
        debug_log: DebugLog | None = None,
    ) -> None:
        self.client = client
        self.renderer = renderer
        self.tools = tools
        self.history_limit = history_limit
        self.max_tool_calls = max_tool_calls
        self.reliability_preflight = reliability_preflight
        self.debug_log = debug_log or getattr(client, "debug_log", default_debug_log())
        self.max_prompt_chars = prompt_char_budget(client.max_tokens)
        self.history: list[ChatMessage] = []
        self.total_usage = TokenUsage()
        self.last_preflight_roles: tuple[str, ...] = ()

    def run(
        self,
        user_input: str,
        event_handler: ToolEventHandler | None = None,
    ) -> TurnResult:
        """Execute one user turn without reading or writing terminal streams."""
        self.debug_log.record(
            "turn.start", user_input=user_input,
            history=[{"role": item.role, "content": item.content} for item in self.history],
            reliability_preflight=self.reliability_preflight,
        )
        context_history = trim_history(self.history, self.history_limit)
        request_history = [*context_history, ChatMessage("user", user_input)]
        preflight_guidance = ""
        preflight_usage = TokenUsage()
        if self.reliability_preflight:
            (
                preflight_guidance,
                preflight_usage,
                self.last_preflight_roles,
            ) = assess_reliability_roles(
                self.client, self.tools, user_input, event_handler
            )
            self.debug_log.record(
                "reliability.preflight", selected_roles=self.last_preflight_roles,
                guidance=preflight_guidance,
            )
        else:
            self.last_preflight_roles = ()
        with self.tools.request_context(user_input, request_history):
            turn_result = complete_with_tools(
                client=self.client,
                renderer=self.renderer,
                tools=self.tools,
                history=self.history,
                history_limit=self.history_limit,
                max_prompt_chars=self.max_prompt_chars,
                user_input=user_input,
                max_tool_calls=self.max_tool_calls,
                event_handler=event_handler,
                additional_guidance=preflight_guidance,
                required_roles=self.last_preflight_roles,
                debug_log=self.debug_log,
            )
        turn_result.usage.add(preflight_usage)
        answer = strip_think_blocks(turn_result.answer)
        result = TurnResult(answer, turn_result.usage)
        self.total_usage.add(result.usage)
        self.history.extend(
            (
                ChatMessage("user", user_input),
                ChatMessage("assistant", answer),
            )
        )
        self.debug_log.record(
            "turn.complete", answer=answer,
            usage={
                "prompt_tokens": result.usage.prompt_tokens,
                "completion_tokens": result.usage.completion_tokens,
                "total_tokens": result.usage.total_tokens,
            },
        )
        return result


def assess_reliability_roles(
    client: OpenAIChatClient,
    tools: ToolRuntime,
    user_input: str,
    event_handler: ToolEventHandler | None = None,
) -> tuple[str, TokenUsage, tuple[str, ...]]:
    """Select useful tool roles without making their success mandatory."""
    catalog = tools.reliability_catalog()
    if not catalog:
        return "", TokenUsage(), ()
    system_prompt = (
        "You are a reliability planner, not an answerer. Select every available "
        "tool role whose use would materially improve factual reliability, exact "
        "calculation, or environment awareness for the request. Reference facts "
        "remembered by the model should select a suitable lookup or verification "
        "role. Empirical physical constants, material properties, current facts, "
        "technical specifications, and other externally established values are "
        "reference facts unless the user supplied them. Do not select lookup for "
        "deterministic mathematical functions, identities, formulas, degree/radian "
        "conversion, or quantities fully computable from user-provided inputs. "
        "Lookup is relevant only when the solution requires an external fact that "
        "cannot be derived or computed from the request. The word exact does not "
        "make a mathematical result a reference fact: exact and approximate values "
        "of the same function should both use computation. Selecting lookup for a "
        "self-contained mathematics problem is an error. "
        "Non-trivial numerical work should select a computation role. "
        "Return exactly one JSON object with keys roles (an array using only the "
        "available role names) and reason (a short string). Select no irrelevant "
        "roles. Tool failure will be handled by fallback later.\n\n"
        f"Available roles and tools: {json.dumps(catalog, ensure_ascii=False)}\n\n"
        f"Tool-owned selection guidance:\n{tools.turn_guidance()}"
    )
    previous_thinking = client.enable_thinking
    client.enable_thinking = False
    try:
        completion = complete_with_activity(
            client,
            "preflight",
            event_handler,
            system_prompt,
            [ChatMessage("user", user_input)],
            [],
        )
    except RuntimeError:
        return "", TokenUsage(), ()
    finally:
        client.enable_thinking = previous_thinking
    try:
        raw = strip_markdown_json_fence(strip_think_blocks(completion.content))
        parsed = json.loads(raw)
        raw_roles = parsed.get("roles", []) if isinstance(parsed, dict) else []
    except (AttributeError, json.JSONDecodeError):
        return "", completion.usage, ()
    selected = sorted(
        {
            role
            for role in raw_roles
            if isinstance(role, str) and role in catalog
        }
    )
    if not selected:
        return "", completion.usage, ()
    guidance = (
        "Automatic reliability preflight selected these relevant roles: "
        f"{', '.join(selected)}. Before the final answer, make one reasonable "
        "tool attempt for every selected role. If an attempt is unavailable or "
        "fails, continue using your best knowledge and state material assumptions."
    )
    return guidance, completion.usage, tuple(selected)


def trim_history(history: list[ChatMessage], limit: int) -> list[ChatMessage]:
    if limit == 0:
        return []
    return history[-limit:]


def split_elastic_budget(elastic_budget: int, history_chars: int, file_chars: int) -> tuple[int, int]:
    if elastic_budget <= 0:
        return 0, 0
    if history_chars == 0:
        return 0, elastic_budget
    if file_chars == 0:
        return elastic_budget, 0

    history_budget = min(history_chars, int(elastic_budget * HISTORY_CONTEXT_SHARE))
    file_budget = min(file_chars, elastic_budget - history_budget)
    leftover = elastic_budget - history_budget - file_budget
    if leftover > 0 and history_budget < history_chars:
        extra = min(leftover, history_chars - history_budget)
        history_budget += extra
        leftover -= extra
    if leftover > 0 and file_budget < file_chars:
        file_budget += min(leftover, file_chars - file_budget)
    return history_budget, file_budget


def fit_file_values(values: list[str], budget: int) -> list[str]:
    if not values:
        return []
    if budget <= 0:
        return [truncate_tail(value, 0, "file") for value in values]
    total_chars = sum(len(value) for value in values)
    if total_chars <= budget:
        return values

    fitted: list[str] = []
    remaining_budget = budget
    remaining_values = len(values)
    for value in values:
        per_file_budget = remaining_budget // remaining_values
        fitted_value = truncate_tail(value, per_file_budget, "file")
        fitted.append(fitted_value)
        remaining_budget -= len(fitted_value)
        remaining_values -= 1
    return fitted


def truncate_tail(value: str, max_chars: int, label: str) -> str:
    if len(value) <= max_chars:
        return value
    if max_chars <= 0:
        return f"[{label} omitted due to context budget]"
    suffix = f"\n[{label} truncated: {len(value) - max_chars} chars omitted]"
    if max_chars <= len(suffix):
        return suffix[-max_chars:]
    return value[: max_chars - len(suffix)].rstrip() + suffix


def truncate_head(value: str, max_chars: int, label: str) -> str:
    if len(value) <= max_chars:
        return value
    if max_chars <= 0:
        return f"[{label} omitted due to context budget]"
    prefix = f"[{label} truncated: {len(value) - max_chars} chars omitted]\n"
    if max_chars <= len(prefix):
        return prefix[:max_chars]
    return prefix + value[-(max_chars - len(prefix)) :].lstrip()


def print_help() -> None:
    print(
        "Commands:\n"
        "  /help       show this help\n"
        "  /exit       stop the agent\n"
        "  /quit       stop the agent\n"
        "  /reset      clear conversation history\n"
        "  /history    print conversation history\n"
        "  /system     print the rendered system prompt for the current turn\n"
        "  /disable_thinking  send enable_thinking=false on future requests\n"
        "  /enable_thinking   send enable_thinking=true on future requests\n"
    )


def color_enabled() -> bool:
    return "NO_COLOR" not in os.environ


def colorize(value: str, color: str) -> str:
    if not color_enabled():
        return value
    return f"{color}{value}{ANSI_RESET}"


def format_token_usage(turn_usage: TokenUsage, total_usage: TokenUsage) -> str:
    return (
        "tokens: "
        f"in {turn_usage.prompt_tokens}, "
        f"out {turn_usage.completion_tokens}, "
        f"total {turn_usage.total_tokens} "
        f"(vllm total in {total_usage.prompt_tokens}, out {total_usage.completion_tokens})"
    )


def handle_command(
    command: str,
    history: list[ChatMessage],
    renderer: PromptRenderer,
    max_prompt_chars: int,
    client: OpenAIChatClient,
    tools: ToolRuntime,
) -> bool:
    name = command.strip().split(maxsplit=1)[0]
    if name in ("/exit", "/quit"):
        return False
    if name == "/help":
        print_help()
        return True
    if name == "/reset":
        history.clear()
        print("history cleared")
        return True
    if name == "/history":
        print(format_history(history))
        return True
    if name == "/system":
        print(append_tool_instructions(renderer.render_system(max_prompt_chars), tools))
        return True
    if name == "/disable_thinking":
        client.enable_thinking = False
        print("thinking-disable hint enabled (provider support is model-dependent)")
        return True
    if name == "/enable_thinking":
        client.enable_thinking = True
        print("thinking-enable hint enabled (provider support is model-dependent)")
        return True
    print(f"unknown command: {name}")
    print("type /help for available commands")
    return True


def repl(
    client: OpenAIChatClient,
    renderer: PromptRenderer,
    tools: ToolRuntime,
    work_dir: Path,
    history_limit: int,
    max_tool_calls: int,
    reliability_preflight: bool,
) -> int:
    configure_terminal_input()
    session = AgentSession(
        client=client,
        renderer=renderer,
        tools=tools,
        history_limit=history_limit,
        max_tool_calls=max_tool_calls,
        reliability_preflight=reliability_preflight,
    )
    input_history: list[str] = []
    print("terminal agent ready. type /help for commands.")
    print(f"work file tools root: {work_dir.resolve()}")
    print(f"debug log: {session.debug_log.path}")
    while True:
        try:
            read_result = read_user_input("\n> ", input_history)
        except KeyboardInterrupt:
            print("\ninterrupted")
            return 130
        if read_result.status == ReadStatus.EOF:
            print()
            return 0
        if read_result.status == ReadStatus.INVALID:
            print(f"input decode error: {read_result.error}; line skipped", file=sys.stderr)
            continue

        user_input = read_result.text.strip()

        if not user_input:
            continue
        append_input_history(input_history, user_input)
        if user_input.startswith("/"):
            if not handle_command(
                user_input,
                session.history,
                renderer,
                session.max_prompt_chars,
                client,
                tools,
            ):
                return 0
            continue

        try:
            turn_result = session.run(user_input, print_tool_event)
        except RuntimeError as exc:
            session.debug_log.record(
                "turn.error", error_type=type(exc).__name__, error=str(exc)
            )
            print(f"error: {exc}", file=sys.stderr)
            continue

        answer = turn_result.answer
        print(f"\n{colorize(answer, ANSI_AGENT)}")
        print(
            colorize(
                format_token_usage(turn_result.usage, session.total_usage),
                ANSI_AGENT,
            )
        )


def main() -> int:
    args = parse_args()
    debug_log = default_debug_log()
    debug_log.record(
        "agent.start",
        argv=sys.argv,
        base_url=args.base_url,
        model=args.model,
        work_dir=str(args.work_dir.resolve()),
        embedded_mcp=not args.disable_embedded_mcp,
        external_mcp=[name for name, _ in args.mcp_servers],
    )
    renderer = PromptRenderer(args.system_prompt)
    try:
        tools = create_mcp_tools(args)
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        return 130
    except (McpError, OSError, RuntimeError, TypeError, ValueError) as exc:
        print(f"MCP startup failed: {exc}", file=sys.stderr)
        if not args.disable_embedded_mcp:
            print(
                "Use --disable-embedded-mcp to run without bundled tools.",
                file=sys.stderr,
            )
        return 2
    try:
        client = OpenAIChatClient(
            base_url=args.base_url,
            model=args.model,
            api_key=args.api_key,
            timeout=args.timeout,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            enable_thinking=args.enable_thinking,
            debug_log=debug_log,
        )
        return repl(
            client,
            renderer,
            tools,
            args.work_dir,
            args.history_limit,
            args.max_tool_calls,
            args.reliability_preflight,
        )
    finally:
        tools.close()


def create_mcp_tools(args: argparse.Namespace) -> McpToolCollection:
    """Connect embedded and configured external MCP tool servers."""
    collection = McpToolCollection()
    try:
        if not args.disable_embedded_mcp:
            command = [
                sys.executable,
                str(SCRIPT_DIR / "mcp_server.py"),
                "--transport",
                "stdio",
                "--work-dir",
                str(args.work_dir),
            ]
            for value in args.tool_param:
                command.extend(("--tool-param", value))
            for value in args.add_mcp:
                command.extend(("--add-mcp", value))
            collection.add(
                StdioMcpClient("embedded", command, trusted_context=True),
                prefix=False,
            )
        elif args.tool_param:
            raise RuntimeError(
                "--tool-param requires the embedded MCP server; configure external servers directly"
            )
        for name, url in args.mcp_servers:
            collection.add(HttpMcpClient(name, url, timeout=args.timeout), prefix=True)
    except BaseException:
        collection.close()
        raise
    print(
        f"MCP connected: {collection.server_count} server(s), "
        f"{collection.tool_count} tool(s)"
    )
    default_debug_log().record(
        "mcp.connected",
        server_count=collection.server_count,
        tool_count=collection.tool_count,
        tools=[
            definition.get("function", {}).get("name")
            for definition in collection.api_definitions()
        ],
    )
    return collection


def read_user_input(prompt: str, input_history: list[str]) -> ReadResult:
    if sys.stdin.isatty():
        return read_user_input_raw(prompt, input_history)

    try:
        text = input(prompt)
    except EOFError:
        return ReadResult(ReadStatus.EOF)
    except UnicodeDecodeError as exc:
        return ReadResult(ReadStatus.INVALID, error=str(exc))
    if SURROGATE_RE.search(text):
        return ReadResult(ReadStatus.INVALID, error="input contains undecodable bytes")
    return ReadResult(ReadStatus.OK, text=text)


def read_user_input_raw(prompt: str, input_history: list[str]) -> ReadResult:
    fd = sys.stdin.fileno()
    old_attrs = termios.tcgetattr(fd)
    buffer: list[str] = []
    history_index: int | None = None
    decoder = codecs.getincrementaldecoder("utf-8")("strict")

    def visible_prompt() -> str:
        return prompt.rsplit("\n", 1)[-1]

    previous_rows = terminal_rows_for_text(visible_prompt())

    def redraw() -> None:
        nonlocal previous_rows
        text = visible_prompt() + "".join(buffer)
        if previous_rows > 1:
            sys.stdout.write(f"\x1b[{previous_rows - 1}A")
        sys.stdout.write("\r\x1b[J")
        sys.stdout.write(colorize(text, ANSI_USER))
        sys.stdout.flush()
        previous_rows = terminal_rows_for_text(text)

    try:
        tty.setraw(fd)
        sys.stdout.write(colorize(prompt, ANSI_USER))
        sys.stdout.flush()
        while True:
            byte = read_raw_stdin_byte(fd)
            if byte == b"":
                sys.stdout.write("\r\n")
                sys.stdout.flush()
                return ReadResult(ReadStatus.EOF)

            if byte in (b"\r", b"\n"):
                drain_paired_newline(fd)
                tail = decoder.decode(b"", final=True)
                if tail:
                    buffer.append(tail)
                sys.stdout.write("\r\n")
                sys.stdout.flush()
                return ReadResult(ReadStatus.OK, "".join(buffer))

            if byte == b"\x03":
                raise KeyboardInterrupt

            if byte == b"\x04":
                if not buffer:
                    sys.stdout.write("\r\n")
                    sys.stdout.flush()
                    return ReadResult(ReadStatus.EOF)
                continue

            if byte in (b"\x7f", b"\x08"):
                decoder.reset()
                if buffer:
                    buffer.pop()
                    redraw()
                continue

            if byte == b"\x1b":
                sequence = read_escape_sequence()
                if sequence == b"\x1b":
                    buffer = []
                    history_index = None
                    decoder.reset()
                    redraw()
                    continue
                if sequence == b"\x1b[A":
                    if input_history:
                        if history_index is None:
                            history_index = len(input_history) - 1
                        else:
                            history_index = max(0, history_index - 1)
                        buffer = list(input_history[history_index])
                        decoder.reset()
                        redraw()
                    continue
                if sequence == b"\x1b[B":
                    if history_index is not None:
                        if history_index >= len(input_history) - 1:
                            history_index = None
                            buffer = []
                        else:
                            history_index += 1
                            buffer = list(input_history[history_index])
                        decoder.reset()
                        redraw()
                    continue
                continue

            try:
                char = decoder.decode(byte, final=False)
            except UnicodeDecodeError as exc:
                return ReadResult(ReadStatus.INVALID, error=str(exc))
            if char:
                buffer.append(char)
                redraw()
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_attrs)


def read_escape_sequence() -> bytes:
    fd = sys.stdin.fileno()
    sequence = b"\x1b"
    read_ready, _, _ = select.select([fd], [], [], 0.01)
    if not read_ready:
        return sequence
    sequence += read_raw_stdin_byte(fd)
    read_ready, _, _ = select.select([fd], [], [], 0.01)
    if read_ready:
        sequence += read_raw_stdin_byte(fd)
    return sequence


def read_raw_stdin_byte(fd: int) -> bytes:
    if PENDING_STDIN_BYTES:
        byte = bytes(PENDING_STDIN_BYTES[:1])
        del PENDING_STDIN_BYTES[:1]
        return byte
    return os.read(fd, 1)


def drain_paired_newline(fd: int) -> None:
    read_ready, _, _ = select.select([fd], [], [], 0.001)
    if not read_ready:
        return
    byte = os.read(fd, 1)
    if byte not in (b"\r", b"\n"):
        PENDING_STDIN_BYTES.extend(byte)


def append_input_history(input_history: list[str], value: str) -> None:
    if input_history and input_history[-1] == value:
        return
    input_history.append(value)


def configure_terminal_input() -> None:
    if not sys.stdin.isatty():
        return
    if not hasattr(termios, "IUTF8"):
        return
    try:
        attrs = termios.tcgetattr(sys.stdin.fileno())
        attrs[0] |= termios.IUTF8
        termios.tcsetattr(sys.stdin.fileno(), termios.TCSANOW, attrs)
    except termios.error:
        return


def terminal_rows_for_text(text: str) -> int:
    columns = max(1, shutil.get_terminal_size(fallback=(80, 24)).columns)
    rows = 1
    column = 0
    for char in text:
        if char == "\n":
            rows += 1
            column = 0
            continue
        width = display_width(char)
        if width == 0:
            continue
        if column > 0 and column + width > columns:
            rows += 1
            column = 0
        column += width
    return rows


def display_width(char: str) -> int:
    if char == "\t":
        return 4
    if unicodedata.combining(char):
        return 0
    category = unicodedata.category(char)
    if category.startswith("C"):
        return 0
    if unicodedata.east_asian_width(char) in ("F", "W"):
        return 2
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
