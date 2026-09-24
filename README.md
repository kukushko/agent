# Terminal Agent

Minimal terminal chat loop for an OpenAI-compatible local LLM endpoint.

Docker is optional and is required only for the isolated Python tool package.
The agent can still run file and web tools when Docker is unavailable.

## Requirements

- Python 3.12 for the terminal agent and dependency installation
- an OpenAI-compatible Chat Completions endpoint
- Docker with daemon access for `python.*` and `jobs.*` tools
- ripgrep for accelerated and regular-expression `files.search` calls
- a local BGE-M3 ONNX export under `models/bge-m3` for optional semantic search

Install the pinned Python dependencies with the Python 3.12 interpreter:

```bash
python3.12 -m pip install -r requirements.txt
```

The sandbox image also provides Python 3.12 independently of the host runtime.
If ripgrep is missing, fixed-string file searches use the built-in Python
fallback.

## Run

```bash
python3.12 agent.py
python3.12 agent.py --system custom_prompt.txt
```

Defaults:

- API base URL: `http://192.168.0.108:8000/v1`
- model: `/models/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf`
- system prompt: `SYSTEM_PROMPT.txt`
- context window: approximately 131k tokens
- history limit: 24 prior messages
- Qwen thinking mode: enabled
- output token limit: 16384
- work file tools root: `./work`
- max sequential tool calls per turn: 4
- bundled MCP server: enabled as a managed stdio child process

Agent-wide settings can be overridden with command-line arguments or their
`AGENT_*` environment variables:

| Setting | Argument | Environment variable |
| --- | --- | --- |
| API endpoint | `--base-url` | `AGENT_BASE_URL` |
| model | `--model` | `AGENT_MODEL` |
| API key | `--api-key` | `AGENT_API_KEY` |
| system prompt | `--system` | `AGENT_SYSTEM_PROMPT` |
| request timeout | `--timeout` | `AGENT_TIMEOUT` |
| temperature | `--temperature` | `AGENT_TEMPERATURE` |
| output token limit | `--max-tokens` | `AGENT_MAX_TOKENS` |
| thinking default | `--enable-thinking` / `--disable-thinking` | `AGENT_ENABLE_THINKING` |
| reliability preflight | `--disable-reliability-preflight` | `AGENT_RELIABILITY_PREFLIGHT` |
| work directory | `--work-dir` | `AGENT_WORK_DIR` |
| tool-call limit | `--max-tool-calls` | `AGENT_MAX_TOOL_CALLS` |
| history limit | `--history-limit` | `AGENT_HISTORY_LIMIT` |

The embedded MCP server is enabled by default. `--disable-embedded-mcp` runs the
agent without bundled tools, and repeatable `--add-mcp NAME=URL` arguments add
external Streamable HTTP MCP servers.

Run `python3.12 agent.py --help` for the complete command-line reference.

Included system prompts:

- `SYSTEM_PROMPT.txt` - general AI assistant chat
- `CHAT_HUMAN_PROMPT.txt` - free-form human-like conversation
- `CODING_PROMPT.txt` - programming and debugging

## Commands

- `/help` - show available commands
- `/exit` or `/quit` - stop the agent
- `/reset` - clear conversation history
- `/history` - print conversation history
- `/system` - print the rendered system prompt for the current turn
- `/disable_thinking` - send `chat_template_kwargs.enable_thinking=false` on future requests
- `/enable_thinking` - send `chat_template_kwargs.enable_thinking=true` on future requests

After each assistant response, the agent prints a gray usage line based on the
OpenAI-compatible `usage` object returned by vLLM:

```text
tokens: in 123, out 45, total 168 (vllm total in 1000, out 250)
```

If the server does not return `usage`, the counters for that request remain
zero.

For chat templates that support the option, thinking can be requested or
discouraged per request:

```bash
python3.12 agent.py --disable-thinking
AGENT_ENABLE_THINKING=false python3.12 agent.py
```

The client sends this as `chat_template_kwargs.enable_thinking`. This is a hint,
not a capability guarantee: some models ignore it and continue producing
reasoning. By default it is sent as `true`. During an interactive session, use
`/disable_thinking` and `/enable_thinking` to switch the hint without restarting
the agent.

## MCP Architecture

The agent is an MCP host and client. It does not discover or execute bundled
tool packages in its own process. By default it starts the adjacent
`mcp_server.py` as a managed child process and connects over MCP stdio:

```text
agent.py
  └── MCP stdio
      └── mcp_server.py
          └── ToolRegistry and bundled tool packages
```

The child inherits the terminal process group, exits normally when its MCP stdin
reaches EOF, and is explicitly terminated by the agent if it does not stop in
time. Consequently Ctrl+C and normal agent shutdown do not leave the embedded
server running.

The model may use either OpenAI-compatible structured calls or the retained
text-call form:

```text
<tool_call>{"name":"files.read","arguments":{"path":"notes.txt"}}</tool_call>
```

The agent routes both forms through the same MCP connection, sends the result
back to the model, and asks it to continue. Tool calls are limited by
`--max-tool-calls`. Each result sent back to the model includes the arguments
that produced it, preserving provenance when the same tool is called more than
once.

Available tools:

- `files.list` - list files under the work directory
- `files.read` - read a UTF-8 file under the work directory
- `files.search` - search work files recursively with line and column metadata
- `files.semantic_search` - retrieve conceptually relevant excerpts with a lazy local index when BGE-M3 is installed
- `files.write` - write UTF-8 text under the work directory
- `files.write_history` - write the current conversation history under the work directory
- `python.check_syntax` - check a work file with Python 3.12
- `python.run` - run a Python 3.12 work file in an isolated container
- `python.eval` - evaluate one Python 3.12 expression in an isolated container
- `web.search` - search the public web through a configured provider
- `web.fetch_as_markdown` - download public pages through a fixed proxy into `work/offline`
- `jobs.run` - run persistent sandboxed Python that orchestrates other tools

The default work directory is `./work` from the current working directory. Use
`--work-dir some/path` or `AGENT_WORK_DIR=some/path` to change it.

All tool paths must be relative. The runtime resolves every requested path
against the configured work directory and rejects absolute paths,
parent-directory escapes, and symlink escapes outside that directory.

`files.search` performs fixed-string search by default and supports optional
regular expressions, case-insensitive matching, glob filtering, and bounded
result counts. Results include relative file paths, one-based line and column
numbers, and bounded line previews. It uses `rg --json` when ripgrep is
available. If `rg` is missing, fixed-string searches transparently use a Python
fallback; regex searches return a localized error without disabling the rest of
the `files` package.

### Semantic File Search

`files.semantic_search` performs dense conceptual retrieval over supported
UTF-8 files (`.md`, `.txt`, `.text`, `.rst`, `.py`, `.json`, `.yaml`, `.yml`,
and `.csv`). Unlike `files.read`, each match already contains a bounded excerpt,
its source path, chunk number, one-based line range, and cosine score. Use the
ordinary `files.search` tool for exact strings, identifiers, and regexes.

The semantic tool is optional. MCP publishes it only when `models/bge-m3`
contains the required local tokenizer and ONNX model and the pinned NumPy,
ONNX Runtime, and tokenizers dependencies are installed. Missing or incomplete
model data produces a startup warning for the optional semantic package while
all other `files.*` tools remain available.

Startup performs only cheap file and dependency probes. It does not import
ONNX Runtime or load model weights. The first real semantic query loads the
CPU model; the same model instance is then reused until the MCP process exits.
Concurrent calls are serialized so they cannot load duplicate model copies.

Indexes are built lazily and incrementally outside `work/`. Each source file is
mirrored using this transparent rule:

```text
work/offline/batch/article.md
tmp/indexes/offline/batch/article.md.index/
  metadata.json
  chunks.jsonl
  embeddings.npy
```

`metadata.json` records the exact source hash, index format, chunker settings,
and model fingerprint. An absent, incomplete, changed, or incompatible index is
rebuilt; an unchanged one is reused. Indexes for deleted files are removed when
their containing search scope is synchronized. Metadata is written last so an
interrupted build is never mistaken for a committed current index.

The package-owned defaults are:

| Tool parameter | Default |
| --- | --- |
| `files:semantic_model_path` | `models/bge-m3` |
| `files:semantic_index_root` | `tmp/indexes` |
| `files:semantic_chunk_chars` | `1800` |
| `files:semantic_overlap_chars` | `200` |
| `files:semantic_batch_size` | `8` |
| `files:semantic_max_file_bytes` | `8388608` |
| `files:semantic_max_files` | `500` |
| `files:semantic_max_scope_bytes` | `67108864` |

Relative model and index paths resolve beside the MCP distribution. As with
other package settings, a standalone MCP server consumes overrides itself and
an embedded agent only forwards them generically:

```bash
python3.12 agent.py \
  --tool-param files:semantic_model_path=models/bge-m3 \
  --tool-param files:semantic_index_root=tmp/indexes
```

Tool calls must be strict JSON. If the model emits a malformed tool call, the
runtime sends a parse error and the exact required format back to the model so
it can retry. A call must contain one JSON object inside one `<tool_call>` tag,
without surrounding prose or additional calls. Invalid or empty API response
content is reported as a recoverable turn error instead of terminating the
interactive session. OpenAI-compatible responses that move a recognized call
to `message.tool_calls` and set `message.content` to `null` are normalized back
into the same internal tool-call protocol before execution.

The prompt budget assumes a 131072-token context window. If a completion reaches
`finish_reason: length` before producing text or a tool call, the client retries
that API request once with at least 32768 output tokens, a concise-response
instruction, and `enable_thinking=false` as a best-effort provider hint. Models
are not assumed to support disabling thinking. Usage from both attempts is
included in the turn total.

Some models repeat the same textual tool call until the output budget is
exhausted. When a length-truncated response consists only of complete identical
tool calls followed by an optional truncated copy, the client safely accepts
the first call. Truncated prose, differing calls, and malformed calls are still
retried or rejected. Required-tool requests additionally use the textual
`</tool_call>` terminator as a stop sequence. If the provider omits that stop
marker from the returned content, the client restores it only after validating
the resulting single call.

### Tool Packages

Each tool module explicitly exports one or more `ToolPackage` subclasses through
`TOOL_PACKAGES`. A package defines a lowercase namespace, receives a shared
`ToolEnvironment`, and exposes only methods marked with the minimal `@tool`
decorator. Tool methods must have a docstring, complete parameter type hints,
and a typed dictionary return value. Default parameter values distinguish
optional parameters from required ones.

```python
from tools import Delegation, ToolPackage, tool


class ExampleTools(ToolPackage):
    namespace = "example"

    @tool(
        delegation=Delegation(
            costs={"tool_calls": 1, "example_queries": 1},
            quota_defaults={"tool_calls": 200, "example_queries": 25},
        ),
        epistemic_roles=("lookup", "verification"),
        reliability_guidance=(
            "Verify material remembered reference values when this tool is available.",
        ),
    )
    def search(self, query: str, limit: int = 10) -> dict[str, object]:
        """Search for matching values."""
        return {"query": query, "limit": limit}


TOOL_PACKAGES = (ExampleTools,)
```

The runtime registers this method as `example.search`, validates its signature,
and generates its prompt schema from the type hints, defaults, and docstring.
The decorator also owns portable metadata such as whether delegated execution
may call the tool, its named resource costs, optional prompt instructions, and
epistemic roles and role-specific reliability guidance. Roles describe generic ways a tool can improve reliability,
such as `lookup`, `verification`, `computation`, `workspace`, or
`orchestration`. The MCP client uses the active catalog to generate a soft
policy: remembered reference facts should preferably be verified, non-trivial
calculations should preferably use computation tools, and failed optional
verification falls back to the model's best knowledge instead of blocking an
answer. Lookup applies to externally established facts, not deterministic
mathematics: formulas, degree/radian conversion, and exact or approximate
function values computable from user inputs belong to computation tools.
Infrastructure treats resource names as opaque values and does not identify
tools by their namespace or method name.

Before each normal user turn, a short reliability preflight selects
the applicable roles from the active catalog. The selected role names are added
to the current turn as salient guidance. After the model chooses its first
useful tool, the completion loop requests any remaining selected role before
allowing a draft answer and restricts that generation to matching tools with
`tool_choice=required`. The same gate remains as a fallback if the model tries
to answer before making any required attempt.
A failed call still satisfies the attempt, so tool failure remains non-blocking
and falls back to the model's best knowledge. Preflight usage is included in the
turn token totals.
Set `AGENT_RELIABILITY_PREFLIGHT=false` or pass
`--disable-reliability-preflight` to remove this extra model request.
Modules whose names start with `_`, plus the infrastructure modules `base` and
`runtime`, are not discovered as tool modules.

`ToolEnvironment.context` exposes the current request-scoped `ToolContext` only
while a user turn is being processed. It is propagated through sequential and
nested calls with `contextvars` and raises an error outside an active turn.
Packages can invoke another tool through `ToolEnvironment.tools`; these calls go
through the same registry, argument validation, and result normalization.
Long-lived optional capabilities are obtained from the type-keyed
`ToolEnvironment.services` registry, keeping package-specific services out of
the common environment contract.

Package-specific settings live with their package rather than in `agent.py`.
For convenience, repeatable `--tool-param PACKAGE:NAME=VALUE` arguments passed
to the agent are forwarded to its embedded MCP child. Package and parameter
names are validated by that server during tool discovery. These overrides are
rejected when `--disable-embedded-mcp` is used because external MCP servers own
their configuration.

### External MCP Servers

Add independently running Streamable HTTP servers with unique local names:

```bash
python3.12 agent.py \
  --add-mcp docs=http://192.168.0.108:9100/mcp \
  --add-mcp services=https://tools.example.com/mcp
```

External tools receive the configured server name as a prefix. For example, a
`search` tool advertised by `docs` is exposed to the model as `docs.search`.
Bundled tools retain their short names such as `files.read`. Duplicate exposed
names fail during startup rather than silently replacing a tool.

When the embedded server is enabled, each `--add-mcp` specification is also
forwarded to that child. The agent and embedded server create independent
connections to the same external endpoint. Agent calls use the direct
connection; jobs use the embedded server's private dependency connection.
External dependencies are not re-exported by the embedded server, so the model
still sees each external tool exactly once.

The client prefers the stateless MCP `2026-07-28` protocol, including
`server/discover`, per-request metadata, and HTTP routing headers. It falls back
to the initialize/initialized lifecycle for 2025-era servers. Paginated
`tools/list`, `tools/call`, JSON responses, and Streamable HTTP SSE responses are
supported. The current HTTP client does not yet implement OAuth discovery or
interactive authorization, so authenticated remote servers must currently be
placed behind an already authorized endpoint.

The embedded connection receives a private `_meta` value containing only the
current bounded conversation history needed by `files.write_history`. This
metadata is never sent to external MCP servers.

### Standalone Bundled Server

The same bundled server can run independently over HTTP:

```bash
python3.12 mcp_server.py \
  --transport http \
  --listen 127.0.0.1:9000 \
  --work-dir work \
  --jobs-dir tmp/jobs \
  --add-mcp docs=http://127.0.0.1:9100/mcp \
  --tool-param web:url=http://192.168.0.108:9080/search
```

Connect the agent to that instance instead of starting a child:

```bash
python3.12 agent.py \
  --disable-embedded-mcp \
  --add-mcp local=http://127.0.0.1:9000/mcp
```

The HTTP server has no built-in authentication and should not be exposed to an
untrusted network. Its default listen address is `127.0.0.1:9000`.

## Python Jobs

`jobs.run` executes a concise Python 3.12 program in a fresh Docker container.
The container has no network, sees only its own persistent job directory, and
cannot directly access the agent work directory. Tool calls cross the sandbox
through an atomic filesystem request/response broker handled by the MCP server.

Job code uses a generated synchronous proxy:

```python
matches = tools.files.search("TODO", file_pattern="*.py").matches
result = sorted({match.path for match in matches})
```

A tool with exactly one required parameter accepts it positionally:

```python
text = tools.files.read("notes.txt").content
result = len(text.splitlines())
```

Otherwise use keyword arguments:

```python
tools.files.write(path="answer.txt", content="42\n")
result = {"written": "answer.txt"}
```

Tool response objects support both attribute and item access. Failed calls raise
`ToolError`, which job code may catch. The program must assign its final
JSON-compatible value to the global `result` variable. Built-in `open()` reads
and writes private job artifacts; files in `work/` must be accessed through
`tools.files.*`. A missing `open()` target reports this boundary explicitly
rather than implying that the corresponding work file does not exist.

The `tools` proxy is injected automatically and is also importable, so both
direct `tools.files.read(...)` usage and generated `import tools.files` code are
accepted. If otherwise invalid generated code contains exactly one additional
JSON-escaping layer, `jobs.run` removes that layer only when the repaired source
successfully compiles; valid Python strings are left unchanged.

Each job currently has these package-owned default limits, enforced by the
host-side broker:

| Resource | Limit |
| --- | ---: |
| wall time | 600 seconds |
| total brokered calls | 200 |
| file reads | 100 |
| web searches | 10 |
| result size | 1 MiB |
| stdout | 1 MiB |
| stderr | 1 MiB |
| memory | 512 MiB |
| CPU | 1 core |
| processes | 64 |

The package-owned defaults can be overridden for the embedded server:

```bash
python3.12 agent.py \
  --tool-param jobs:wall_time=300 \
  --tool-param jobs:quota_tool_calls=80 \
  --tool-param jobs:quota_work_file_reads=40 \
  --tool-param jobs:quota_web_searches=5
```

Quota categories are arbitrary names declared with costs and recommended limits
in each tool's `@tool` metadata. The job broker merges those declarations and
performs only generic accounting. A `jobs:quota_<resource>` override is valid
for any resource present in the active catalog. Bundled MCP metadata is
preserved across the transport; compatible external tools without delegation
metadata default to one `tool_calls` unit with a 200-call limit.

Tools whose decorators disallow delegation are not exposed inside jobs. The
bundled declarations currently exclude recursive jobs, history export, and
nested Python execution.

Runs are retained separately from `work/` under `tmp/jobs/<job-id>/`:

```text
job.py                 exact model-generated program
manifest.json          identity and creation metadata
policy.json            applied limits
tools.json             tool schemas visible to the job
state.json             status, duration, and quota usage
result.json            final JSON-compatible value
error.json             execution error, when present
stdout.txt             bounded program output
stderr.txt             bounded diagnostics
calls.jsonl            broker call journal
artifacts/             private intermediate files
requests/              pending, processing, and completed requests
responses/             broker responses
```

Every job receives its own directory and Docker mount. Requests and responses
are published with atomic renames. The host validates IPC directories and opens
container-controlled JSON files without following symlinks.

## Web Access

The `web.search` tool returns normalized titles, URLs, snippets, source names,
and publication dates from a provider-neutral search interface. Search results
are external, untrusted content. The initial provider implementation uses a
fixed SearXNG JSON endpoint, but the tool schema and result format do not expose
or depend on SearXNG-specific fields.

The package uses these local defaults:

| Tool parameter | Default | Meaning |
| --- | --- | --- |
| `web:provider` | `searxng` | search provider implementation |
| `web:url` | `http://192.168.0.108:9080/search` | fixed provider endpoint |
| `web:timeout` | `15` | request timeout in seconds |
| `web:fetch_proxy_url` | `http://192.168.0.108:3128` | mandatory HTTP proxy for page downloads |
| `web:fetch_timeout` | `20` | timeout for one page in seconds |
| `web:fetch_batch_timeout` | `60` | total batch timeout in seconds |
| `web:fetch_max_response_bytes` | `2097152` | maximum compressed/network bytes per page |
| `web:fetch_max_decoded_bytes` | `8388608` | maximum decoded bytes per page |
| `web:fetch_max_total_bytes` | `20971520` | maximum decoded bytes per batch |
| `web:fetch_max_redirects` | `5` | maximum redirects per page |
| `web:offline_directory` | `offline` | output directory relative to `work/` |

`web.search` returns five results by default and accepts at most 20. Its optional
`language` argument is passed to the provider, and `time_range` accepts `day`,
`month`, or `year`. Override deployment-specific settings without changing
source code:

```bash
python3.12 agent.py \
  --tool-param web:url=http://search-host:9080/search \
  --tool-param web:timeout=10
```

The response body is limited to 2 MiB. Provider HTTP, connection, JSON, and
response-shape failures are returned as ordinary failed tool calls rather than
terminating the agent. The SearXNG client contacts its configured endpoint
directly instead of inheriting process proxy variables, which prevents a proxy
from accidentally intercepting private-network search traffic.

The tool-owned prompt guidance treats search as source discovery rather than
answer extraction. It tells the model to use focused queries, use multiple
angles for broad/current synthesis, and not confuse `max_results` with the
number of final answer items. A successful result also carries a `next_action`
reminder: vague snippets and landing-page labels must not be promoted to facts;
the model should refine the query or fetch promising pages. "Untrusted" refers
to prompt-injection safety, not a prohibition on extracting factual evidence.

`web.fetch_as_markdown` accepts up to ten public HTTP(S) URLs in one call. It
downloads them only through `web:fetch_proxy_url`; it never falls back to a
direct connection and does not inherit process proxy or `NO_PROXY` settings.
Destination URLs and every redirect are checked before use. URL credentials,
non-HTTP schemes, local/private literal addresses, local host suffixes, and
ports other than 80 and 443 are rejected. Proxy URLs containing credentials
are also rejected so secrets cannot leak through command lines or debug logs.

Each call creates `work/offline/fetch-.../`. Extracted Markdown, one JSON
sidecar per requested URL, and an aggregate `manifest.json` are written there.
Safe filenames combine the request index, an ASCII URL slug, and a stable URL
hash. The tool returns those relative paths instead of putting complete pages
into model context, so the model can inspect them with `files.read` and
`files.search`. Individual failures are recorded without discarding successful
pages from the same batch.

HTML extraction uses Trafilatura with markdownify as its fallback; plain text
is stored directly. Network bodies, decoded bodies, extracted characters, total
batch size, redirects, and elapsed batch time are bounded. Downloaded documents
carry an explicit untrusted-content notice because page text is data, not agent
instructions.

These defaults belong to the `web` tool package and therefore to the MCP
server. A standalone server can override them directly; the agent merely
forwards generic package parameters when it manages the embedded server:

```bash
python3.12 mcp_server.py --tool-param web:fetch_proxy_url=http://proxy:3128
python3.12 agent.py --tool-param web:fetch_proxy_url=http://proxy:3128
```

## Python Sandbox

The `python` tool package runs every operation in a fresh Docker container. The
model uses the same relative paths as the `files` package; the host work
directory is mounted read-write at an internal location that is not exposed in
tool arguments. Runtime networking is disabled for all current Python tools.

The sandbox container uses a read-only root filesystem, drops all Linux
capabilities, enables `no-new-privileges`, runs as the host user, and applies
CPU, memory, process, time, and output-size limits. The Docker socket and host
directories outside the configured work directory are not mounted.

`python.run` accepts optional `stdin_path`, `stdout_path`, and `stderr_path`
arguments. Redirected output remains in the work directory and the tool result
contains only its path, byte count, and truncation status. Without an output
path, bounded UTF-8 output is returned directly in the tool result.

`python.eval` preloads the standard `math` module and its public functions, so
both `math.sin(math.radians(50))` and `sin(radians(50))` work directly. It
accepts either one expression or a short statement snippet; a snippet must
assign its returned value to `result`. Related comparison values should be
returned as a dictionary with descriptive keys. A
redirected result of at most 4096 bytes is also returned inline, so a scalar
calculation is not hidden from the model merely because it supplied an output
path; larger redirected output remains only in the work file. A
nonzero Python exit code, timeout, or output-limit termination marks the outer
tool result as failed while keeping
the structured stdout and stderr diagnostics available to the model.

The predefined image is `terminal-agent-python312-sandbox:local`, built from
`sandbox/python312/Dockerfile`. Its Python 3.12 base image is pinned by digest.
The image is checked lazily on the first Python tool call. A SHA-256 fingerprint
of every file in the build context is stored as an image label; a missing image
or changed fingerprint triggers an automatic rebuild.

`ToolEnvironment.sandbox.is_available` probes the Docker CLI, daemon access, and
current-user permissions without building the image. If Docker is unavailable,
the entire `python` package is omitted from both prompt and native API schemas,
and the agent logs a warning while keeping other tools available.

In an interactive terminal, the agent uses its own small UTF-8 line reader.
Backspace deletes whole Unicode characters, and Up/Down browse previously typed
input lines. Esc clears the current input line. User input is rendered in
bright white, and agent responses are rendered in gray when stdout is a TTY. The
line reader redraws the whole wrapped input block on edits, so backspace works
correctly for input longer than the terminal width.

## Prompt Macros

The system prompt supports these placeholders:

- `{{HISTORY}}` - formatted conversation history
- `{{USER_INPUT}}` - current user message
- `{{FILE:path/to/file.txt}}` - UTF-8 text from another file

Relative `FILE` paths are resolved from the directory containing
`SYSTEM_PROMPT.txt`.

Conversation history is sent to the Chat Completions API as prior `user` and
`assistant` messages. The `{{HISTORY}}` and `{{USER_INPUT}}` prompt macros are
kept for compatibility, but in normal API calls they are replaced with short
references to the chat messages instead of embedding the full conversation into
the system prompt.

File macro content is fitted into the context window approximately. The script
uses a simple character-based estimate and truncates oversized file inserts
from the end.

Qwen-style `<think>...</think>` reasoning blocks are removed before responses
are printed and before assistant messages are saved back into conversation
history.

## Project Layout

```text
agent.py                 terminal client, completion loop, and MCP host
debuglog.py              always-on structured agent trace writer
mcp_server.py            standalone bundled MCP tool server
mcpbridge/               stdio/HTTP MCP transports and protocol adapters
jobruntime/              persistent job controller and filesystem broker
toolconfig.py            shared package-parameter parser
toolpolicy.py            generic rendering of tool-owned reliability guidance
tools/                   server-side tool packages and registry
semanticsearch/          lazy ONNX embeddings and mirrored incremental indexes
websearch/               provider-neutral search types and provider adapters
webfetch/                proxy-only page transport, extraction, and offline storage
sandbox/                 sandbox interface and Docker implementation
sandbox/python312/       reproducible Python 3.12 sandbox image context
workpaths.py             shared work-directory path validation
test_agent_tools.py      agent, registry, file, and web tool tests
test_webfetch.py         web transport, conversion, and offline storage tests
test_jobs.py             job policy, quota, and IPC safety tests
test_semantic_search.py  semantic chunking, indexing, availability, and reuse tests
test_mcp.py              MCP protocol and transport integration tests
test_sandbox_tools.py    sandbox and Python tool tests
TODO.md                  deferred architectural work and experiments
```

`work/` contains agent-created artifacts and is intentionally excluded from
Git. `log/` contains runtime diagnostics and is also intentionally excluded.
Source code, documentation, identifiers, comments, and commit messages are
maintained in English; project-owner communication is in Russian.

## Debug Tracing

The agent always creates one JSON Lines trace file per process under `log/` and
prints its path at startup. The trace includes complete LLM payloads and
responses, reliability-preflight decisions, tool names, arguments and results,
turn results, token usage, retries, and errors. HTTP authorization headers and
the configured API key are not recorded. User messages, system prompts, tool
output, and other potentially sensitive application data are recorded, so the
directory should be treated as private runtime data.

Interactive progress uses paired dark-blue lines with an explicit 256-color
palette value so a slow operation does not look like a stalled process. Reliability preflight and every logical LLM
operation print a start line followed by success or failure and the usage
reported for that operation. Tool execution prints its arguments before the
call and its result afterward. A tool's `selection tokens` are the tokens spent
by the immediately preceding LLM operation that produced that call; executing
the tool itself does not consume LLM tokens. For example:

```text
[preflight] calling LLM
[preflight] success; tokens: in 500, out 80, total 580; completed
[llm:decision] calling LLM
[llm:decision] success; tokens: in 1200, out 300, total 1500; completed
[tool:web.search] calling {"query":"lead density"}
[tool:web.search] success; selection tokens: in 1200, out 300, total 1500; completed
```

Terminal rendering truncates an argument block after 2,000 characters to
remain readable; the full arguments remain available in the JSONL trace. ANSI
color is emitted even when stdout is wrapped or redirected because some VM and
container consoles do not report a TTY; set `NO_COLOR` to disable it explicitly.

## Development

`AgentSession` is the stream-independent entry point for programmatic and e2e
tests. It owns conversation history and cumulative usage, executes one request
with `run()`, and reports tool progress through an optional structured
`ToolEvent` callback. The terminal REPL is a thin consumer of the same API, so
tests do not need to replace global stdin or stdout streams. Events identify
their `category` (`llm` or `tool`), `phase` (`started` or `finished`), status,
optional arguments, and per-operation token usage.
`last_preflight_roles` exposes the most recent planner decision for diagnostics.
Pass `reliability_preflight=False` to the constructor for deterministic tests
that exercise only the completion/tool loop.

```python
events = []
session = AgentSession(client, renderer, tools, history_limit=24, max_tool_calls=4)
result = session.run("Solve the task", events.append)
print(result.answer)
```

Run the full test suite and basic whitespace validation before committing:

```bash
python3 -m unittest -v
git diff --check
```

The tests use fake providers and sandboxes where practical. MCP integration
tests start temporary stdio and loopback HTTP servers, but the suite does not
require a running model server, SearXNG instance, or Docker daemon.
