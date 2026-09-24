# Hybrid vs. Jobs Tool Mode Benchmark

Date: 2026-09-24

## Purpose

This benchmark compares the agent's `hybrid` and `jobs` direct-tool modes on
the same set of practical tasks. It is intended as an exploratory engineering
measurement, not a statistically rigorous model evaluation.

The main questions were:

- Can the model build useful end-to-end programs around `jobs.run`?
- Does keeping intermediate tool results inside a job reduce token use?
- What failure modes remain when the model must generate the orchestration
  program itself?
- Is jobs mode ready to replace hybrid mode as the default?

## Environment

- Model: `/models/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf`
- OpenAI-compatible endpoint: `http://192.168.0.108:8000/v1`
- Model context window: 131072 tokens
- Main completion limit: 16384 tokens
- Thinking: enabled and not assumed to be disableable
- Python sandbox: Python 3.12 in Docker
- Reliability preflight: enabled
- Default sequential outer tool-call limit: 4
- Runs: one fresh agent session for each task and mode
- Total sessions: 20 (10 tasks in both modes)

The two modes were alternated between tasks to reduce systematic bias from
endpoint load or changing web data. Temperature and all other agent defaults
were left unchanged.

## Token Accounting

The terminal agent's final token line includes all main-agent LLM requests in
the turn, including preflight and continuation requests. It does not include
nested model requests made by `llm.ask` inside a job.

For jobs mode, this report adds `llm_input_tokens` and `llm_output_tokens` from
the corresponding persisted `tmp/jobs/<job-id>/state.json` files. Therefore,
the jobs-mode totals below include both main-agent and nested-model usage.

Ordinary delegated tools such as file access, web search, page fetching, and
Python execution do not consume LLM tokens by themselves.

## Tasks and Results

| Task | Hybrid result | Jobs result | Hybrid tokens | Jobs main tokens | Nested job LLM tokens | Jobs combined tokens |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| Current gold price with source | Success | Success | 30,426 | 28,644 | 0 | 28,644 |
| Current Bitcoin price with source and timestamp | Success | Success after 2 jobs | 45,234 | 10,272 | 16,157 | 26,429 |
| Latest Python 3.12 release and date | Success | Success | 27,768 | 5,917 | 24,450 | 30,367 |
| Total mass of two hollow lead spheres | Success | Success | 15,762 | 7,634 | 2,342 | 9,976 |
| Five-term Taylor approximation of `sin(52°)` | Success | Success after 2 jobs | 11,576 | 10,923 | 0 | 10,923 |
| Three longest non-empty lines in `result.py` | Failed: outer call limit | Success | 38,606 | 5,615 | 0 | 5,615 |
| Semantic search for Moon-landing mentions | Timed out | Timed out | N/A | N/A | N/A | N/A |
| Fetch and summarize `example.com` | Success | Success | 20,992 | 5,768 | 2,982 | 8,750 |
| Current EUR/USD rate with provenance | Success | Failed: outer call limit | 31,833 | 19,742 | 10,436 | 30,178 |
| Qwen developer and Qwen3-8B license | Success | Success after 3 jobs | 39,347 | 14,932 | 92,456 | 107,388 |

Strict task success was 8/10 in both modes.

For the nine task pairs that completed or failed without the semantic-search
timeout:

| Metric | Hybrid | Jobs, main only | Jobs, including nested LLM |
| --- | ---: | ---: | ---: |
| Total tokens | 261,544 | 109,447 | 258,270 |
| Mean tokens per task | 29,060 | 12,161 | 28,697 |
| Median comparable total | 30,426 | N/A | 26,429 |

The combined jobs total was close to hybrid overall because two failed nested
LLM attempts in the Qwen task consumed 92,456 tokens. Excluding that outlier,
jobs mode used approximately 32% fewer tokens across the remaining eight
non-timeout tasks. This exclusion is diagnostic only; the outlier is part of
the real observed cost and must not be omitted from the primary comparison.

## Latency

For the nine non-semantic task pairs:

- Hybrid total wall time: approximately 130.7 seconds
- Jobs total wall time: approximately 267.2 seconds
- Hybrid mean: approximately 14.5 seconds per task
- Jobs mean: approximately 29.7 seconds per task

Both semantic-search runs reached the 240-second harness timeout. Including
those timeouts, the observed totals were approximately 370.7 seconds for
hybrid and 507.2 seconds for jobs.

Jobs mode was therefore roughly twice as slow on the non-timeout tasks. Its
additional cost came from Docker startup, multiple delegated calls inside a
job, and nested `llm.ask` requests.

## Positive Findings

Jobs mode performed especially well when the requested operation mapped
naturally to a deterministic program:

- The longest-line task used 5,615 tokens instead of 38,606 and succeeded where
  hybrid exhausted its outer tool-call budget.
- Fetching and processing `example.com` used 8,750 combined tokens instead of
  20,992.
- The lead-density lookup plus numerical calculation used 9,976 combined
  tokens instead of 15,762.
- The Bitcoin task used 26,429 combined tokens instead of 45,234.

These results validate the central jobs-mode idea: intermediate data can remain
inside the sandbox and does not always need to be copied through the main model
context.

The model also demonstrated that it can generate a complete web workflow in a
single job: fallback searches, URL selection, batch fetching, managed-file
reading, nested semantic extraction, and a compact result.

## Observed Failure Modes

### Generated Python quality

Jobs failed or required retries because generated programs contained:

- no assignment to the required global `result` variable;
- an undefined variable;
- a value of the wrong type for `llm.ask.data`;
- incomplete handling of empty search results;
- incorrect assumptions about a tool result's fields.

A small code-generation error is expensive because the main model typically
generates and starts an entirely new job to repair it.

### Repeated outer jobs

Observed `jobs.run` counts included:

- Bitcoin: 2 jobs;
- Taylor approximation: 2 jobs;
- EUR/USD: 4 jobs, ending in failure;
- Qwen information: 3 jobs.

The model can construct an end-to-end job, but does not do so consistently.

### Nested LLM outliers

Nested `llm.ask` can eliminate the expected token savings when the job sends it
large page contents or it exhausts its output budget while reasoning.

During the Qwen task, two failed nested attempts used:

- 53,912 input and 1,024 output tokens;
- 37,008 input and 512 output tokens.

The final answer was substantially correct, but the combined jobs-mode cost was
2.7 times the hybrid cost. One returned source URL also appeared questionable,
although other supplied sources supported the conclusion.

### Hybrid iteration failures

Hybrid mode is not uniformly safer. In the longest-line task it issued
`files.search`, an unsuitable failing `python.eval`, `files.list`, and
`files.read`, then exhausted the outer call limit before answering. Jobs mode
completed the same operation in one successful call.

### No voluntary orchestration in hybrid mode

`jobs.run` was available to the model alongside all ordinary tools in every
hybrid session. Nevertheless, the model did not voluntarily select it in any
of the ten hybrid tasks, including tasks that required a predictable sequence
of searches, page fetches, file reads, filtering, or calculation. It preferred
to request each ordinary tool interactively and return every intermediate
result through the main context.

This is important because the jobs mechanism was not merely hidden from the
hybrid model or unavailable to it. The model had both the tool declaration and
prompt guidance explaining when orchestration would be useful, yet it did not
choose to create a task-specific program to reduce future calls and context
growth.

The observation supports a broader working hypothesis about current LLM
agents: when direct tools are available, they tend to optimize for the next
immediate action rather than for the total cost of the complete tool workflow.
They do not reliably invent an intermediate executable procedure, even when
doing so would reduce repeated inference and token use. In effect, forcing the
jobs-only catalog changed the model's planning behavior in a way that guidance
alone did not.

This benchmark used one model and cannot establish that the limitation is
universal or fundamental to every LLM architecture. It does, however, show
that tool availability plus written encouragement is insufficient for this
model. Any design that depends on voluntary self-orchestration should therefore
be evaluated empirically rather than assumed to work.

### Semantic indexing scope

Both modes timed out while running recursive semantic search over `path="."`.
At the time of the test, `work/` contained approximately 202 files, many of
them accumulated downloads under `offline/`. The index tree contained about
258 files and occupied approximately 3.2 MiB, but CPU indexing and
synchronization still failed to finish within 240 seconds.

This indicates that an unrestricted recursive semantic scope becomes
impractical as downloaded material accumulates. It is independent of the
direct-tool mode because both modes invoke the same semantic-search service.

### Benchmark harness cleanup

Python's `subprocess.run(..., timeout=...)` raises `TimeoutExpired` without
automatically terminating the timed-out process. The exploratory harness left
one agent, its MCP child, and one job container running after the semantic
timeout. These exact benchmark-created processes were manually terminated.

This was a defect in the one-off benchmark harness, not evidence that the
agent's normal Ctrl+C shutdown path is broken.

## Interpretation

There is no single obvious architectural defect that explains the jobs-mode
failures. Most failures came from the current model's ability to plan and
generate a correct orchestration program on its first attempt.

Conversely, hybrid-mode behavior suggests that a model capable of writing an
orchestration program may still not choose to do so when lower-level tools are
also offered. Tool selection policy and program-generation capability are
separate concerns: jobs-only mode addresses the former by constraining the
action space, while model quality determines the latter.

A more capable model could plausibly make jobs mode the clear winner by:

- constructing the complete workflow before its first call;
- following declared result contracts reliably;
- handling empty results and fallback branches proactively;
- always producing the required `result` value;
- bounding data passed to `llm.ask`;
- avoiding full job regeneration for minor mistakes.

Cheap validation before container execution may still be useful for syntax,
the presence of a `result` assignment, and basic call-signature checks. It
cannot reliably detect incorrect field assumptions, runtime control-flow bugs,
or poor semantic strategy. A model-driven automatic repair phase would trade
fewer visible job failures for additional hidden LLM tokens and latency, and
the benchmark does not establish that this would be beneficial.

## Conclusion

Hybrid should remain the default mode for the current model.

Jobs mode is already valuable for file transformations, aggregation, loops,
calculations, and predictable multi-step workflows. It often reduces tokens
substantially, but its mean advantage is currently erased by rare and very
expensive failures. It is also materially slower.

The most informative next experiment is to repeat this exact task set with a
stronger model and compare success rate, number of regenerated jobs, nested LLM
outliers, combined token use, and latency. Multiple repetitions per task would
be needed before making a default-mode decision.
