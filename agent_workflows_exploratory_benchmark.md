# LLMs Can Write Tool Workflows. Will They Choose To?

**An exploratory benchmark of direct tool calls and executable orchestration**  
**24 September 2026 · Technical note**

The agent in this experiment could either call tools one at a time or write a Python program that calls those same tools inside a sandbox. When both options were available, the tested model chose the program route **zero times in ten tasks**. When the direct tools were removed from its top-level catalog, it did write programs and sometimes used far fewer tokens. That change also exposed costly program-generation failures and one extreme nested-LLM outlier.

This is an observation about **one model, one agent configuration, and one run per task and mode**. It motivates a question for agent builders and model trainers: can a model learn *when* to turn a tool sequence into an executable workflow, as well as *how* to write that workflow?

## The setup

The [open-source terminal agent](https://github.com/kukushko/agent) provides ordinary tools for files, web search, page fetching, and Python execution. It also provides `jobs.run`, which executes model-written Python in a Docker sandbox. A job can call delegated tools, branch on their results, process intermediate data locally, and return a compact result. It may also invoke a delegated `llm.ask` for a semantic step.

Two top-level tool catalogs were compared:

| Mode | Tools available to the main model |
| --- | --- |
| **Hybrid** | Ordinary direct tools **and** `jobs.run` |
| **Jobs** | Only the orchestration entrypoint `jobs.run`; the program inside the job can still call the delegated tools |

The model was `/models/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf` behind an OpenAI-compatible local endpoint, with thinking enabled, a 131,072-token context window, a 16,384-token main completion limit, reliability preflight, and a default limit of four sequential outer tool calls. The sandbox used Python 3.12. Ten practical tasks were run once in each mode, in fresh agent sessions. The modes were alternated between tasks to reduce systematic effects from endpoint load and changing web data. Temperature and other agent defaults were unchanged. There was no additional explicit-instruction or few-shot ablation.

**Accounting:** The main-agent token count includes its preflight and continuation requests. The jobs-mode *combined* count adds the input and output tokens of nested `llm.ask` calls recorded in each job's state. Delegated file, web, and Python tools consume no LLM tokens on their own. Token totals below are counts, not monetary cost.

## Results

| Task | Hybrid outcome | Jobs outcome | Hybrid tokens | Jobs main | Jobs nested LLM | Jobs combined |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| Current gold price with source | Success | Success | 30,426 | 28,644 | 0 | 28,644 |
| Current Bitcoin price with source and timestamp | Success | Success after 2 jobs | 45,234 | 10,272 | 16,157 | 26,429 |
| Latest Python 3.12 release and date | Success | Success | 27,768 | 5,917 | 24,450 | 30,367 |
| Total mass of two hollow lead spheres | Success | Success | 15,762 | 7,634 | 2,342 | 9,976 |
| Five-term Taylor approximation of `sin(52°)` | Success | Success after 2 jobs | 11,576 | 10,923 | 0 | 10,923 |
| Three longest non-empty lines in `result.py` | Failed: outer call limit | Success | 38,606 | 5,615 | 0 | 5,615 |
| Semantic search for Moon-landing mentions | Timed out | Timed out | — | — | — | — |
| Fetch and summarize `example.com` | Success | Success | 20,992 | 5,768 | 2,982 | 8,750 |
| Current EUR/USD rate with provenance | Success | Failed: outer call limit | 31,833 | 19,742 | 10,436 | 30,178 |
| Qwen developer and Qwen3-8B license | Success | Success after 3 jobs | 39,347 | 14,932 | 92,456 | 107,388 |

Strict task success was **8/10 in each mode**. On the nine pairs without the semantic-search timeout:

| Measure | Hybrid | Jobs, main only | Jobs, combined |
| --- | ---: | ---: | ---: |
| Total tokens | 261,544 | 109,447 | 258,270 |
| Mean tokens per task | 29,060 | 12,161 | 28,697 |
| Median comparable total | 30,426 | — | 26,429 |
| Total wall time | ~130.7 s | — | ~267.2 s |

The jobs total is only **1.3% lower** than hybrid after including nested LLM usage, while wall time is about **2.0× higher**. Excluding the Qwen information task *for diagnosis only*, jobs used about **32% fewer tokens** across the other eight non-timeout pairs. The excluded task is real observed cost and belongs in the primary total. Both semantic-search runs hit a 240-second harness timeout.

## What happened

### The model could orchestrate, but did not choose to in hybrid mode

`jobs.run` and guidance on when to use it were present throughout the hybrid sessions. The model nevertheless made **0/10 voluntary `jobs.run` selections**. In jobs-only mode, it generated programs, including workflows with fallback searches, URL selection, batch fetching, file reading, semantic extraction, and compact output. This separates two questions in this setup: whether the model *can* construct a workflow, and whether it *selects* that option when direct calls are also available.

The small sample cannot show that models generally prefer local actions, or that another prompt could not change this model's choice. It does show that making the orchestration tool available and describing its intended use did not suffice in these ten sessions. Constraining the top-level action space changed the observed behavior.

### Keeping intermediate data in the job sometimes helped greatly

The longest-lines task succeeded in jobs mode with **5,615** tokens; hybrid used **38,606** and exhausted its outer call budget. Fetching and summarizing `example.com` used **8,750** combined jobs tokens versus **20,992** in hybrid. The lead-spheres calculation used **9,976** versus **15,762**, and the Bitcoin task **26,429** versus **45,234**.

These cases suggest a useful mechanism: a program can pass intermediate tool results directly between operations, so the main model need not repeatedly receive and reason over every page, file, or partial calculation. The comparison does not isolate that mechanism from all other effects of changing the catalog and generated actions.

### Errors and nested inference could erase the benefit

Generated jobs sometimes lacked the required global `result`, referenced undefined variables, passed the wrong type to `llm.ask.data`, failed to handle empty search results, or assumed incorrect result fields. Repair often meant generating a new job: two jobs for Bitcoin and Taylor, four for EUR/USD before failure, and three for the Qwen information task.

The Qwen task incurred **92,456 nested tokens**, including two failed nested attempts of 53,912 input + 1,024 output tokens and 37,008 input + 512 output tokens. Its jobs total reached **107,388**, about **2.7×** hybrid's **39,347**. The final answer was substantially correct, though one cited URL appeared questionable. Moving work into a job does not save tokens if the job sends large contexts back through another LLM loop.

Hybrid had a failure of its own: in the longest-lines task, it used unsuitable calls and hit the four-call outer limit before answering. The semantic-search timeout affected both modes and points to the underlying indexing operation rather than to the choice of top-level catalog. Recursive search over the working directory included roughly 202 files and accumulated downloads; both sessions reached the 240-second cutoff.

## Interpretation and limits

For this model and configuration, **hybrid remains the sensible default**: both modes solved eight tasks, jobs was roughly twice as slow on the nine non-timeout pairs, and a single nested-LLM outlier erased most of its token advantage. Jobs is already promising for deterministic transformations, aggregation, and predictable multi-step workflows. The present result is a case study in tool selection and cost distribution, rather than a win for one mode on average.

There are important limits to generalization:

- One quantized model was tested, with one run per task and mode; there are no variance estimates or cross-model comparisons.
- Ten tasks do not establish a general tool-selection frequency. The 0/10 observation applies to these sessions and their available tools and guidance.
- Success judgments included source and answer quality; the reported counts should be read with the original task-level descriptions, not treated as a standardized benchmark score.
- Web data and endpoint load can change between paired runs despite alternating modes.
- The outer tool-call limit of four affects failures and may amplify the value of a single job.
- The one-off harness did not terminate a timed-out process automatically; the benchmark-created processes were cleaned up manually. This does not measure the agent's normal shutdown path.
- The repository contains this benchmark summary and agent implementation, but the note alone does not provide a complete set of prompts, raw trajectories, and automated graders for independent exact replication.

## Next experiment

Keep the task set and evaluation criteria fixed; publish exact prompts, per-run traces, and the harness. Run repeated trials across several models with all tokens, time, retries, and answer quality recorded. For the *same model*, compare four catalogs or policies: **hybrid**, **hybrid with stronger orchestration instructions**, **hybrid with examples that demonstrate when to choose a job**, and **jobs-only**. Include tasks where a direct call is cheaper as well as tasks where a program should win. Limit or separately budget nested `llm.ask` calls.

The key outcome is not simply how often the model writes Python. It is whether the model chooses orchestration *on the tasks where it improves end-to-end cost and success*, while retaining direct calls when they are preferable.

---

**Source:** [Original benchmark and full engineering notes](https://github.com/kukushko/agent/blob/4c5f0e441de66678ecb9bc973bf5f30f7b72ea5a/BENCHMARK_JOBS_MODE.md) · [Project and implementation](https://github.com/kukushko/agent). This edited technical note preserves the reported measurements and does not claim new runs.
