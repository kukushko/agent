# Future Work

## Job-Centric Tool Mode

- Explore an optional mode where the main model sees only `jobs.run`, while a
  job can use delegated `tools.*` capabilities through MCP.
- Add a delegated-only, stateless `llm.ask(instruction, data, schema)` tool for
  bounded semantic interpretation inside jobs. It must return schema-validated
  JSON, expose no tools of its own, prevent recursion, and have strict per-job
  call, token, and time quotas.
- Introduce generic direct/delegated exposure metadata rather than branching on
  concrete tool names. Preserve conservative behavior for external MCP tools
  without project-specific metadata.
- Evaluate direct, job-centric, and hybrid modes with repeatable e2e benchmarks
  before changing the default tool exposure model.

## Semantic Retrieval Follow-ups

- Evaluate lexical or hybrid ranking, better language-aware chunking, and ONNX
  INT8 quantization against the initial dense BGE-M3 implementation.
- Add explicit index inspection and cleanup administration only if normal lazy
  synchronization proves insufficient.
