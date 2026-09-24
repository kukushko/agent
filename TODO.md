# Future Work

## Job-Centric Tool Mode

- Explore an optional mode where the main model sees only `jobs.run`, while a
  job can use delegated `tools.*` capabilities through MCP.
- Evaluate direct, job-centric, and hybrid modes with repeatable e2e benchmarks
  before changing the default tool exposure model.

## Semantic Retrieval Follow-ups

- Evaluate lexical or hybrid ranking, better language-aware chunking, and ONNX
  INT8 quantization against the initial dense BGE-M3 implementation.
- Add explicit index inspection and cleanup administration only if normal lazy
  synchronization proves insufficient.
