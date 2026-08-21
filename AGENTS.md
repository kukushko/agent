# Project Instructions

## Language

- Write and maintain source code, comments, identifiers, tests, commit messages,
  and project documentation in English.
- Communicate with the project owner in Russian unless they explicitly request
  another language.

## Project Maintenance

- Keep implementation, tests, and documentation consistent with one another.
- Update README.md and other relevant documentation whenever behavior,
  configuration, commands, defaults, or user-facing features change.
- Add or update automated tests for behavioral changes and bug fixes when
  practical.
- Prefer clear, simple, maintainable solutions over unnecessary abstraction.
- Preserve existing user files and unrelated changes in the working tree.
- Run the relevant test suite after modifying the code and report the result.

## Architecture Boundaries

- Treat separation between tool packages and infrastructure as a mandatory
  maintainability constraint, not an optional cleanup concern.
- Infrastructure such as the agent, MCP bridge, sandbox, job runtime, registry,
  and prompt assembly must not branch on or embed concrete tool package names,
  namespaces, method names, result fields, or domain semantics.
- Keep tool-specific behavior, availability requirements, delegation rules,
  resource costs, quota defaults, prompt guidance, and result conventions in
  the owning tool package and its declarative metadata.
- Transport and orchestration layers may validate, preserve, render, and enforce
  generic metadata contracts, but must treat tool and resource identifiers as
  opaque values.
- Supply optional package-specific capabilities through explicit service
  contracts in `ToolEnvironment.services`; do not add concrete package services
  to the shared environment interface.
- Prefer extending a generic metadata or service contract when infrastructure
  appears to need knowledge of a particular tool. Do not solve the immediate
  case with name comparisons, suffix checks, hard-coded prompt examples, or
  special-case dispatch.
- Preserve compatibility for external MCP tools that do not provide project
  metadata through documented, conservative generic defaults.
- Add boundary-focused tests when changing tool discovery, metadata transport,
  prompt construction, delegation, quotas, or orchestration. A useful check is
  that infrastructure continues to work with synthetic tool and resource names.
