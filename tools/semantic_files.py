"""Optional semantic search over files in the configured work directory."""

from __future__ import annotations

from pathlib import Path

from semanticsearch import SemanticSearchService

from .base import Availability, Delegation, ToolEnvironment, ToolPackage, tool


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = "models/bge-m3"
DEFAULT_INDEX_ROOT = "tmp/indexes"
DEFAULT_CHUNK_CHARS = "1800"
DEFAULT_OVERLAP_CHARS = "200"
DEFAULT_BATCH_SIZE = "8"
DEFAULT_MAX_FILE_BYTES = "8388608"
DEFAULT_MAX_FILES = "500"
DEFAULT_MAX_SCOPE_BYTES = "67108864"
DEFAULT_MAX_RESULTS = 10
MAX_RESULTS = 50
MAX_QUERY_CHARS = 4_000


class SemanticFileTools(ToolPackage):
    """Optional semantic file search backed by a local embedding model."""

    namespace = "files"
    parameter_defaults = {
        "semantic_model_path": DEFAULT_MODEL_PATH,
        "semantic_index_root": DEFAULT_INDEX_ROOT,
        "semantic_chunk_chars": DEFAULT_CHUNK_CHARS,
        "semantic_overlap_chars": DEFAULT_OVERLAP_CHARS,
        "semantic_batch_size": DEFAULT_BATCH_SIZE,
        "semantic_max_file_bytes": DEFAULT_MAX_FILE_BYTES,
        "semantic_max_files": DEFAULT_MAX_FILES,
        "semantic_max_scope_bytes": DEFAULT_MAX_SCOPE_BYTES,
    }

    @classmethod
    def check_availability(cls, environment: ToolEnvironment) -> Availability:
        if environment.services.optional(SemanticSearchService) is not None:
            return Availability(True)
        model_path = resolve_project_path(
            environment.tool_parameters.get(cls.namespace, {}).get(
                "semantic_model_path", cls.parameter_defaults["semantic_model_path"]
            )
        )
        availability = SemanticSearchService.probe(model_path)
        return Availability(availability.available, availability.reason)

    def __init__(self, environment: ToolEnvironment) -> None:
        super().__init__(environment)
        if environment.services.optional(SemanticSearchService) is None:
            service = SemanticSearchService(
                work_dir=environment.work_dir,
                index_root=resolve_project_path(self.parameter("semantic_index_root")),
                model_path=resolve_project_path(self.parameter("semantic_model_path")),
                chunk_chars=self._positive_int("semantic_chunk_chars"),
                overlap_chars=self._non_negative_int("semantic_overlap_chars"),
                batch_size=self._positive_int("semantic_batch_size"),
                max_file_bytes=self._positive_int("semantic_max_file_bytes"),
                max_files=self._positive_int("semantic_max_files"),
                max_scope_bytes=self._positive_int("semantic_max_scope_bytes"),
            )
            environment.services.register(SemanticSearchService, service)

    @tool(
        delegation=Delegation(
            costs={"tool_calls": 1, "semantic_searches": 1},
            quota_defaults={"tool_calls": 200, "semantic_searches": 20},
            instructions=(
                "Use semantic search to select relevant excerpts instead of reading large files in full. Results expose excerpt text through .matches, so further files.read calls are usually unnecessary.",
            ),
        ),
        prompt_instructions=(
            "Semantic file search: use files.semantic_search for conceptual or natural-language retrieval across documents, especially downloaded pages or files too large to read in full. Use files.search instead for exact strings, identifiers, or regular expressions.",
            "Semantic matches already contain bounded content plus source paths and line coordinates. Analyze those excerpts directly; do not read every source file afterward unless essential context is demonstrably missing.",
        ),
        epistemic_roles=("workspace",),
        reliability_guidance=(
            "Use semantic file search to retrieve relevant excerpts from large document collections instead of iteratively reading whole files.",
        ),
    )
    def semantic_search(
        self,
        query: str,
        path: str = ".",
        file_pattern: str | None = None,
        max_results: int = DEFAULT_MAX_RESULTS,
        min_score: float = 0.0,
    ) -> dict[str, object]:
        """Find conceptually relevant bounded excerpts in supported UTF-8 work files using a lazy local embedding index."""
        if not query.strip():
            raise ValueError("query must not be empty")
        if len(query) > MAX_QUERY_CHARS:
            raise ValueError(f"query exceeds {MAX_QUERY_CHARS} characters")
        if file_pattern is not None and not file_pattern.strip():
            raise ValueError("file_pattern must not be empty")
        if not 1 <= max_results <= MAX_RESULTS:
            raise ValueError(f"max_results must be between 1 and {MAX_RESULTS}")
        if not -1.0 <= min_score <= 1.0:
            raise ValueError("min_score must be between -1 and 1")
        service = self.environment.services.require(SemanticSearchService)
        return service.search(
            query.strip(), path, file_pattern, max_results, min_score
        )

    def _positive_int(self, name: str) -> int:
        try:
            value = int(self.parameter(name))
        except ValueError as exc:
            raise ValueError(f"files:{name} must be an integer") from exc
        if value < 1:
            raise ValueError(f"files:{name} must be positive")
        return value

    def _non_negative_int(self, name: str) -> int:
        try:
            value = int(self.parameter(name))
        except ValueError as exc:
            raise ValueError(f"files:{name} must be an integer") from exc
        if value < 0:
            raise ValueError(f"files:{name} must be non-negative")
        return value


def resolve_project_path(value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


TOOL_PACKAGES = (SemanticFileTools,)
