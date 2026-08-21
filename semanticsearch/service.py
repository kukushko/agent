"""Incremental disk-backed semantic search with lazy model initialization."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import re
import secrets
import shutil
import threading
import time
from bisect import bisect_right
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from workpaths import WorkPathResolver


INDEX_FORMAT_VERSION = 1
CHUNKER_NAME = "character-window-v1"
SUPPORTED_SUFFIXES = frozenset(
    (".csv", ".json", ".md", ".py", ".rst", ".text", ".txt", ".yaml", ".yml")
)
REQUIRED_MODEL_FILES = (
    "config.json",
    "tokenizer.json",
    "onnx/model.onnx",
    "onnx/model.onnx_data",
)


@dataclass(frozen=True)
class SemanticSearchAvailability:
    available: bool
    reason: str | None = None


class SemanticSearchService:
    """Build mirrored per-file indexes and reuse one lazily loaded embedding model."""

    def __init__(
        self,
        work_dir: Path,
        index_root: Path,
        model_path: Path,
        chunk_chars: int = 1_800,
        overlap_chars: int = 200,
        batch_size: int = 8,
        max_file_bytes: int = 8_388_608,
        max_files: int = 500,
        max_scope_bytes: int = 67_108_864,
    ) -> None:
        self.paths = WorkPathResolver(work_dir)
        self.work_dir = self.paths.root
        self.index_root = index_root.resolve()
        self.model_path = model_path.resolve()
        self.chunk_chars = chunk_chars
        self.overlap_chars = overlap_chars
        self.batch_size = batch_size
        self.max_file_bytes = max_file_bytes
        self.max_files = max_files
        self.max_scope_bytes = max_scope_bytes
        if self.index_root == self.work_dir or self.index_root.is_relative_to(self.work_dir):
            raise ValueError("files:semantic_index_root must be outside the work directory")
        if chunk_chars < 256:
            raise ValueError("files:semantic_chunk_chars must be at least 256")
        if not 0 <= overlap_chars < chunk_chars:
            raise ValueError(
                "files:semantic_overlap_chars must be non-negative and smaller than semantic_chunk_chars"
            )
        if batch_size < 1:
            raise ValueError("files:semantic_batch_size must be positive")
        if max_file_bytes < 1:
            raise ValueError("files:semantic_max_file_bytes must be positive")
        if max_files < 1:
            raise ValueError("files:semantic_max_files must be positive")
        if max_scope_bytes < 1:
            raise ValueError("files:semantic_max_scope_bytes must be positive")
        self._model: Any | None = None
        self._load_failure: str | None = None
        self._lock = threading.RLock()
        self._model_fingerprint = model_fingerprint(self.model_path)

    @classmethod
    def probe(cls, model_path: Path) -> SemanticSearchAvailability:
        """Perform only cheap filesystem and import-spec checks."""
        resolved = model_path.resolve()
        if not resolved.is_dir():
            return SemanticSearchAvailability(False, f"model directory not found: {resolved}")
        missing = [name for name in REQUIRED_MODEL_FILES if not (resolved / name).is_file()]
        if missing:
            return SemanticSearchAvailability(
                False, f"model directory is incomplete; missing {missing[0]}"
            )
        missing_modules = [
            name
            for name in ("numpy", "onnxruntime", "tokenizers")
            if importlib.util.find_spec(name) is None
        ]
        if missing_modules:
            return SemanticSearchAvailability(
                False, f"missing Python dependency: {missing_modules[0]}"
            )
        return SemanticSearchAvailability(True)

    def search(
        self,
        query: str,
        path: str,
        file_pattern: str | None,
        max_results: int,
        min_score: float,
    ) -> dict[str, object]:
        """Synchronize the requested scope and return its closest text chunks."""
        started = time.monotonic()
        with self._lock:
            base = self.paths.resolve(path)
            if not base.exists():
                raise ValueError(f"path does not exist: {path}")
            files = self._source_files(base, file_pattern)
            if len(files) > self.max_files:
                raise ValueError(
                    f"semantic search scope contains more than {self.max_files} supported files"
                )
            scope_bytes = sum(source.stat().st_size for source in files)
            if scope_bytes > self.max_scope_bytes:
                raise ValueError(
                    f"semantic search scope exceeds {self.max_scope_bytes} bytes"
                )
            if not files:
                return self._empty_result(query, base, started)
            model, cold_start, load_seconds = self._get_model()
            indexed = reused = 0
            indexes: list[tuple[Path, list[dict[str, object]], Any]] = []
            index_started = time.monotonic()
            for source in files:
                loaded = self._load_current_index(source)
                if loaded is None:
                    loaded = self._build_index(source, model)
                    indexed += 1
                else:
                    reused += 1
                indexes.append((source, loaded[0], loaded[1]))
            removed = self._remove_orphan_indexes(base, set(files))
            index_seconds = time.monotonic() - index_started

            import numpy as np

            search_started = time.monotonic()
            query_vector = model.encode(
                [query],
                batch_size=1,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )[0].astype(np.float32, copy=False)
            matches: list[dict[str, object]] = []
            for source, chunks, embeddings in indexes:
                scores = embeddings @ query_vector
                for chunk, score in zip(chunks, scores.tolist()):
                    if score < min_score:
                        continue
                    matches.append(
                        {
                            "path": source.relative_to(self.work_dir).as_posix(),
                            **chunk,
                            "score": round(float(score), 6),
                        }
                    )
            matches.sort(key=lambda item: (-float(item["score"]), str(item["path"]), int(item["chunk"])))
            matches = matches[:max_results]
            search_seconds = time.monotonic() - search_started
            return {
                "query": query,
                "path": base.relative_to(self.work_dir).as_posix() or ".",
                "matches": matches,
                "returned": len(matches),
                "files_considered": len(files),
                "indexed": indexed,
                "reused": reused,
                "removed": removed,
                "cold_start": cold_start,
                "timings": {
                    "model_load_seconds": round(load_seconds, 3),
                    "index_seconds": round(index_seconds, 3),
                    "search_seconds": round(search_seconds, 3),
                    "total_seconds": round(time.monotonic() - started, 3),
                },
            }

    def _get_model(self):
        if self._model is not None:
            return self._model, False, 0.0
        if self._load_failure is not None:
            raise RuntimeError(f"semantic embedding model failed to load: {self._load_failure}")
        started = time.monotonic()
        try:
            self._model = OnnxEmbeddingModel(self.model_path)
        except Exception as exc:
            self._load_failure = str(exc)
            raise RuntimeError(f"semantic embedding model failed to load: {exc}") from exc
        return self._model, True, time.monotonic() - started

    def _source_files(self, base: Path, pattern: str | None) -> list[Path]:
        candidates = [base] if base.is_file() else sorted(base.rglob("*"))
        result = []
        for candidate in candidates:
            try:
                resolved = candidate.resolve()
                relative = resolved.relative_to(self.work_dir)
            except (OSError, ValueError):
                continue
            if not candidate.is_file() or candidate.is_symlink():
                continue
            if resolved.suffix.lower() not in SUPPORTED_SUFFIXES:
                continue
            if pattern is not None and not relative.match(pattern):
                continue
            if resolved.stat().st_size > self.max_file_bytes:
                continue
            result.append(resolved)
        return result

    def _index_dir(self, source: Path) -> Path:
        relative = source.relative_to(self.work_dir)
        return self.index_root / relative.parent / f"{relative.name}.index"

    def _load_current_index(self, source: Path):
        import numpy as np

        index_dir = self._index_dir(source)
        try:
            metadata = json.loads((index_dir / "metadata.json").read_text("utf-8"))
            if metadata != self._expected_metadata(source, metadata.get("chunk_count")):
                return None
            chunks = [json.loads(line) for line in (index_dir / "chunks.jsonl").read_text("utf-8").splitlines()]
            embeddings = np.load(index_dir / "embeddings.npy", allow_pickle=False)
            if len(chunks) != len(embeddings) or len(chunks) != metadata["chunk_count"]:
                return None
            return chunks, embeddings
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
            return None

    def _build_index(self, source: Path, model):
        import numpy as np

        text = source.read_text(encoding="utf-8")
        chunks = chunk_text(text, self.chunk_chars, self.overlap_chars)
        if not chunks:
            chunks = [{"chunk": 0, "start_line": 1, "end_line": 1, "content": ""}]
        embeddings = model.encode(
            [str(chunk["content"]) for chunk in chunks],
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).astype(np.float32, copy=False)
        index_dir = self._index_dir(source)
        index_dir.mkdir(parents=True, exist_ok=True)
        write_text_atomic(
            index_dir / "chunks.jsonl",
            "".join(json.dumps(chunk, ensure_ascii=False) + "\n" for chunk in chunks),
        )
        temporary_embeddings = index_dir / f".embeddings.{secrets.token_hex(4)}.npy"
        np.save(temporary_embeddings, embeddings, allow_pickle=False)
        os.replace(temporary_embeddings, index_dir / "embeddings.npy")
        write_text_atomic(
            index_dir / "metadata.json",
            json.dumps(self._expected_metadata(source, len(chunks)), indent=2) + "\n",
        )
        return chunks, embeddings

    def _expected_metadata(self, source: Path, chunk_count: object) -> dict[str, object]:
        return {
            "source_path": source.relative_to(self.work_dir).as_posix(),
            "source_sha256": file_sha256(source),
            "source_size": source.stat().st_size,
            "index_format_version": INDEX_FORMAT_VERSION,
            "chunker": {
                "name": CHUNKER_NAME,
                "max_chars": self.chunk_chars,
                "overlap_chars": self.overlap_chars,
            },
            "embedding_model": {
                "path": self.model_path.name,
                "fingerprint": self._model_fingerprint,
            },
            "chunk_count": chunk_count,
        }

    def _remove_orphan_indexes(self, base: Path, sources: set[Path]) -> int:
        relative = base.relative_to(self.work_dir)
        mirror = self.index_root / relative
        candidates = [self._index_dir(base)] if base.is_file() else list(mirror.rglob("*.index")) if mirror.exists() else []
        removed = 0
        for index_dir in candidates:
            if not index_dir.is_dir():
                continue
            try:
                metadata = json.loads((index_dir / "metadata.json").read_text("utf-8"))
                source = (self.work_dir / metadata["source_path"]).resolve()
                source.relative_to(self.work_dir)
            except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
                source = None
            if source not in sources:
                shutil.rmtree(index_dir)
                removed += 1
        return removed

    def _empty_result(self, query: str, base: Path, started: float) -> dict[str, object]:
        return {
            "query": query,
            "path": base.relative_to(self.work_dir).as_posix() or ".",
            "matches": [],
            "returned": 0,
            "files_considered": 0,
            "indexed": 0,
            "reused": 0,
            "removed": self._remove_orphan_indexes(base, set()),
            "cold_start": False,
            "timings": {"model_load_seconds": 0.0, "index_seconds": 0.0, "search_seconds": 0.0, "total_seconds": round(time.monotonic() - started, 3)},
        }


def chunk_text(text: str, max_chars: int, overlap_chars: int) -> list[dict[str, object]]:
    """Split text into overlapping, whitespace-aware windows with line coordinates."""
    if not text.strip():
        return []
    line_starts = [0]
    line_starts.extend(match.end() for match in re.finditer("\n", text))
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + max_chars, len(text))
        if end < len(text):
            boundary = max(text.rfind("\n", start + max_chars // 2, end), text.rfind(" ", start + max_chars // 2, end))
            if boundary > start:
                end = boundary
        content = text[start:end].strip()
        if content:
            chunks.append(
                {
                    "chunk": len(chunks),
                    "start_line": bisect_right(line_starts, start),
                    "end_line": bisect_right(line_starts, max(start, end - 1)),
                    "content": content,
                }
            )
        if end >= len(text):
            break
        start = max(start + 1, end - overlap_chars)
    return chunks


class OnnxEmbeddingModel:
    """Small SentenceTransformer-compatible wrapper for the bundled BGE-M3 ONNX export."""

    def __init__(self, model_path: Path) -> None:
        import onnxruntime as ort
        from tokenizers import Tokenizer

        self.tokenizer = Tokenizer.from_file(str(model_path / "tokenizer.json"))
        self.tokenizer.enable_truncation(max_length=8_192)
        self.session = ort.InferenceSession(
            str(model_path / "onnx" / "model.onnx"),
            providers=["CPUExecutionProvider"],
        )
        self.input_names = {value.name for value in self.session.get_inputs()}

    def encode(
        self,
        sentences: list[str],
        *,
        batch_size: int,
        convert_to_numpy: bool,
        normalize_embeddings: bool,
        show_progress_bar: bool,
    ):
        import numpy as np

        del convert_to_numpy, show_progress_bar
        vectors = []
        for offset in range(0, len(sentences), batch_size):
            encodings = self.tokenizer.encode_batch(sentences[offset : offset + batch_size])
            width = max(len(encoding.ids) for encoding in encodings)
            input_ids = np.full((len(encodings), width), 1, dtype=np.int64)
            attention_mask = np.zeros((len(encodings), width), dtype=np.int64)
            for row, encoding in enumerate(encodings):
                length = len(encoding.ids)
                input_ids[row, :length] = encoding.ids
                attention_mask[row, :length] = 1
            inputs = {"input_ids": input_ids, "attention_mask": attention_mask}
            if "token_type_ids" in self.input_names:
                inputs["token_type_ids"] = np.zeros_like(input_ids)
            outputs = self.session.run(None, inputs)
            batch_vectors = np.asarray(outputs[0])[:, 0, :].astype(np.float32)
            if normalize_embeddings:
                norms = np.linalg.norm(batch_vectors, axis=1, keepdims=True)
                batch_vectors /= np.maximum(norms, np.finfo(np.float32).eps)
            vectors.append(batch_vectors)
        return np.concatenate(vectors, axis=0)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1_048_576), b""):
            digest.update(block)
    return digest.hexdigest()


def model_fingerprint(model_path: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(model_path.rglob("*")):
        if not path.is_file() or path.name.startswith(".") or "imgs" in path.parts:
            continue
        stat = path.stat()
        digest.update(path.relative_to(model_path).as_posix().encode())
        digest.update(str(stat.st_size).encode())
        digest.update(str(stat.st_mtime_ns).encode())
        if stat.st_size <= 1_048_576:
            digest.update(path.read_bytes())
    return digest.hexdigest()


def write_text_atomic(path: Path, content: str) -> None:
    temporary = path.with_name(f".{path.name}.{secrets.token_hex(4)}.tmp")
    temporary.write_text(content, encoding="utf-8")
    os.replace(temporary, path)
