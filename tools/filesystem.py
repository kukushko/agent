"""Tools for files inside the configured work directory."""

from __future__ import annotations

import json
import os
import re
import selectors
import shutil
import subprocess
import time
from pathlib import Path
from typing import Iterator

from .base import Delegation, ToolPackage, tool
from workpaths import WorkPathResolver


MAX_READ_CHARS = 32_768
MAX_WRITE_CHARS = 262_144
MAX_LIST_ENTRIES = 200
DEFAULT_SEARCH_RESULTS = 100
MAX_SEARCH_RESULTS = 500
MAX_SEARCH_LINE_CHARS = 1_000
MAX_SEARCH_BUFFER_BYTES = 1_048_576
MAX_SEARCH_ERROR_BYTES = 8_192
MAX_SEARCH_FILE_BYTES = 16_777_216
SEARCH_TIMEOUT_SECONDS = 10.0


class FileTools(ToolPackage):
    """Tools constrained to the configured work directory."""

    namespace = "files"

    def __init__(self, environment) -> None:
        super().__init__(environment)
        self.paths = WorkPathResolver(environment.work_dir)
        self.root = self.paths.root

    @tool(
        epistemic_roles=("workspace",),
        reliability_guidance=(
            "Use workspace tools for claims about files or environment state rather than guessing.",
        ),
    )
    def list(self, path: str = ".") -> dict[str, object]:
        """List files recursively under a directory in the work directory."""
        base = self._resolve(path)
        if not base.exists():
            raise ValueError(f"path does not exist: {path}")
        if not base.is_dir():
            raise ValueError(f"path is not a directory: {path}")

        entries: list[str] = []
        omitted = 0
        for candidate in sorted(base.rglob("*")):
            try:
                resolved = candidate.resolve()
                resolved.relative_to(self.root)
            except (OSError, ValueError):
                continue
            if not candidate.is_file():
                continue
            if len(entries) >= MAX_LIST_ENTRIES:
                omitted += 1
                continue
            entries.append(str(resolved.relative_to(self.root)))
        return {"root": str(self.root), "files": entries, "omitted": omitted}

    @tool(
        delegation=Delegation(
            costs={"tool_calls": 1, "work_file_reads": 1},
            quota_defaults={"tool_calls": 200, "work_file_reads": 100},
            instructions=(
                "Use this tool instead of Python open() for user-requested work files. The returned object exposes text through .content; use .content.splitlines() when processing lines.",
            ),
        ),
        epistemic_roles=("workspace",),
        reliability_guidance=(
            "Use workspace tools for claims about files or environment state rather than guessing.",
        ),
    )
    def read(self, path: str) -> dict[str, object]:
        """Read a UTF-8 file from the work directory."""
        resolved = self._resolve(path)
        if not resolved.exists():
            raise ValueError(f"file does not exist: {path}")
        if not resolved.is_file():
            raise ValueError(f"path is not a file: {path}")
        content = resolved.read_text(encoding="utf-8")
        truncated = len(content) > MAX_READ_CHARS
        if truncated:
            omitted = len(content) - MAX_READ_CHARS
            suffix = f"\n[work file truncated: {omitted} chars omitted]"
            content = content[: MAX_READ_CHARS - len(suffix)].rstrip() + suffix
        return {
            "path": str(resolved.relative_to(self.root)),
            "content": content,
            "truncated": truncated,
        }

    @tool
    def write(self, path: str, content: str) -> dict[str, object]:
        """Write UTF-8 text to a file in the work directory."""
        if len(content) > MAX_WRITE_CHARS:
            raise ValueError(f"content exceeds {MAX_WRITE_CHARS} characters")
        resolved = self._resolve(path)
        resolved.parent.mkdir(parents=True, exist_ok=True)
        resolved.write_text(content, encoding="utf-8")
        return {
            "path": str(resolved.relative_to(self.root)),
            "chars_written": len(content),
        }

    @tool(
        delegation=Delegation(
            allowed=False,
            costs={},
            quota_defaults={},
            reason="Requires the active interactive request context",
        )
    )
    def write_history(self, path: str) -> dict[str, object]:
        """Write the current conversation history to a UTF-8 work file."""
        history = self.environment.context.history
        content = "\n\n".join(
            f"{message.role.upper()}:\n{message.content}" for message in history
        ) or "(empty)"
        if len(content) > MAX_WRITE_CHARS:
            raise ValueError(f"history exceeds {MAX_WRITE_CHARS} characters")
        resolved = self._resolve(path)
        resolved.parent.mkdir(parents=True, exist_ok=True)
        resolved.write_text(content, encoding="utf-8")
        return {
            "path": str(resolved.relative_to(self.root)),
            "messages_written": len(history),
            "chars_written": len(content),
        }

    @tool(
        epistemic_roles=("workspace",),
        reliability_guidance=(
            "Use workspace tools for claims about files or environment state rather than guessing.",
        ),
    )
    def search(
        self,
        query: str,
        path: str = ".",
        file_pattern: str | None = None,
        regex: bool = False,
        case_sensitive: bool = True,
        max_results: int = DEFAULT_SEARCH_RESULTS,
    ) -> dict[str, object]:
        """Search work files recursively and return relative paths, line numbers, columns, and matching text."""
        if not query:
            raise ValueError("query must not be empty")
        if file_pattern is not None and not file_pattern.strip():
            raise ValueError("file_pattern must not be empty")
        if not 1 <= max_results <= MAX_SEARCH_RESULTS:
            raise ValueError(
                f"max_results must be between 1 and {MAX_SEARCH_RESULTS}"
            )

        base = self._resolve(path)
        if not base.exists():
            raise ValueError(f"path does not exist: {path}")
        if not base.is_file() and not base.is_dir():
            raise ValueError(f"path is not a file or directory: {path}")

        ripgrep = shutil.which("rg")
        if ripgrep is not None:
            result = search_with_ripgrep(
                executable=ripgrep,
                root=self.root,
                base=base,
                query=query,
                file_pattern=file_pattern,
                regex=regex,
                case_sensitive=case_sensitive,
                max_results=max_results,
            )
        elif regex:
            raise RuntimeError(
                "regex search requires ripgrep, but the rg executable is unavailable"
            )
        else:
            result = search_with_python(
                root=self.root,
                base=base,
                query=query,
                file_pattern=file_pattern,
                case_sensitive=case_sensitive,
                max_results=max_results,
            )
        return {
            "query": query,
            "path": base.relative_to(self.root).as_posix() or ".",
            **result,
        }

    def _resolve(self, raw_path: str) -> Path:
        return self.paths.resolve(raw_path)


TOOL_PACKAGES = (FileTools,)


def search_with_ripgrep(
    executable: str,
    root: Path,
    base: Path,
    query: str,
    file_pattern: str | None,
    regex: bool,
    case_sensitive: bool,
    max_results: int,
) -> dict[str, object]:
    """Search with ripgrep while bounding time, memory, and returned matches."""
    relative_base = base.relative_to(root).as_posix() or "."
    argv = [
        executable,
        "--json",
        "--line-number",
        "--column",
        "--color",
        "never",
        "--hidden",
        "--no-ignore",
        "--sort",
        "path",
        "--max-columns",
        str(MAX_SEARCH_LINE_CHARS * 4),
        "--max-columns-preview",
        "--max-filesize",
        str(MAX_SEARCH_FILE_BYTES),
    ]
    if not regex:
        argv.append("--fixed-strings")
    if not case_sensitive:
        argv.append("--ignore-case")
    if file_pattern is not None:
        argv.extend(["--glob", file_pattern])
    argv.extend(["--", query, relative_base])

    process = subprocess.Popen(
        argv,
        cwd=root,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert process.stdout is not None
    assert process.stderr is not None
    os.set_blocking(process.stdout.fileno(), False)
    os.set_blocking(process.stderr.fileno(), False)
    selector = selectors.DefaultSelector()
    selector.register(process.stdout, selectors.EVENT_READ, "stdout")
    selector.register(process.stderr, selectors.EVENT_READ, "stderr")
    stdout_buffer = bytearray()
    stderr_buffer = bytearray()
    matches: list[dict[str, object]] = []
    truncated = False
    deadline = time.monotonic() + SEARCH_TIMEOUT_SECONDS
    try:
        while selector.get_map():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                process.kill()
                raise RuntimeError(
                    f"search exceeded the {SEARCH_TIMEOUT_SECONDS:g} second timeout"
                )
            events = selector.select(timeout=min(remaining, 0.1))
            if not events and process.poll() is not None:
                drain_registered_streams(selector, stdout_buffer, stderr_buffer)
                break
            for key, _ in events:
                try:
                    chunk = os.read(key.fileobj.fileno(), 65_536)
                except BlockingIOError:
                    continue
                if not chunk:
                    selector.unregister(key.fileobj)
                    continue
                if key.data == "stderr":
                    remaining_error = MAX_SEARCH_ERROR_BYTES - len(stderr_buffer)
                    stderr_buffer.extend(chunk[: max(0, remaining_error)])
                    continue
                stdout_buffer.extend(chunk)
                if len(stdout_buffer) > MAX_SEARCH_BUFFER_BYTES:
                    process.kill()
                    raise RuntimeError("ripgrep produced an oversized JSON record")
                while b"\n" in stdout_buffer:
                    raw_line, _, rest = stdout_buffer.partition(b"\n")
                    stdout_buffer = bytearray(rest)
                    append_ripgrep_matches(
                        raw_line, matches, max_results + 1
                    )
                    if len(matches) > max_results:
                        truncated = True
                        process.terminate()
                        break
                if truncated:
                    break
            if truncated:
                break
        if not truncated:
            while b"\n" in stdout_buffer:
                raw_line, _, rest = stdout_buffer.partition(b"\n")
                stdout_buffer = bytearray(rest)
                append_ripgrep_matches(
                    raw_line, matches, max_results + 1
                )
                if len(matches) > max_results:
                    truncated = True
                    break
        try:
            return_code = process.wait(timeout=1.0)
        except subprocess.TimeoutExpired:
            process.kill()
            return_code = process.wait(timeout=1.0)
    finally:
        selector.close()
        if process.poll() is None:
            process.kill()
            process.wait(timeout=1.0)
        process.stdout.close()
        process.stderr.close()

    if not truncated and return_code not in (0, 1):
        error = stderr_buffer.decode("utf-8", errors="replace").strip()
        raise RuntimeError(error or f"ripgrep failed with exit code {return_code}")
    if truncated:
        matches = matches[:max_results]
    matches.sort(key=lambda match: (match["path"], match["line"], match["column"]))
    return {
        "engine": "ripgrep",
        "matches": matches,
        "returned": len(matches),
        "files_with_matches": len({match["path"] for match in matches}),
        "truncated": truncated,
    }


def drain_registered_streams(
    selector: selectors.BaseSelector,
    stdout_buffer: bytearray,
    stderr_buffer: bytearray,
) -> None:
    """Read remaining nonblocking bytes after ripgrep exits."""
    for key in list(selector.get_map().values()):
        while True:
            try:
                chunk = os.read(key.fileobj.fileno(), 65_536)
            except BlockingIOError:
                break
            if not chunk:
                selector.unregister(key.fileobj)
                break
            target = stdout_buffer if key.data == "stdout" else stderr_buffer
            limit = (
                MAX_SEARCH_BUFFER_BYTES
                if key.data == "stdout"
                else MAX_SEARCH_ERROR_BYTES
            )
            target.extend(chunk[: max(0, limit - len(target))])


def append_ripgrep_matches(
    raw_line: bytes,
    matches: list[dict[str, object]],
    max_results: int,
) -> None:
    """Append match events from one ripgrep JSON line."""
    if not raw_line:
        return
    try:
        event = json.loads(raw_line)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"invalid JSON from ripgrep: {exc}") from exc
    if event.get("type") != "match":
        return
    data = event.get("data", {})
    path = json_text(data.get("path"))
    raw_text = json_text(data.get("lines"))
    line_number = data.get("line_number")
    if path is None or raw_text is None or line_number is None:
        return
    line = raw_text.rstrip("\r\n")
    path = Path(path).as_posix()
    encoded_line = line.encode("utf-8")
    for submatch in data.get("submatches", []):
        if len(matches) >= max_results:
            return
        byte_column = int(submatch.get("start", 0))
        column = len(encoded_line[:byte_column].decode("utf-8", errors="replace")) + 1
        matches.append(
            {
                "path": path,
                "line": int(line_number),
                "column": column,
                "text": preview_line(line, column - 1),
            }
        )


def json_text(value: object) -> str | None:
    if not isinstance(value, dict):
        return None
    text = value.get("text")
    return text if isinstance(text, str) else None


def search_with_python(
    root: Path,
    base: Path,
    query: str,
    file_pattern: str | None,
    case_sensitive: bool,
    max_results: int,
) -> dict[str, object]:
    """Fixed-string fallback used when ripgrep is unavailable."""
    matches: list[dict[str, object]] = []
    files_with_matches: set[str] = set()
    truncated = False
    flags = 0 if case_sensitive else re.IGNORECASE
    pattern = re.compile(re.escape(query), flags)
    deadline = time.monotonic() + SEARCH_TIMEOUT_SECONDS
    for candidate in iter_search_candidates(base):
        if time.monotonic() >= deadline:
            raise RuntimeError(
                f"search exceeded the {SEARCH_TIMEOUT_SECONDS:g} second timeout"
            )
        try:
            resolved = candidate.resolve()
            relative = resolved.relative_to(root).as_posix()
        except (OSError, ValueError):
            continue
        if not candidate.is_file():
            continue
        pattern_path = candidate.relative_to(base if base.is_dir() else base.parent)
        if file_pattern is not None and not pattern_path.match(file_pattern):
            continue
        try:
            if candidate.stat().st_size > MAX_SEARCH_FILE_BYTES:
                continue
            content = candidate.read_bytes()
            if b"\0" in content:
                continue
            text = content.decode("utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        for line_number, line in enumerate(text.splitlines(), start=1):
            for found in pattern.finditer(line):
                if len(matches) >= max_results:
                    truncated = True
                    break
                files_with_matches.add(relative)
                matches.append(
                    {
                        "path": relative,
                        "line": line_number,
                        "column": found.start() + 1,
                        "text": preview_line(line, found.start()),
                    }
                )
            if truncated:
                break
        if truncated:
            break
    return {
        "engine": "python",
        "matches": matches,
        "returned": len(matches),
        "files_with_matches": len(files_with_matches),
        "truncated": truncated,
    }


def iter_search_candidates(base: Path) -> Iterator[Path]:
    """Yield files deterministically without materializing the whole tree."""
    if base.is_file():
        yield base
        return
    for directory, directory_names, file_names in os.walk(base, followlinks=False):
        directory_names.sort()
        file_names.sort()
        parent = Path(directory)
        for file_name in file_names:
            yield parent / file_name


def preview_line(line: str, match_index: int) -> str:
    if len(line) <= MAX_SEARCH_LINE_CHARS:
        return line
    half = MAX_SEARCH_LINE_CHARS // 2
    start = max(0, min(match_index - half, len(line) - MAX_SEARCH_LINE_CHARS))
    end = start + MAX_SEARCH_LINE_CHARS
    prefix = "…" if start > 0 else ""
    suffix = "…" if end < len(line) else ""
    return f"{prefix}{line[start:end]}{suffix}"
