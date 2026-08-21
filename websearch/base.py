"""Provider-neutral types for web search."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class SearchResult:
    """One normalized result returned by a search provider."""

    title: str
    url: str
    snippet: str
    source: str | None = None
    published_at: str | None = None


@dataclass(frozen=True)
class SearchResponse:
    """Normalized results and provider diagnostics for one query."""

    results: tuple[SearchResult, ...]
    warnings: tuple[str, ...] = ()


class SearchProvider(Protocol):
    """Backend interface consumed by the web tool package."""

    def search(
        self,
        query: str,
        max_results: int,
        language: str | None,
        time_range: str | None,
    ) -> SearchResponse:
        """Search the web and return provider-neutral results."""
