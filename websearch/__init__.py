"""Provider-neutral web search interfaces and implementations."""

from .base import SearchProvider, SearchResponse, SearchResult
from .searxng import SearxngSearchProvider

__all__ = ["SearchProvider", "SearchResponse", "SearchResult", "SearxngSearchProvider"]
