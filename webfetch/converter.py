"""HTML-to-Markdown conversion using maintained third-party libraries."""

from __future__ import annotations

from dataclasses import dataclass

import trafilatura
from bs4 import BeautifulSoup
from markdownify import markdownify


@dataclass(frozen=True)
class ConversionResult:
    """Normalized converted content and converter diagnostics."""

    markdown: str
    title: str | None
    converter: str
    fallback_used: bool


def html_to_markdown(html: str, source_url: str) -> ConversionResult:
    """Extract useful HTML content as Markdown with a whole-document fallback."""
    title = extract_title(html)
    markdown = trafilatura.extract(
        html,
        url=source_url,
        output_format="markdown",
        include_comments=False,
        include_tables=True,
        include_links=True,
        include_images=False,
        include_formatting=True,
        favor_recall=True,
    )
    if markdown and markdown.strip():
        return ConversionResult(markdown.strip(), title, "trafilatura", False)
    fallback = markdownify(
        html,
        heading_style="ATX",
        bullets="-",
        strip=["script", "style", "noscript", "template"],
    ).strip()
    if not fallback:
        raise RuntimeError("page contains no extractable text")
    return ConversionResult(fallback, title, "markdownify", True)


def extract_title(html: str) -> str | None:
    """Return a bounded document title without implementing another extractor."""
    soup = BeautifulSoup(html, "html.parser")
    if soup.title is None:
        return None
    title = soup.title.get_text(" ", strip=True)
    return title[:500] if title else None
