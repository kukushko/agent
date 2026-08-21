"""Generic prompt policy derived from tool-owned reliability metadata."""

from __future__ import annotations

from collections.abc import Iterable


def render_reliability_turn_guidance(
    roles: Iterable[str], guidance: Iterable[str]
) -> str:
    """Build a concise reminder from opaque tool-owned metadata."""
    available = sorted(set(roles))
    instructions = sorted(set(guidance))
    if not available and not instructions:
        return ""
    header = (
        "Automatic reliability guidance: before answering, apply relevant "
        f"available tool roles ({', '.join(available)})."
    )
    details = " ".join(instructions)
    fallback = (
        " If an optional tool attempt fails, fall back to your best knowledge and "
        "identify material assumptions. Do not use tools that cannot meaningfully "
        "improve this request."
    )
    return f"{header} {details}{fallback}"
