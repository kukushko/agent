#!/usr/bin/env python3
"""Check Python source syntax without writing bytecode."""

from __future__ import annotations

import sys
import tokenize


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: check_python.py PATH", file=sys.stderr)
        return 2
    path = sys.argv[1]
    try:
        with tokenize.open(path) as source_file:
            source = source_file.read()
        compile(source, path, "exec")
    except (OSError, SyntaxError, UnicodeError) as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(f"Syntax OK: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
