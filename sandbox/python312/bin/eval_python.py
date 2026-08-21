#!/usr/bin/env python3
"""Evaluate one Python expression or a short result-producing snippet."""

from __future__ import annotations

import math
import sys


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: eval_python.py EXPRESSION", file=sys.stderr)
        return 2
    try:
        namespace = {
            "__builtins__": __builtins__,
            "math": math,
            **{
                name: getattr(math, name)
                for name in dir(math)
                if not name.startswith("_")
            },
        }
        source = sys.argv[1]
        try:
            expression = compile(source, "<tool-expression>", "eval")
        except SyntaxError:
            statements = compile(source, "<tool-statements>", "exec")
            exec(statements, namespace)
            if "result" not in namespace:
                raise ValueError(
                    "statement snippets must assign the value to 'result'"
                )
            result = namespace["result"]
        else:
            result = eval(expression, namespace)
    except BaseException as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    if result is not None:
        print(repr(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
