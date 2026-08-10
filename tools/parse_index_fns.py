#!/usr/bin/env python3
"""Parse SPYRE_INDEX_FN log lines and summarize unique index expressions.

Usage:
    grep "^SPYRE_INDEX_FN" run_*.log | python3 tools/parse_index_fns.py
    python3 tools/parse_index_fns.py all_index_fns.txt
"""
from __future__ import annotations

import sys
from collections import Counter

import sympy


def parse_line(line: str) -> tuple[str, str, sympy.Expr] | None:
    """Parse one SPYRE_INDEX_FN line.

    Format: SPYRE_INDEX_FN <buf> <read|write> <pretty> | <srepr>
    Returns (buf_name, direction, expr) or None if unparseable.
    """
    line = line.strip()
    if not line.startswith("SPYRE_INDEX_FN "):
        return None
    rest = line[len("SPYRE_INDEX_FN "):]
    parts = rest.split(None, 2)  # buf, direction, remainder
    if len(parts) < 3:
        return None
    buf, direction, remainder = parts
    if " | " not in remainder:
        return None
    _, srepr_str = remainder.split(" | ", 1)
    try:
        expr = sympy.sympify(srepr_str)
    except Exception:
        return None
    return buf, direction, expr


def main() -> None:
    if len(sys.argv) > 1:
        with open(sys.argv[1]) as f:
            lines = f.readlines()
    else:
        lines = sys.stdin.readlines()

    counts: Counter[str] = Counter()
    exprs: dict[str, sympy.Expr] = {}

    for line in lines:
        result = parse_line(line)
        if result is None:
            continue
        _buf, _direction, expr = result
        key = sympy.srepr(expr)
        counts[key] += 1
        exprs[key] = expr

    print(f"Unique index expressions: {len(counts)}")
    print()
    for key, count in counts.most_common():
        expr = exprs[key]
        print(f"  count={count:4d}  {expr}")


if __name__ == "__main__":
    main()
