#!/usr/bin/env python3
"""Parse SPYRE_INDEX_FN log lines and summarize unique index expressions.

Usage:
    grep "^SPYRE_INDEX_FN" run_*.log | python3 tools/parse_index_fns.py
    python3 tools/parse_index_fns.py all_index_fns.txt
"""
from __future__ import annotations

import ast
import sys
from collections import Counter, defaultdict

import sympy


def parse_line(line: str) -> tuple[str, str, sympy.Expr, list[int]] | None:
    """Parse one SPYRE_INDEX_FN line.

    Format: SPYRE_INDEX_FN <buf> <read|write> <pretty> | <srepr> | <ranges>
    Returns (buf_name, direction, expr, ranges) or None if unparseable.
    """
    line = line.strip()
    if not line.startswith("SPYRE_INDEX_FN "):
        return None
    rest = line[len("SPYRE_INDEX_FN "):]
    parts = rest.split(None, 2)  # buf, direction, remainder
    if len(parts) < 3:
        return None
    buf, direction, remainder = parts
    fields = remainder.split(" | ")
    if len(fields) < 2:
        return None
    ranges: list[int] = []
    try:
        ranges = ast.literal_eval(fields[1])
    except Exception:
        pass
    srepr_str = fields[2] if len(fields) >= 3 else fields[1]
    try:
        expr = sympy.sympify(srepr_str)
    except Exception:
        return None
    return buf, direction, expr, ranges


def main() -> None:
    if len(sys.argv) > 1:
        with open(sys.argv[1]) as f:
            lines = f.readlines()
    else:
        lines = sys.stdin.readlines()

    counts: Counter[str] = Counter()
    exprs: dict[str, sympy.Expr] = {}
    ranges_seen: dict[str, list[list[int]]] = defaultdict(list)

    for line in lines:
        result = parse_line(line)
        if result is None:
            continue
        _buf, _direction, expr, ranges = result
        key = sympy.srepr(expr)
        counts[key] += 1
        exprs[key] = expr
        if ranges and ranges not in ranges_seen[key]:
            ranges_seen[key].append(ranges)

    for key in ranges_seen:
        ranges_seen[key].sort(key=lambda r: (len(r), r))

    print(f"Unique index expressions: {len(counts)}")
    print()
    for key, count in counts.most_common():
        expr = exprs[key]
        range_strs = ", ".join(str(r) for r in ranges_seen[key])
        print(f"  count={count:4d}  {expr}    ranges={range_strs}")


if __name__ == "__main__":
    main()
