#!/usr/bin/env python3
"""Parse SPYRE_INDEX_FN log lines and summarize unique index expressions by group.

Usage:
    grep "^SPYRE_INDEX_FN" run_*.log | python3 tools/parse_index_fns.py
    python3 tools/parse_index_fns.py all_index_fns.txt
"""
from __future__ import annotations

import ast
import sys
from collections import Counter, defaultdict

import sympy


# ---------------------------------------------------------------------------
# Classification
#
# _classify(expr, var_ranges) is called per (expr, var_ranges) pair.
# Add new groups inside _classify — first match wins, "other" catches the rest.
#
# var_ranges: dict mapping symbol name -> range, e.g. {"d0": 32, "d1": 64}
# Symbols are already squeezed (no size-1 dims) and include reduction dims.
#
# Helpers:
#   _coeffs(expr)             -> list[int] | None  (None = symbolic strides)
#   _is_linear(expr)          -> bool
#   _has_tmp(expr)            -> bool
#   _expected_strides(ranges) -> list[int] row-major strides for given ranges
# ---------------------------------------------------------------------------

def _coeffs(expr: sympy.Expr) -> list[int] | None:
    syms = sorted(expr.free_symbols, key=str)
    result = []
    for s in syms:
        c = expr.coeff(s)
        if not c.is_number:
            return None  # symbolic stride — dynamic shape
        result.append(int(c))
    return result


def _is_linear(expr: sympy.Expr) -> bool:
    return not expr.has(sympy.Mod) and not expr.has(sympy.floor)


def _has_tmp(expr: sympy.Expr) -> bool:
    return any(str(s).startswith("tmp") for s in expr.free_symbols)


def _row_major_strides(ranges: list[int]) -> list[int]:
    strides = [1] * len(ranges)
    for i in range(len(ranges) - 2, -1, -1):
        strides[i] = strides[i + 1] * ranges[i + 1]
    return strides


def _col_major_strides(ranges: list[int]) -> list[int]:
    strides = [1] * len(ranges)
    for i in range(1, len(ranges)):
        strides[i] = strides[i - 1] * ranges[i - 1]
    return strides


def _is_permuted_row_major(check_coeffs: list[int], ranges: list[int]) -> bool:
    """True if check_coeffs are the strides of some contiguous permutation of ranges.

    Tries all orderings of the ranges, computes row-major strides for each,
    and checks if any permuted stride assignment matches check_coeffs.
    """
    from itertools import permutations
    n = len(ranges)
    for perm in permutations(range(n)):
        # perm[i] = which range index goes to memory position i
        mem_order_ranges = [ranges[perm[i]] for i in range(n)]
        mem_strides = _row_major_strides(mem_order_ranges)
        # assign strides back to original symbol positions
        assigned = [0] * n
        for mem_pos, orig_pos in enumerate(perm):
            assigned[orig_pos] = mem_strides[mem_pos]
        if assigned == check_coeffs:
            return True
    return False


def _classify(expr: sympy.Expr, var_ranges: dict[str, int]) -> str:
    # --- ADD NEW GROUPS HERE ---
    if not expr.free_symbols:
        return "scalar"

    if not _is_linear(expr):
        return "non-linear"

    if _has_tmp(expr):
        return "indirect"

    coeffs = _coeffs(expr)

    if coeffs is None:
        return "symbolic"

    if coeffs == [1]:
        return "single-dim"

    # Strip constant offset; classify the pure part
    const = int(expr.as_coeff_add()[0])
    pure_coeffs = _coeffs(expr - const) if const != 0 else coeffs

    if pure_coeffs is None:
        return "symbolic"

    check_coeffs = pure_coeffs

    # Use var_ranges to get ordered ranges matching the symbols (d0, d1, ...)
    syms = sorted(expr.free_symbols, key=str)
    sym_names = [str(s) for s in syms]
    all_syms_present = var_ranges and all(n in var_ranges for n in sym_names)
    is_broadcast = all_syms_present and set(sym_names) < set(var_ranges.keys())

    if all_syms_present:
        ranges = [var_ranges[n] for n in sym_names]
        expected = _row_major_strides(ranges)
        col_major = _col_major_strides(ranges)

        if check_coeffs == expected:
            return "row-major+missing-dim" if is_broadcast else "row-major"
        if check_coeffs == col_major:
            return "col-major+missing-dim" if is_broadcast else "col-major"
        if _is_permuted_row_major(check_coeffs, ranges):
            return "permuted+missing-dim" if is_broadcast else "permuted"

    # Fallback (no var_ranges or dim mismatch): strictly decreasing, last = 1
    if check_coeffs[-1] == 1 and all(
        check_coeffs[i] > check_coeffs[i + 1] for i in range(len(check_coeffs) - 1)
    ):
        return "row-major"

    # --- END OF GROUPS ---
    return "other"


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def parse_line(line: str) -> tuple[str, str, sympy.Expr, dict[str, int]] | None:
    """Parse one SPYRE_INDEX_FN line.

    Format: SPYRE_INDEX_FN <buf> <read|write> <pretty> | <var_ranges> | <srepr>
    Returns (buf_name, direction, expr, var_ranges) or None if unparseable.
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
    var_ranges: dict[str, int] = {}
    try:
        var_ranges = ast.literal_eval(fields[1])
    except Exception:
        pass
    srepr_str = fields[2] if len(fields) >= 3 else fields[1]
    try:
        expr = sympy.sympify(srepr_str)
    except Exception:
        return None
    return buf, direction, expr, var_ranges


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    if len(sys.argv) > 1:
        with open(sys.argv[1]) as f:
            lines = f.readlines()
    else:
        lines = sys.stdin.readlines()

    # Key is (expr_srepr, var_ranges_tuple) — classify per pair
    counts: Counter[tuple] = Counter()
    exprs: dict[str, sympy.Expr] = {}
    groups: dict[tuple, str] = {}

    for line in lines:
        result = parse_line(line)
        if result is None:
            continue
        _buf, _direction, expr, var_ranges = result
        expr_key = sympy.srepr(expr)
        pair_key = (expr_key, tuple(sorted(var_ranges.items())))
        counts[pair_key] += 1
        exprs[expr_key] = expr
        if pair_key not in groups:
            groups[pair_key] = _classify(expr, var_ranges)

    group_order = [
        "scalar", "symbolic", "non-linear", "indirect",
        "single-dim",
        "row-major", "row-major+missing-dim",
        "col-major", "col-major+missing-dim",
        "permuted", "permuted+missing-dim",
        "other",
    ]

    by_group: dict[str, list[tuple]] = defaultdict(list)
    for pair_key, group in groups.items():
        by_group[group].append(pair_key)

    for group in group_order:
        pairs = by_group.get(group, [])
        if not pairs:
            continue
        unique = len(pairs)
        total = sum(counts[p] for p in pairs)
        print(f"=== {group} ({unique} unique, {total} total) ===")
        for pair_key in sorted(pairs, key=lambda p: -counts[p]):
            expr_key, vr_tuple = pair_key
            expr = exprs[expr_key]
            print(f"  count={counts[pair_key]:4d}  {expr}    var_ranges={dict(vr_tuple)}")
        print()


if __name__ == "__main__":
    main()
