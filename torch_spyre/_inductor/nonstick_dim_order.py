# Copyright 2025 The Torch-Spyre Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Reorder non-stick device dimensions for better work division.

Runs between propagate_layouts and optimize_restickify. Walks the graph
backward visiting ops that have opinions about their inputs' non-stick
layout. For each such op, computes and writes replacement candidate STL
sets onto its input buffers so the stick optimizer sees the right shapes.

Currently handles MATMUL_REDUCTION_OPS with two transforms:
  - flat-M projection: collapse higher-rank views to canonical 2-D [M,K]
  - dim reorder: swap largest outer dim into the sandwich slot for parallelism
"""

import math
from typing import TYPE_CHECKING

import sympy
from torch._inductor.dependencies import MemoryDep
from torch._inductor.graph import GraphLowering
from torch._inductor.ir import ComputedBuffer, Reduction
from torch._inductor.virtualized import V
from torch_spyre._C import DataFormats, ElementArrangement, SpyreTensorLayout

if TYPE_CHECKING:
    from torch._inductor.ir import FixedLayout, Operation

    from .propagate_layouts import PropArg

from .constants import MATMUL_REDUCTION_OPS
from .errors import Unsupported
from .logging_utils import get_inductor_logger
from .pass_utils import (
    concretize_expr,
    find_reduction_var,
    get_matmul_m_size,
    get_matmul_n_size,
    try_device_coordinates,
)

logger = get_inductor_logger("nonstick_dim_order")


def _flat_dense_projection_x_layout(
    x: "PropArg",
    y: "PropArg",
    output: "FixedLayout",
    output_dep: MemoryDep,
    reduction_var: "sympy.Symbol",
    m_size: int,
    n_size: int,
) -> "SpyreTensorLayout | None":
    """Return a canonical flat-M layout for a logically 2-D dense projection.

    A fused attention producer can retain a higher-rank contiguous host view
    such as ``[B, L, H, D]`` even though the projection reads it as the logical
    matrix ``[B*L, H*D]``.  Preserving those physical outer axes makes the
    backend encode the shared-weight projection as a BMM, which is much slower
    than the equivalent flat MM.  Collapse only when the complete access is
    provably one dense row-major 2-D matrix and the weight is shared (rank 2).

    Genuine BMMs retain a rank-3 output and/or a batched weight and therefore do
    not enter this path.
    """
    if (
        m_size <= 1
        or n_size <= 1
        or len(output.size) != 2
        or len(x.layout.size) <= 2
        or len(y.layout.size) != 2
        or not x.layouts
        or x.layouts[0].element_arrangement != ElementArrangement.STANDARD
        or x.layouts[0].device_dtype != DataFormats.SEN169_FP16
    ):
        return None

    x_size = [concretize_expr(s) for s in x.layout.size]
    x_stride = [concretize_expr(s) for s in x.layout.stride]
    y_size = [concretize_expr(s) for s in y.layout.size]
    out_size = [concretize_expr(s) for s in output.size]
    out_stride = [concretize_expr(s) for s in output.stride]

    def is_dense_contiguous(size: list[int], stride: list[int]) -> bool:
        expected = 1
        for dim_size, dim_stride in zip(reversed(size), reversed(stride)):
            if dim_size != 1 and dim_stride != expected:
                return False
            expected *= dim_size
        return True

    if not is_dense_contiguous(x_size, x_stride) or not is_dense_contiguous(
        out_size, out_stride
    ):
        return None

    active_x_vars = set(x.dep.index.free_symbols) & set(x.dep.ranges)
    row_vars = active_x_vars - {reduction_var}
    if reduction_var not in active_x_vars or len(row_vars) != 1:
        return None
    (row_var,) = row_vars

    row_size = concretize_expr(x.dep.ranges[row_var])
    reduction_size = concretize_expr(x.dep.ranges[reduction_var])
    active_out_vars = set(output_dep.index.free_symbols) & set(output_dep.ranges)
    generated_vars = active_out_vars - {row_var}
    if row_var not in active_out_vars or len(generated_vars) != 1:
        return None
    (generated_var,) = generated_vars
    generated_size = concretize_expr(output_dep.ranges[generated_var])

    if (
        row_size != m_size
        or generated_size != n_size
        or math.prod(x_size) != row_size * reduction_size
        or math.prod(y_size) != reduction_size * generated_size
        or math.prod(out_size) != m_size * n_size
        or row_var in y.dep.index.free_symbols
        or {reduction_var, generated_var}
        != (set(y.dep.index.free_symbols) & set(y.dep.ranges))
    ):
        return None

    expected_index = reduction_size * row_var + reduction_var
    expected_output_index = generated_size * row_var + generated_var
    if (
        sympy.simplify(x.dep.index - expected_index) != 0
        or sympy.simplify(output_dep.index - expected_output_index) != 0
    ):
        return None

    return SpyreTensorLayout(
        [row_size, reduction_size],
        [reduction_size, 1],
        x.layout.dtype,
        [0, 1],
        ElementArrangement.STANDARD,
    )


def _reorder_stl(
    stl: SpyreTensorLayout,
    dep: MemoryDep,
    name: str = "",
) -> SpyreTensorLayout:
    """Swap the largest non-stick dim into the slot between the two stick dims.

    A factorised stick produces two dims that share the same loop variable:
    floor(d/64) at position outer_stick and Mod(d/64) at the last position.
    This function moves the largest remaining dim into outer_stick+1 — the
    slot between them — so the compiler assigns the most iterations to the
    widest loop variable.
    """
    # Non-STANDARD element arrangements have hardware-defined dimension
    # semantics; reordering them corrupts the DDL template matching.
    if stl.element_arrangement != ElementArrangement.STANDARD:
        return stl
    device_size = list(stl.device_size)
    stride_map = list(stl.stride_map)
    n = len(device_size)
    if n <= 2:
        return stl

    idc = try_device_coordinates(stl, dep, {})
    if idc is None:
        return stl

    # Find the stick variable from the last dim's coordinate.
    stick_syms = idc[-1].free_symbols
    if not stick_syms:
        # Degenerate/broadcast stick (constant 0): nothing to do.
        return stl

    # Find the outer stick dim: the non-last dim that shares the stick variable.
    outer_stick = None
    for i in range(n - 2, -1, -1):
        if idc[i].free_symbols & stick_syms:
            outer_stick = i
            break
    if outer_stick is None:
        return stl  # unsplit stick, no slot to fill

    slot = outer_stick + 1
    logger.debug(
        "nonstick_dim_order: %s idc=%s outer_stick=%d slot=%d n=%d",
        name,
        [str(x) for x in idc],
        outer_stick,
        slot,
        n,
    )
    if slot >= n - 1:
        logger.debug(
            "nonstick_dim_order: skipping %s — no room between stick dims"
            " (outer_stick=%d, n=%d)",
            name,
            outer_stick,
            n,
        )
        return stl

    # Only move dims from outside (before outer_stick) into the slot,
    # and only if the largest outside dim is bigger than what's already there.
    # Exclude dims with constant (zero free-symbol) coordinates — these are
    # padding/gap dims prepended by restickify/compact and must not be moved.
    candidates = [d for d in range(outer_stick) if idc[d].free_symbols]
    if not candidates:
        return stl
    largest = max(candidates, key=lambda d: device_size[d])
    if device_size[largest] <= device_size[slot]:
        return stl  # already optimal or nothing to gain

    # Swap largest into slot.
    new_order = list(range(n))
    new_order[slot], new_order[largest] = new_order[largest], new_order[slot]
    new_device_size = [device_size[d] for d in new_order]
    new_stride_map = [stride_map[d] for d in new_order]
    return SpyreTensorLayout(
        device_size=new_device_size,
        stride_map=new_stride_map,
        device_dtype=stl.device_dtype,
    )


def _compute_nonstick_layouts(
    buf: ComputedBuffer,
    x_dep: MemoryDep,
    op: "Operation",
) -> "list[SpyreTensorLayout] | None":
    """Return replacement candidate STLs for buf, or None if no change is needed.

    Called once per (buf, op) pair. Tries flat-M projection first; if that
    does not apply, tries the generic dim reorder. Returns None if neither
    transform changes anything.
    """
    from .propagate_layouts import (
        PropArg,
    )  # local to avoid circular import at module level  # noqa: F401

    if not buf.layouts:
        return None

    reads = [r for r in op.get_read_writes().reads if isinstance(r, MemoryDep)]
    y_deps = [r for r in reads if r.name != x_dep.name]
    out_deps = list(op.get_read_writes().writes)

    if len(y_deps) >= 1 and len(out_deps) >= 1:
        y_dep = y_deps[0]
        out_dep = out_deps[0]
        y_buf = V.graph.get_buffer(y_dep.name)
        out_buf = V.graph.get_buffer(op.get_name())
        if hasattr(y_buf, "layouts") and y_buf.layouts:
            x_prop = PropArg(x_dep, buf.get_layout(), list(buf.layouts))
            y_prop = PropArg(y_dep, y_buf.get_layout(), list(y_buf.layouts))
            out_host = out_buf.get_layout()
            try:
                reduction_var = find_reduction_var((x_dep,), out_dep)
                m_size = get_matmul_m_size(op)
                n_size = get_matmul_n_size(op)
                flat_stl = _flat_dense_projection_x_layout(
                    x_prop, y_prop, out_host, out_dep, reduction_var, m_size, n_size
                )
                if flat_stl is not None:
                    logger.info(
                        "nonstick_dim_order: flat-M projection on %s"
                        " — replacing %d candidate(s) with %s",
                        x_dep.name,
                        len(buf.layouts),
                        list(flat_stl.device_size),
                    )
                    return [flat_stl]
            except Unsupported:
                pass

    write_dep = next(iter(buf.get_read_writes().writes), None)
    if write_dep is None:
        return None
    new_layouts = [_reorder_stl(stl, write_dep, x_dep.name) for stl in buf.layouts]
    changed = any(
        list(a.device_size) != list(b.device_size)
        for a, b in zip(buf.layouts, new_layouts)
    )
    if not changed:
        return None
    for i, (old, new) in enumerate(zip(buf.layouts, new_layouts)):
        if list(old.device_size) != list(new.device_size):
            logger.debug(
                "[NDO] %s[%d]  %s -> %s  stride_map %s -> %s",
                x_dep.name,
                i,
                list(old.device_size),
                list(new.device_size),
                list(old.stride_map),
                list(new.stride_map),
            )
    return new_layouts


def reorder_nonstick_dims(graph: GraphLowering) -> None:
    """Reorder non-stick dims on op inputs for better work division.

    Walks the graph backward. For each op that has opinions about its inputs'
    non-stick layout, computes and writes replacement candidate STL sets onto
    those input buffers. Graph inputs (weights, activations passed in directly)
    are skipped — their layouts are fixed by the caller.
    """
    log: dict[str, list] = {}
    graph_inputs = set(V.graph.graph_input_names)

    for op in reversed(graph.operations):
        if not isinstance(getattr(op, "data", None), Reduction):
            continue
        if op.data.reduction_type not in MATMUL_REDUCTION_OPS:
            continue

        for dep in op.get_read_writes().reads:
            if not isinstance(dep, MemoryDep):
                continue
            if dep.name in graph_inputs:
                continue
            buf = V.graph.get_buffer(dep.name)
            if not isinstance(buf, ComputedBuffer) or not hasattr(buf, "layouts"):
                logger.debug(
                    "nonstick_dim_order: skipping %s — %s",
                    dep.name,
                    "not a ComputedBuffer"
                    if not isinstance(buf, ComputedBuffer)
                    else "no .layouts attribute",
                )
                continue

            new_layouts = _compute_nonstick_layouts(buf, dep, op)
            if new_layouts is not None:
                buf.layouts[:] = new_layouts
                log[dep.name] = list(new_layouts)
                logger.info(
                    "nonstick_dim_order: reordered %s (%d candidates)",
                    dep.name,
                    len(new_layouts),
                )

    V.graph.nonstick_reorder_log = log
