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

import os
from math import prod
from typing import Optional

import sympy
import torch
from torch._inductor.ir import (
    ComputedBuffer,
    FixedLayout,
    InputBuffer,
    Operation,
    Pointwise,
    Reduction,
    StorageBox,
    TensorBox,
)
from torch._inductor.dependencies import MemoryDep
from torch._inductor.virtualized import V

from torch_spyre._C import SpyreTensorLayout

from .constants import BATCH_MATMUL_OP, MATMUL_REDUCTION_OP
from .errors import Unsupported
from .ir import FixedTiledLayout
from .logging_utils import get_inductor_logger
from .pass_utils import device_coordinates, host_coordinates
from .views import matching_dim

logger = get_inductor_logger("plan_restickify")

# full_state: op_name -> winning input buffer name (str) or None if no conflict.
#   Kernel-independent — used as guidance in the second stickify pass.
# pruned_state: buf_name -> FixedTiledLayout (the chosen output layout for an intermediate
#   buffer). Used for conflict detection in downstream ops: call device_coordinates(layout, dep)
#   with the consumer's dep to get the correct consumer-namespace stick expr.
#   Pruned after last use.
FullState = dict[str, Optional[str]]
PrunedState = dict[str, Optional[FixedTiledLayout]]
FrontierEntry = tuple[FullState, PrunedState, int]
Frontier = list[FrontierEntry]

BEAM_WIDTH = 128

# Set by plan_restickify after each compilation — readable from tests.
last_frontier: Frontier = []


def convert_input_layouts(operations: list[Operation]) -> None:
    """Convert graph InputBuffers from FixedLayout to FixedTiledLayout.

    Must run before propagate_spyre_tensor_layouts, which requires all buffers
    (including graph inputs) to have FixedTiledLayout.
    """
    if len(V.graph.graph_input_names) > 0:
        for name, real_input in zip(V.graph.graph_input_names, V.get_real_inputs()):
            if isinstance(real_input, torch.Tensor):
                stl = real_input.device_tensor_layout()
                if stl is None:
                    raise Unsupported(
                        f"missing device_tensor_layout on graph input {name}"
                    )
                tb = V.graph.graph_inputs[name]
                if (
                    not isinstance(tb, TensorBox)
                    or not isinstance(tb.data, StorageBox)
                    or not isinstance(tb.data.data, InputBuffer)
                ):
                    raise Unsupported(
                        f"graph input {name} is not a TensorBox(StorageBox(InputBuffer))"
                    )
                ptl = tb.data.data.layout
                if not isinstance(ptl, FixedLayout):
                    raise Unsupported(f"graph input {name} does not have a FixedLayout")
                tb.data.data.layout = FixedTiledLayout(
                    ptl.device, ptl.dtype, ptl.size, ptl.stride, stl
                )


def _buf_stick(
    dep: MemoryDep,
    layout: FixedTiledLayout,
) -> Optional[sympy.Expr]:
    """Return stick_expr (idc[-1]) for a buffer in dep's kernel namespace.

    Returns None for scalar buffers (idc[-1] == 0).
    """
    idc = device_coordinates(layout, dep)
    stick_expr = idc[-1]
    return None if stick_expr == 0 else stick_expr


def _buf_layout(name: str, pruned: dict) -> Optional[FixedTiledLayout]:
    """Return the planned layout for a buffer if in pruned_state, else first-pass layout."""
    planned = pruned.get(name)
    if planned is not None:
        return planned
    buf = V.graph.get_buffer(name)
    if buf is None:
        return None
    layout = buf.get_layout()
    return layout if isinstance(layout, FixedTiledLayout) else None


def _pointwise_output_layout(
    op: ComputedBuffer,
    chosen_expr: sympy.Expr,
) -> Optional[FixedTiledLayout]:
    """Compute the output FixedTiledLayout for a pointwise op given a chosen stick expr.

    Simulates pointwise_layout's dim_order path: builds SpyreTensorLayout with the
    chosen stick dimension last. Used to propagate correct layouts in the planner.
    """
    rw = op.get_read_writes()
    out_dep = next((d for d in rw.writes if isinstance(d, MemoryDep)), None)
    if out_dep is None:
        return None
    layout = op.get_layout()
    if not isinstance(layout, FixedLayout):
        return None
    # host_coordinates only needs layout.size and layout.stride — both available
    # on FixedLayout (the output's pre-device-assignment layout).
    out_coords = host_coordinates(layout, out_dep)
    out_stick_dim = matching_dim(out_coords, chosen_expr)
    if out_stick_dim is None:
        out_stick_dim = -1
    dim_order = [
        d for d in range(len(layout.size)) if d != out_stick_dim and out_coords[d] != 0
    ]
    dim_order += [
        d for d in range(len(layout.size)) if d != out_stick_dim and out_coords[d] == 0
    ]
    dim_order += [out_stick_dim]
    stl = SpyreTensorLayout(layout.size, layout.stride, layout.dtype, dim_order)
    return FixedTiledLayout(layout.device, layout.dtype, layout.size, layout.stride, stl)


def analyze_stick_conflicts(operations: list[Operation], K: int = BEAM_WIDTH) -> Frontier:
    """Beam search over stick dimension choices to minimise total restickify cost.

    Runs after a first pass of propagate_spyre_tensor_layouts, so every buffer
    already has a FixedTiledLayout with physical device coordinates.

    For each ComputedBuffer:
    - Pointwise: beam-search over stick choices, branching at conflicts.
    - Matmul/bmm: forced restickify cost if inputs don't satisfy hardware
      constraints; output stick is deterministic (generated dim). No branching.
    - Other non-pointwise: passthrough — output stick read from layout, no cost.

    State design:
    - full_state  maps op_name -> winning input buffer name (str or None).
      Kernel-independent — returned as guidance to the second stickify pass.
    - pruned_state maps buf_name -> FixedTiledLayout (planned output layout).
      Used for conflict detection in downstream ops via device_coordinates(layout, dep).
      Pruned after last use.

    Returns frontier sorted by cost ascending.
    """
    # Precompute last-use index for each buffer so stale pruned_state entries
    # can be dropped after their last consumer is processed.
    last_use: dict[str, int] = {}
    for i, op in enumerate(operations):
        if not isinstance(op, ComputedBuffer):
            continue
        rw = op.get_read_writes()
        for dep in rw.reads:
            if isinstance(dep, MemoryDep):
                last_use[dep.name] = i

    frontier: Frontier = [({}, {}, 0)]
    _beam_pruned_warned = False

    for i, op in enumerate(operations):
        if not isinstance(op, ComputedBuffer):
            continue

        if isinstance(op.data, Pointwise):
            rw = op.get_read_writes()

            # Collect (dep, elems) for each tiled input.
            raw_args = []
            for dep in rw.reads:
                if not isinstance(dep, MemoryDep):
                    continue
                buf = V.graph.get_buffer(dep.name)
                if buf is None:
                    continue
                fp_layout = buf.get_layout()
                if not isinstance(fp_layout, FixedTiledLayout):
                    continue
                elems = prod(int(s) for s in fp_layout.size)
                raw_args.append((dep, elems))

            new_frontier: Frontier = []
            for full, pruned, cost in frontier:
                resolved = []
                for dep, elems in raw_args:
                    layout = _buf_layout(dep.name, pruned)
                    se = _buf_stick(dep, layout) if layout is not None else None
                    resolved.append((dep.name, se, elems))

                candidate_exprs = list({se for _, se, _ in resolved if se is not None})

                if len(candidate_exprs) <= 1:
                    chosen_expr = candidate_exprs[0] if candidate_exprs else None
                    out_layout = _pointwise_output_layout(op, chosen_expr) if chosen_expr is not None else None
                    new_frontier.append((
                        {**full, op.get_name(): None},
                        {**pruned, op.get_name(): out_layout},
                        cost,
                    ))
                else:
                    for chosen_expr in candidate_exprs:
                        restickify_cost = sum(
                            elems
                            for _, se, elems in resolved
                            if se is not None and se != chosen_expr
                        )
                        winning_name = next(
                            (name for name, se, _ in resolved if se == chosen_expr),
                            None,
                        )
                        out_layout = _pointwise_output_layout(op, chosen_expr)
                        new_frontier.append((
                            {**full, op.get_name(): winning_name},
                            {**pruned, op.get_name(): out_layout},
                            cost + restickify_cost,
                        ))

            frontier = new_frontier

        elif (
            isinstance(op.data, Reduction)
            and op.data.reduction_type in (MATMUL_REDUCTION_OP, BATCH_MATMUL_OP)
        ):
            rw = op.get_read_writes()
            reads = [d for d in rw.reads if isinstance(d, MemoryDep)]
            x_dep, y_dep = reads[0], reads[1]
            out_dep = next(d for d in rw.writes if isinstance(d, MemoryDep))

            x_buf_size = prod(int(s) for s in V.graph.get_buffer(x_dep.name).get_layout().size)
            y_buf_size = prod(int(s) for s in V.graph.get_buffer(y_dep.name).get_layout().size)
            out_coords = host_coordinates(op.get_layout(), out_dep)

            new_frontier = []
            for full, pruned, cost in frontier:
                x_layout = _buf_layout(x_dep.name, pruned)
                y_layout = _buf_layout(y_dep.name, pruned)
                x_stick = _buf_stick(x_dep, x_layout) if x_layout else None
                y_stick = _buf_stick(y_dep, y_layout) if y_layout else None

                # x: stick must be on reduction_dim — NOT in out_coords.
                # y: stick must be on generated_dim — IS in out_coords.
                x_needs = x_stick is not None and matching_dim(out_coords, x_stick) is not None
                y_needs = y_stick is not None and matching_dim(out_coords, y_stick) is None

                forced_cost = (x_buf_size if x_needs else 0) + (y_buf_size if y_needs else 0)

                # Output stick is always on generated_dim — use y's stick after any restickify.
                y_stick_after = y_stick if not y_needs else _buf_stick(
                    y_dep, _buf_layout(y_dep.name, {})
                )
                out_layout_tiled = _pointwise_output_layout(op, y_stick_after) if y_stick_after else None

                new_frontier.append((
                    {**full, op.get_name(): None},
                    {**pruned, op.get_name(): out_layout_tiled},
                    cost + forced_cost,
                ))

            frontier = new_frontier

        # Sort, beam-prune, then drop stale pruned_state entries.
        frontier.sort(key=lambda x: x[2])
        if len(frontier) > K and not _beam_pruned_warned:
            _beam_pruned_warned = True
            logger.warning(
                f"plan_restickify: beam pruned from {len(frontier)} to {K} "
                f"at {op.get_name()} — consider increasing BEAM_WIDTH"
            )
        frontier = frontier[:K]
        frontier = [
            (full, {k: v for k, v in pruned.items() if last_use.get(k, -1) > i}, cost)
            for full, pruned, cost in frontier
        ]

    return frontier


def plan_restickify(operations: list[Operation]) -> Optional[FullState]:
    """Plan optimal restickify placement.

    Runs after a first pass of propagate_spyre_tensor_layouts (which assigns
    FixedTiledLayout to all buffers).  Uses physical device coordinates and
    host dim indices — no sympy namespace bridging required.

    Returns the best FullState (op name -> chosen stick expr) for guidance,
    or None if the graph has no pointwise ops.
    """

    frontier = analyze_stick_conflicts(operations)
    if os.getenv("SPYRE_CAPTURE_RESTICKIFY_PLAN"):
        global last_frontier
        last_frontier = frontier
    if not frontier:
        return None
    print(f"[plan_restickify] top {min(3, len(frontier))} of {len(frontier)} states:")
    for rank, (full, pruned, cost) in enumerate(frontier[:3]):
        print(f"  [{rank}] cost={cost}")
        for op in operations:
            if not isinstance(op, ComputedBuffer) or not isinstance(op.data, Pointwise):
                continue
            rw = op.get_read_writes()
            winning_name = full.get(op.get_name())
            if winning_name is None:
                continue
            # Print which inputs get restickified (those that aren't the winner).
            winning_layout = pruned.get(op.get_name())
            for dep in rw.reads:
                if not isinstance(dep, MemoryDep) or dep.name == winning_name:
                    continue
                in_layout = _buf_layout(dep.name, pruned)
                in_stick = _buf_stick(dep, in_layout) if in_layout else None
                win_stick = _buf_stick(dep, winning_layout) if winning_layout else None
                if in_stick is not None:
                    buf = V.graph.get_buffer(dep.name)
                    elems = prod(int(s) for s in buf.get_layout().size) if buf else "?"
                    print(f"    restickify {dep.name}({in_stick} -> {win_stick}) before {op.get_name()} [{elems} elems]")
    best_full, _pruned, _best_cost = frontier[0]
    return best_full if any(v is not None for v in best_full.values()) else None
