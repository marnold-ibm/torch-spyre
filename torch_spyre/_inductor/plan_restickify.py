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

from .constants import BATCH_MATMUL_OP, MATMUL_REDUCTION_OP
from .errors import Unsupported
from .ir import FixedTiledLayout
from .logging_utils import get_inductor_logger
from .pass_utils import host_coordinates, device_coordinates
from .views import matching_dim

logger = get_inductor_logger("plan_restickify")

# A state maps buffer name -> chosen host stick dim index (int), or None for no-stick/scalar.
State = dict[str, Optional[int]]
# Each frontier entry carries:
#   full_state  — never pruned; the complete choice history for this path (used as guidance)
#   pruned_state — entries dropped after last use; used for lookups during beam search
#   cost        — cumulative restickify cost
FrontierEntry = tuple[State, State, int]
Frontier = list[FrontierEntry]

BEAM_WIDTH = 128

# Set by plan_restickify after each compilation — readable from tests.
last_frontier: Frontier = []


def convert_input_layouts(operations: list[Operation]) -> None:
    """Convert graph InputBuffers from FixedLayout to FixedTiledLayout."""
    if len(V.graph.graph_input_names) > 0:
        for name, real_input in zip(V.graph.graph_input_names, V.get_real_inputs()):
            if isinstance(real_input, torch.Tensor):
                stl = real_input.device_tensor_layout()
                if stl is None:
                    # All spyre tensors are created with device layouts.
                    # Therefore we expect all graph inputs to have them.
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
                        "graph input {name} is not a TensorBox(StorageBox(InputBuffer))"
                    )
                ptl = tb.data.data.layout
                if not isinstance(ptl, FixedLayout):
                    raise Unsupported("graph input {name} does not have a FixedLayout")
                tb.data.data.layout = FixedTiledLayout(
                    ptl.device, ptl.dtype, ptl.size, ptl.stride, stl
                )


def _mem_deps(rw) -> list[MemoryDep]:
    return [d for d in rw.reads if isinstance(d, MemoryDep)]


def _buf_elems(name: str) -> int:
    buf = V.graph.get_buffer(name)
    return prod(int(s) for s in buf.get_layout().size) if buf is not None else 0


def _input_stick_dim(dep: MemoryDep) -> Optional[int]:
    """Return the host stick dim index for a graph input (has FixedTiledLayout).

    Returns None for intermediate buffers whose layout is not yet assigned.
    """
    buf = V.graph.get_buffer(dep.name)
    if buf is None or not isinstance(buf.get_layout(), FixedTiledLayout):
        return None
    layout = buf.get_layout()
    in_coords = host_coordinates(layout, dep)
    dev_coords = device_coordinates(layout, dep)
    return matching_dim(in_coords, dev_coords[-1])


def _get_stick_dim(dep: MemoryDep, pruned: State) -> Optional[int]:
    """Return the stick dim for dep: from propagated state if available, else from input layout."""
    if dep.name in pruned:
        return pruned[dep.name]
    return _input_stick_dim(dep)


def analyze_stick_conflicts(operations: list[Operation], K: int = BEAM_WIDTH) -> Frontier:
    """Beam search over stick dimension choices to minimise total restickify cost.

    For each ComputedBuffer:
    - Pointwise: beam-search over stick choices, branching at conflicts.
    - Matmul/bmm: forced restickify cost if inputs don't satisfy hardware constraints;
      output stick is deterministic (generated dim). No branching.
    - Other non-pointwise: passthrough — output stick recorded from dep, no cost.

    Each frontier entry is (full_state, pruned_state, cost):
    - full_state: never pruned; complete choice history for this path, used as guidance.
    - pruned_state: entries dropped after last use; used for lookups during beam search.
    - cost: cumulative restickify cost.

    Returns frontier sorted by cost ascending.
    """
    # Precompute last-use index for each buffer so stale pruned_state entries can be dropped.
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

    print(f"[plan] analyze_stick_conflicts: {len(operations)} operations")
    for i, op in enumerate(operations):
        print(f"[plan] visiting {op.get_name()} type={type(op).__name__} data={type(getattr(op, 'data', None)).__name__}")
        if not isinstance(op, ComputedBuffer):
            continue

        rw = op.get_read_writes()
        out_dep = next(iter(rw.writes), None)

        # For each read dep: compute host coords and elem count once (independent of frontier state).
        reads = []
        for dep in _mem_deps(rw):
            buf = V.graph.get_buffer(dep.name)
            in_coords = host_coordinates(buf.get_layout(), dep)
            elems = _buf_elems(dep.name)
            stick_dim = _input_stick_dim(dep)
            print(f"  [plan]   arg={dep.name} in_coords={in_coords} stick_dim={stick_dim}")
            reads.append((dep, in_coords, elems))

        if isinstance(op.data, Pointwise):
            print(f"[plan] op={op.get_name()} reads={[type(d).__name__ for d in rw.reads]}")

            new_frontier: Frontier = []
            for full, pruned, cost in frontier:
                resolved = [
                    (dep, _get_stick_dim(dep, pruned), in_coords, elems)
                    for dep, in_coords, elems in reads
                ]

                # Conflict detection: two inputs conflict if their stick coord expressions differ.
                # Use in_coords[stick_dim] to get the expression for comparison.
                # Map stick coord expr -> dim index; distinct exprs = distinct stick choices.
                candidates = {
                    ic[sd]: sd
                    for _, sd, ic, _ in resolved
                    if sd is not None
                }
                print(f"  [plan]   candidates={candidates}")

                if len(candidates) <= 1:
                    out_stick = next(iter(candidates.values()), None)
                    print(f"  [plan]   no conflict, out_stick={out_stick} total={cost}")
                    new_frontier.append((
                        {**full, op.get_name(): out_stick},
                        {**pruned, op.get_name(): out_stick},
                        cost,
                    ))
                else:
                    for chosen_expr, chosen_dim in candidates.items():
                        restickify_cost = sum(
                            elems
                            for _, sd, ic, elems in resolved
                            if sd is not None and ic[sd] != chosen_expr
                        )
                        total = cost + restickify_cost
                        print(f"  [plan]   choice=dim{chosen_dim} restickify_cost={restickify_cost} total={total}")
                        new_frontier.append((
                            {**full, op.get_name(): chosen_dim},
                            {**pruned, op.get_name(): chosen_dim},
                            total,
                        ))

            new_frontier.sort(key=lambda x: x[2])
            if len(new_frontier) > K and not _beam_pruned_warned:
                _beam_pruned_warned = True
                logger.warning(
                    f"plan_restickify: beam pruned from {len(new_frontier)} to {K} at {op.get_name()} — consider increasing BEAM_WIDTH"
                )
            frontier = new_frontier[:K]
            print(f"  [plan]   frontier size={len(frontier)}, costs={[c for _,_,c in frontier]}")

        elif (
            isinstance(op.data, Reduction)
            and op.data.reduction_type in (MATMUL_REDUCTION_OP, BATCH_MATMUL_OP)
        ):
            (x_dep, x_coords, _), (y_dep, y_coords, _) = reads[0], reads[1]

            # x stick must be on the reduction dim: the x host dim that doesn't appear in out_dep.ranges.
            x_reduction_dim = next(
                (j for j, c in enumerate(x_coords)
                 if len(c.free_symbols) > 0 and c.free_symbols.isdisjoint(out_dep.ranges)),
                None,
            )
            # y stick (and output stick) must be on the generated dim: the y host dim whose
            # loop var appears in out_dep.ranges but not in x_dep.ranges.
            y_generated_dim = next(
                (j for j, c in enumerate(y_coords)
                 if len(c.free_symbols) > 0
                 and not c.free_symbols.isdisjoint(out_dep.ranges)
                 and c.free_symbols.isdisjoint(x_dep.ranges)),
                None,
            )
            print(
                f"[plan] matmul {op.get_name()} x_reduction_dim={x_reduction_dim} y_generated_dim={y_generated_dim}"
            )

            new_frontier = []
            for full, pruned, cost in frontier:
                x_planned = _get_stick_dim(x_dep, pruned)
                y_planned = _get_stick_dim(y_dep, pruned)

                x_needs_restickify = x_planned != x_reduction_dim
                y_needs_restickify = y_planned != y_generated_dim

                forced_cost = (
                    (_buf_elems(x_dep.name) if x_needs_restickify else 0)
                    + (_buf_elems(y_dep.name) if y_needs_restickify else 0)
                )
                print(
                    f"  [plan]   x_stick=dim{x_planned} y_stick=dim{y_planned} forced_cost={forced_cost} total={cost + forced_cost}"
                )
                new_frontier.append((
                    {**full, op.get_name(): y_generated_dim},
                    {**pruned, op.get_name(): y_generated_dim},
                    cost + forced_cost,
                ))

            frontier = sorted(new_frontier, key=lambda x: x[2])

        else:
            # Other non-pointwise ops: layout not yet assigned, propagate None.
            print(f"[plan] passthrough {op.get_name()}")
            frontier = [
                (
                    {**full, op.get_name(): None},
                    {**pruned, op.get_name(): None},
                    cost,
                )
                for full, pruned, cost in frontier
            ]

        # Prune pruned_state entries no longer needed downstream.
        # full_state is never pruned — it carries the complete choice history.
        frontier = [
            (full, {k: v for k, v in pruned.items() if last_use.get(k, -1) > i}, cost)
            for full, pruned, cost in frontier
        ]

    return frontier


def plan_restickify(operations: list[Operation]) -> Optional[State]:
    """Pre-stickify pass: plan optimal restickify placement.

    Runs before propagate_spyre_tensor_layouts. After convert_input_layouts(),
    graph InputBuffers have FixedTiledLayout; intermediate node layouts are not
    yet assigned.

    Returns the best State (op name -> chosen stick var) or None if the graph
    has no pointwise ops.
    """
    convert_input_layouts(operations)
    frontier = analyze_stick_conflicts(operations)
    if os.getenv("SPYRE_CAPTURE_RESTICKIFY_PLAN"):
        global last_frontier
        last_frontier = frontier
    if not frontier:
        return None
    best_full, _pruned, best_cost = frontier[0]
    # Reconstruct where restickifies would be inserted for the best state.
    print(f"\n[plan] final frontier ({len(frontier)} states):")
    for rank, (full, _, cost) in enumerate(frontier):
        print(f"  [{rank}] cost={cost} state={full}")
    restickifies = []
    for op in operations:
        if not isinstance(op, ComputedBuffer) or not isinstance(op.data, Pointwise):
            continue
        rw = op.get_read_writes()
        op_stick = best_full.get(op.get_name())
        for dep in rw.reads:
            if not isinstance(dep, MemoryDep):
                continue
            sv = best_full.get(dep.name) if dep.name in best_full else _input_stick_dim(dep)
            if sv is None:
                continue
            if op_stick is not None and sv != op_stick:
                restickifies.append(f"{dep.name}->{op.get_name()} (sv={sv}, chosen={op_stick})")
    print(f"  [plan] best state: total_cost={best_cost}, restickifies={len(restickifies)}")
    for r in restickifies:
        print(f"    restickify: {r}")
    print(f"[plan] guidance: {best_full}")
    return best_full if best_full else None
