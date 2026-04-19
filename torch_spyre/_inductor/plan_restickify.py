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
from .pass_utils import host_coordinates

logger = get_inductor_logger("plan_restickify")

# A state maps buffer name -> chosen stick variable (or None for no-stick/scalar).
State = dict[str, Optional[sympy.Symbol]]
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


def _stick_var(dep: MemoryDep) -> Optional[sympy.Symbol]:
    """Return the loop variable with coefficient 1 in dep.index (the fastest-moving variable), or None."""
    index = sympy.expand(dep.index)
    for var in dep.ranges:
        if sympy.diff(index, var) == 1:
            return var
    return None


def _mem_deps(rw) -> list[MemoryDep]:
    return [d for d in rw.reads if isinstance(d, MemoryDep)]


def _buf_elems(name: str) -> int:
    buf = V.graph.get_buffer(name)
    return prod(int(s) for s in buf.get_layout().size) if buf is not None else 0


def _planned_stick(
    dep: MemoryDep, pruned: State
) -> Optional[sympy.Symbol]:
    return pruned[dep.name] if dep.name in pruned else _stick_var(dep)


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
        args = [(dep, e) for dep in _mem_deps(rw) if (e := _buf_elems(dep.name)) > 0]
        out_dep = next(iter(rw.writes), None)

        if isinstance(op.data, Pointwise):
            print(f"[plan] op={op.get_name()} reads={[type(d).__name__ for d in rw.reads]}")
            for dep, elems in args:
                print(f"  [plan]   arg={dep.name} dep.index={dep.index} elems={elems}")

            new_frontier: Frontier = []
            for full, pruned, cost in frontier:
                resolved = [
                    (dep.name, _planned_stick(dep, pruned), elems)
                    for dep, elems in args
                ]
                candidate_vars = list({sv for _, sv, _ in resolved if sv is not None})
                print(f"  [plan]   candidate_vars={candidate_vars}")

                if len(candidate_vars) <= 1:
                    out_stick = candidate_vars[0] if candidate_vars else None
                    print(f"  [plan]   no conflict, out_stick={out_stick} total={cost}")
                    new_frontier.append((
                        {**full, op.get_name(): out_stick},
                        {**pruned, op.get_name(): out_stick},
                        cost,
                    ))
                else:
                    for chosen_var in candidate_vars:
                        restickify_cost = sum(
                            elems
                            for _, sv, elems in resolved
                            if sv is not None and sv != chosen_var
                        )
                        total = cost + restickify_cost
                        print(f"  [plan]   choice={chosen_var} restickify_cost={restickify_cost} total={total}")
                        new_frontier.append((
                            {**full, op.get_name(): chosen_var},
                            {**pruned, op.get_name(): chosen_var},
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
            x_dep, y_dep = args[0][0], args[1][0]

            # reduction_var: in x.ranges but not in out.ranges (the K dim)
            # generated_var: last variable in out.ranges (the N/last-col dim)
            reduction_var = next(
                (v for v in x_dep.ranges if v not in out_dep.ranges), None
            )
            generated_var = next(reversed(list(out_dep.ranges.keys())), None)
            print(
                f"[plan] matmul {op.get_name()} reduction_var={reduction_var} generated_var={generated_var}"
            )

            def _col_var_of_buf(name: str) -> Optional[sympy.Symbol]:
                """Return the stride-1 variable in this buffer's write dep (producer namespace).

                Used to compare against pruned[name] without crossing kernel namespaces.
                Returns None for graph inputs (not ComputedBuffers).
                """
                buf = V.graph.get_buffer(name)
                if not isinstance(buf, ComputedBuffer):
                    return None
                buf_rw = buf.get_read_writes()
                out_d = next(iter(buf_rw.writes), None)
                return _stick_var(out_d) if out_d is not None else None

            x_col_var = _col_var_of_buf(x_dep.name)
            y_col_var = _col_var_of_buf(y_dep.name)

            new_frontier = []
            for full, pruned, cost in frontier:
                x_planned = _planned_stick(x_dep, pruned)
                y_planned = _planned_stick(y_dep, pruned)

                # Determine whether each input needs a forced restickify.
                #
                # The challenge: pruned[name] is a sympy Symbol from the *producer*
                # kernel's namespace, while reduction_var/generated_var are from the
                # *consumer* (matmul) kernel's namespace.  Direct cross-kernel
                # comparison is wrong — the symbols are different objects even when
                # they represent the same logical dimension.
                #
                # _col_var_of_buf(name) bridges this by re-deriving the stride-1
                # variable from the producer's own write dep (same namespace as
                # pruned[name]).  Comparing pruned[name] against _col_var_of_buf(name)
                # stays within one kernel's namespace and is sound.
                #
                # For graph inputs (col_var is None), _stick_var(dep) and
                # reduction_var/generated_var are both computed from the matmul dep
                # (consumer namespace), so direct comparison is valid.
                if x_col_var is not None:
                    x_needs_restickify = x_planned != x_col_var
                else:
                    x_needs_restickify = reduction_var is not None and x_planned != reduction_var

                if y_col_var is not None:
                    y_needs_restickify = y_planned != y_col_var
                else:
                    y_needs_restickify = generated_var is not None and y_planned != generated_var

                forced_cost = (
                    (_buf_elems(x_dep.name) if x_needs_restickify else 0)
                    + (_buf_elems(y_dep.name) if y_needs_restickify else 0)
                )

                print(
                    f"  [plan]   x_stick={x_planned} y_stick={y_planned} forced_cost={forced_cost} total={cost + forced_cost}"
                )
                new_frontier.append((
                    {**full, op.get_name(): generated_var},
                    {**pruned, op.get_name(): generated_var},
                    cost + forced_cost,
                ))

            frontier = sorted(new_frontier, key=lambda x: x[2])

        else:
            # Other non-pointwise ops: output stick is fixed by op semantics, no cost.
            out_stick = _stick_var(out_dep) if out_dep is not None else None
            print(f"[plan] passthrough {op.get_name()} out_stick={out_stick}")
            frontier = [
                (
                    {**full, op.get_name(): out_stick},
                    {**pruned, op.get_name(): out_stick},
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
    restickifies = []
    for op in operations:
        if not isinstance(op, ComputedBuffer) or not isinstance(op.data, Pointwise):
            continue
        rw = op.get_read_writes()
        op_stick = best_full.get(op.get_name())
        for dep in rw.reads:
            if not isinstance(dep, MemoryDep):
                continue
            sv = best_full.get(dep.name) if dep.name in best_full else _stick_var(dep)
            if sv is None:
                continue
            if op_stick is not None and sv != op_stick:
                restickifies.append(f"{dep.name}->{op.get_name()} (sv={sv}, chosen={op_stick})")
    print(f"  [plan] best state: total_cost={best_cost}, restickifies={len(restickifies)}")
    for r in restickifies:
        print(f"    restickify: {r}")
    print(f"[plan] guidance: {best_full}")
    return best_full if best_full else None
