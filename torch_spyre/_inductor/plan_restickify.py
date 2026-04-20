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

# State is a tuple of chosen host stick dim indices, parallel to buf_names.
State = tuple[Optional[int], ...]
# Each frontier entry is (state, restickifies, cost).
FrontierEntry = tuple[State, list[str], int]

BEAM_WIDTH = 64

class Frontier:
    def __init__(self, K: int = BEAM_WIDTH):
        self.K = K
        self.buf_names: list[str] = []
        self._buf_idx: dict[str, int] = {}
        self.entries: list[FrontierEntry] = [((), [], 0)]
        self._staged: list[FrontierEntry] = []
        self._beam_pruned_warned = False
        self._live: set[str] = set()
        self.max_live: int = 0

    def add_buf(self, op: ComputedBuffer) -> None:
        name = op.get_name()
        self._buf_idx[name] = len(self.buf_names)
        self.buf_names.append(name)
        self._live.add(name)
        self.max_live = max(self.max_live, len(self._live))

    def append(
        self,
        entry: FrontierEntry,
        stick_dim: Optional[int],
        need_restick: list[str] = [],
        extra_cost: int = 0,
    ) -> None:
        state, restickifies, cost = entry
        self._staged.append(
            (state + (stick_dim,), restickifies + need_restick, cost + extra_cost)
        )

    def trim(self, op_name: str) -> None:
        self._staged.sort(key=lambda e: e[2])
        if len(self._staged) > self.K and not self._beam_pruned_warned:
            self._beam_pruned_warned = True
            logger.warning(
                f"plan_restickify: beam pruned from {len(self._staged)} to {self.K}"
                f" at {op_name} — consider increasing BEAM_WIDTH"
            )
        self.entries = self._staged[: self.K]
        self._staged = []

    def best(self) -> FrontierEntry:
        return self.entries[0]

    def stick_dim(self, dep: MemoryDep, state: State) -> Optional[int]:
        """Return the host stick dim for a dep's buffer.

        For graph inputs (FixedTiledLayout): derive from device_coordinates.
        For intermediates: look up from propagated beam state.
        """
        buf = V.graph.get_buffer(dep.name)
        if buf is not None and isinstance(buf.get_layout(), FixedTiledLayout):
            layout = buf.get_layout()
            in_coords = host_coordinates(layout, dep)
            dev_coords = device_coordinates(layout, dep)
            return matching_dim(in_coords, dev_coords[-1])
        idx = self._buf_idx.get(dep.name, -1)
        return state[idx] if idx >= 0 else None

    def mark_dead(self, names: list[str]) -> None:
        for name in names:
            self._live.discard(name)

    def guidance(self) -> dict[str, Optional[int]]:
        best_state, _, _ = self.best()
        return dict(zip(self.buf_names, best_state))


# Set by plan_restickify after each compilation — readable from tests.
last_frontier: Frontier


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



def analyze_stick_conflicts(operations: list[Operation], K: int = BEAM_WIDTH) -> Frontier:
    """Beam search over stick dimension choices to minimise total restickify cost.

    For each ComputedBuffer:
    - Pointwise: beam-search over stick choices, branching at conflicts.
    - Matmul/bmm: forced restickify cost if inputs don't satisfy hardware constraints;
      output stick is deterministic (generated dim). No branching.
    - Other non-pointwise: passthrough — output stick recorded from dep, no cost.

    Returns Frontier sorted by cost ascending.
    """
    last_use: dict[str, int] = {}
    for i, op in enumerate(operations):
        if not isinstance(op, ComputedBuffer):
            continue
        for dep in _mem_deps(op.get_read_writes()):
            last_use[dep.name] = i

    frontier = Frontier(K)

    print(f"[plan] analyze_stick_conflicts: {len(operations)} operations")
    for i, op in enumerate(operations):
        print(f"[plan] visiting {op.get_name()} type={type(op).__name__} data={type(getattr(op, 'data', None)).__name__}")
        if not isinstance(op, ComputedBuffer):
            continue

        rw = op.get_read_writes()
        out_dep = next(iter(rw.writes), None)

        # Compute host coords and elem count per read dep once, independent of frontier state.
        reads = []
        for dep in _mem_deps(rw):
            buf = V.graph.get_buffer(dep.name)
            in_coords = host_coordinates(buf.get_layout(), dep)
            elems = _buf_elems(dep.name)
            layout = buf.get_layout()
            input_stick_dim = matching_dim(in_coords, device_coordinates(layout, dep)[-1]) if isinstance(layout, FixedTiledLayout) else None
            print(f"  [plan]   arg={dep.name} in_coords={in_coords} input_stick_dim={input_stick_dim}")
            reads.append((dep, in_coords, elems))

        frontier.add_buf(op)
        out_coords = host_coordinates(op.get_layout(), out_dep)

        if isinstance(op.data, Pointwise):
            for entry in frontier.entries:
                state, _, _ = entry
                stick_exprs = {
                    in_coords[sd]
                    for dep, in_coords, elems in reads
                    if (sd := frontier.stick_dim(dep, state)) is not None
                }
                print(f"  [plan]   candidates={stick_exprs}")

                if len(stick_exprs) <= 1:
                    out_stick = matching_dim(out_coords, next(iter(stick_exprs), None))
                    print(f"  [plan]   no conflict, out_stick={out_stick}")
                    frontier.append(entry, out_stick)
                else:
                    for stick_expr in stick_exprs:
                        needs_restick = []
                        restickify_cost = 0
                        for dep, in_coords, elems in reads:
                            sd = frontier.stick_dim(dep, state)
                            if sd is not None and in_coords[sd] != stick_expr:
                                needs_restick.append(f"{dep.name}->{op.get_name()}")
                                restickify_cost += elems
                        out_stick_dim = matching_dim(out_coords, stick_expr)
                        print(f"  [plan]   choice={stick_expr} out_stick_dim={out_stick_dim} cost={restickify_cost}")
                        frontier.append(entry, out_stick_dim, needs_restick, restickify_cost)

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

            for entry in frontier.entries:
                state, _, _ = entry
                x_planned = frontier.stick_dim(x_dep, state)
                y_planned = frontier.stick_dim(y_dep, state)
                needs_restick = (
                    ([f"{x_dep.name}->{op.get_name()}"] if x_planned != x_reduction_dim else [])
                    + ([f"{y_dep.name}->{op.get_name()}"] if y_planned != y_generated_dim else [])
                )
                forced_cost = (
                    (_buf_elems(x_dep.name) if x_planned != x_reduction_dim else 0)
                    + (_buf_elems(y_dep.name) if y_planned != y_generated_dim else 0)
                )
                print(f"  [plan]   x_stick=dim{x_planned} y_stick=dim{y_planned} forced_cost={forced_cost}")
                frontier.append(entry, y_generated_dim, needs_restick, forced_cost)

        else:
            # Other non-pointwise ops: propagate stick dim from first read dep.
            print(f"[plan] passthrough {op.get_name()}")
            for entry in frontier.entries:
                state, _, _ = entry
                dep, in_coords, _ = reads[0]
                sd = frontier.stick_dim(dep, state)
                out_stick = matching_dim(out_coords, in_coords[sd]) if sd is not None else None
                frontier.append(entry, out_stick)

        frontier.trim(op.get_name())
        frontier.mark_dead([name for name, lu in last_use.items() if lu == i])

        print()
        print(f"[plan] frontier after {op.get_name()}:")
        print(f"  Buf names: {frontier.buf_names}")
        for rank, (state, restickifies, cost) in enumerate(frontier.entries):
            print(f"  [{rank}] cost={cost} restickifies={restickifies} state={state}")
        print()

    return frontier


def plan_restickify(operations: list[Operation]) -> Optional[dict[str, Optional[int]]]:
    """Pre-stickify pass: plan optimal restickify placement.

    Runs before propagate_spyre_tensor_layouts. After convert_input_layouts(),
    graph InputBuffers have FixedTiledLayout; intermediate node layouts are not
    yet assigned.

    Returns the best guidance (op name -> chosen stick dim) or None if the graph
    has no pointwise ops.
    """
    convert_input_layouts(operations)
    frontier = analyze_stick_conflicts(operations)
    if os.getenv("SPYRE_CAPTURE_RESTICKIFY_PLAN"):
        global last_frontier
        last_frontier = frontier
    if not frontier.entries:
        return None
    best_state, best_restickifies, best_cost = frontier.best()
    guidance = frontier.guidance()
    print(f"\n[plan] final frontier ({len(frontier.entries)} states):")
    for rank, (state, restickifies, cost) in enumerate(frontier.entries):
        print(f"  [{rank}] cost={cost} restickifies={restickifies} state={dict(zip(frontier.buf_names, state))}")
    print(f"  [plan] best: total_cost={best_cost}, restickifies={best_restickifies}")
    print(f"  [plan] max_live_bufs={frontier.max_live}")
    print()
    print(f"[plan] guidance: {guidance}")
    return guidance if guidance else None
