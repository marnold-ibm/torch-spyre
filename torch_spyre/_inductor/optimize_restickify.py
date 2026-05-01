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


import abc
import math
from dataclasses import dataclass
from typing import Any

from torch._inductor.dependencies import MemoryDep
from torch._inductor.ir import (
    InputBuffer,
    MutationLayoutSHOULDREMOVE,
    StorageBox,
    TensorBox,
)
from torch._inductor.virtualized import V
from torch_spyre._C import SpyreTensorLayout

INF = math.inf


@dataclass(frozen=True)
class LayoutKey:
    device_size: tuple[int, ...]
    stride_map: tuple[int, ...]

    @staticmethod
    def from_stl(stl: SpyreTensorLayout) -> "LayoutKey":
        return LayoutKey(tuple(stl.device_size), tuple(stl.stride_map))


class EdgeCostMap:
    """Thin 2-D cost table for one input arg.

    Cost table indexed [in_key][out_key] where both keys are LayoutKeys
    (device_size, stride_map) — stable across all nodes in the graph.
    """

    def __init__(self, dep: "MemoryDep"):
        # dep is kept for IR passes (insert_restickify) that need the buffer name
        # and read index.  Cost optimization code does not and should not use it
        self.dep = dep
        self._cost: dict[LayoutKey, dict[LayoutKey, float]] = {}
        self._target: dict[LayoutKey, dict[LayoutKey, Any]] = {}

    def set_cost_and_target(
        self, in_key: LayoutKey, out_key: LayoutKey, cost: float, target
    ) -> None:
        self._cost.setdefault(in_key, {})[out_key] = cost
        self._target.setdefault(in_key, {})[out_key] = target

    def format_table(self) -> str:
        lines = []
        for in_key, row in self._cost.items():
            trow = self._target.get(in_key, {})
            for out_key, cost in row.items():
                tgt = trow.get(out_key)
                if cost == INF:
                    lines.append(
                        f"    {list(in_key.stride_map)}->{list(out_key.stride_map)} = MAX (infeasible)"
                    )
                else:
                    lines.append(
                        f"    {list(in_key.stride_map)}->{list(out_key.stride_map)} = {cost}"
                        + (
                            f"  target_stride_map={list(tgt.device_layout.stride_map)}"
                            if tgt is not None
                            else ""
                        )
                    )
        return "\n".join(lines)

    def feasible_for_out(self, out_key: LayoutKey) -> bool:
        return any(out_key in row and row[out_key] < INF for row in self._cost.values())

    def cost(self, in_key: LayoutKey, out_key: LayoutKey) -> "float | None":
        """Cost for in_key -> out_key transition. Returns None on cache miss."""
        return self._cost.get(in_key, {}).get(out_key)

    def target(self, in_key: LayoutKey, out_key: LayoutKey):
        """Target layout for in_key -> out_key, or None if no restickify needed."""
        return self._target.get(in_key, {}).get(out_key)


class RestickNodeCost(abc.ABC):
    def __init__(self, edge_costs, args, out_key_by_expr, compute_fn):
        self.edge_costs = edge_costs
        self.args = args
        self.out_key_by_expr = out_key_by_expr
        self._compute_fn = compute_fn

    def _get_edge_cost(
        self, rc: EdgeCostMap, in_key: LayoutKey, out_key: LayoutKey, arg
    ) -> float:
        c = rc.cost(in_key, out_key)
        if c is None:
            self._compute_fn(rc, in_key, arg, self.out_key_by_expr)
            c = rc.cost(in_key, out_key)
        return INF if c is None else c

    @abc.abstractmethod
    def cost(self, in_layouts: "list[LayoutKey]", out_key: LayoutKey) -> float: ...


class AllSameNode(RestickNodeCost):
    def cost(self, in_layouts: "list[LayoutKey]", out_key: LayoutKey) -> float:
        total = 0.0
        for rc, lk, arg in zip(self.edge_costs, in_layouts, self.args):
            c = self._get_edge_cost(rc, lk, out_key, arg)
            if c == INF:
                return INF
            total += c
        return total


class FixedInOutNode(RestickNodeCost):
    def __init__(
        self,
        edge_costs,
        args,
        out_key_by_expr,
        compute_fn,
        required_out_key: LayoutKey,
        required_in_keys: "list[LayoutKey]",
    ):
        super().__init__(edge_costs, args, out_key_by_expr, compute_fn)
        self.required_out_key = required_out_key
        self.required_in_keys = required_in_keys

    def cost(self, in_layouts: "list[LayoutKey]", out_key: LayoutKey) -> float:
        if out_key != self.required_out_key:
            return INF
        total = 0.0
        for rc, lk, req_key, arg in zip(
            self.edge_costs, in_layouts, self.required_in_keys, self.args
        ):
            c = self._get_edge_cost(rc, lk, req_key, arg)
            if c == INF:
                return INF
            total += c
        return total


def record_stick_decisions(
    op, out_key: LayoutKey, in_layouts: "list[LayoutKey]"
) -> None:
    op.stick_decisions = {
        "out_key": out_key,
        "arg_in_layouts": in_layouts,
    }


def optimize_restickify_locations(operations: list) -> None:
    # Dumb implemntation for now
    greedy_local_min_cost(operations)


def _print_op_layouts(operations: list, label: str) -> None:
    print(f"\n=== op layouts [{label}] ===")
    for op in operations:
        layout = op.layout
        layouts = getattr(op, "layouts", None)
        print(
            f"  {op.get_name()}: layout={type(layout).__name__}({layout})"
            + (f" | layouts={[type(lo).__name__ for lo in layouts]}" if layouts else "")
        )


def greedy_local_min_cost(operations: list) -> None:
    """
    Simple baseline
    Greedy algorithm that processes nodes in topological order and finalizes output stick
    However it uses the cost function at that node to pick a min local cost
    If costs equal, choose left-most argument's stick

    This is largely equal to the previous baseline (always choose first arg's stick)
    But this version has the potential to choose a different arg if the first arg's
    is sub-optimal or inviable.
    """

    print()
    print("=== In greedy_local_min_cost ===")
    _print_op_layouts(operations, "before")

    print()
    print("-- Running greedy algorithm --")
    # Process graph inputs first so all upstreams have committed_layout.
    # For now inputs are always a set of size 1, since we use it as it
    # was tranferred to device
    for name in V.graph.graph_input_names:
        tb = V.graph.graph_inputs[name]
        if (
            isinstance(tb, TensorBox)
            and isinstance(tb.data, StorageBox)
            and isinstance(tb.data.data, InputBuffer)
            and hasattr(tb, "layouts")
        ):
            print(
                f"MRA input layouts: {name} -> {[list(LayoutKey.from_stl(lo.device_layout).stride_map) for lo in tb.layouts]}"
            )
            if not tb.layouts:
                raise AssertionError(f"graph input {name} has empty layouts set")
            layout = next(iter(tb.layouts))
            tb.data.data.layout = layout
            tb.data.data.committed_layout = LayoutKey.from_stl(layout.device_layout)
            tb.committed_layout = tb.data.data.committed_layout
            print(
                f"MRA input committed: {name} -> {list(tb.committed_layout.stride_map)}"
            )
            del tb.layouts

    for op in operations:
        print("Processing node:", op.get_name())
        if not hasattr(op, "layouts"):
            continue  # FallbackKernel and other unhandled op types

        if not hasattr(op, "restick_cost_fn"):
            if not isinstance(op.layout, MutationLayoutSHOULDREMOVE):
                # Must set the layout for Mutation Ops
                # TODO should this be done in propagate_layouts?
                op.layout = op.layouts[0]
                op.committed_layout = LayoutKey.from_stl(op.layouts[0].device_layout)

            # nothing to do here.
            continue

        cost_fn = op.restick_cost_fn

        # Collect each input arg's committed layout (finalized by earlier topo iterations).
        in_layouts = []
        for rc in cost_fn.edge_costs:
            buf = V.graph.get_buffer(rc.dep.name)
            assert hasattr(buf, "committed_layout"), (
                f"buffer {rc.dep.name} has no committed_layout — "
                "topological order violated or input not committed"
            )
            in_layouts.append(buf.committed_layout)

        # TODO: Don't out layout keys every time sigh
        out_layout_keys = [LayoutKey.from_stl(ol.device_layout) for ol in op.layouts]
        assert out_layout_keys, (
            f"op {op.get_name()} has restick_cost_fn but no candidate output layouts"
        )
        layout = None
        out_key = None
        best_cost = float("inf")
        for out_layout, out_layout_key in zip(op.layouts, out_layout_keys):
            out_layout_cost = cost_fn.cost(in_layouts, out_layout_key)
            print(
                f"MRA candidate ({op.get_name()}): "
                f"stick={list(out_layout_key.stride_map)} cost={out_layout_cost}"
            )
            if out_layout_cost < best_cost:
                best_cost = out_layout_cost
                layout = out_layout
                out_key = out_layout_key

        assert out_key is not None, (
            f"({op.get_name()}): all stick possibilities had infinite cost. Cannot proceed"
        )

        print(
            f"MRA select_restickify_locations ({op.get_name()}): "
            f"stick={list(out_key.stride_map)} cost={best_cost} "
            f"in_layouts={[list(lk.stride_map) for lk in in_layouts]}"
        )
        if not isinstance(op.layout, MutationLayoutSHOULDREMOVE):
            op.layout = layout
        op.committed_layout = out_key
        record_stick_decisions(op, out_key, in_layouts)

    _print_op_layouts(operations, "after")
