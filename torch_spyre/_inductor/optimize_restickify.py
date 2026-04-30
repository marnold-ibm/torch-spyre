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
        self.has_no_stick = False  # True for scalar/broadcast args (no real stick)
        self._cost: dict[LayoutKey, dict[LayoutKey, float]] = {}
        self._target: dict[LayoutKey, dict[LayoutKey, Any]] = {}

    def mark_no_stick(self) -> None:
        """Mark this arg as scalar/broadcast — compatible with any output at zero cost."""
        self.has_no_stick = True

    def set_cost_and_target(
        self, in_key: LayoutKey, out_key: LayoutKey, cost: float, target
    ) -> None:
        self._cost.setdefault(in_key, {})[out_key] = cost
        self._target.setdefault(in_key, {})[out_key] = target

    def format_table(self) -> str:
        if self.has_no_stick:
            return "    (no stick — compatible with any output at zero cost)"
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
        if self.has_no_stick:
            return True
        return any(
            out_key in row and row[out_key] < INF for row in self._cost.values()
        )

    def cost(self, in_key: LayoutKey, out_key: LayoutKey) -> float:
        """Cost for in_key -> out_key transition."""
        if self.has_no_stick:
            return 0
        return self._cost.get(in_key, {}).get(out_key, INF)

    def target(self, in_key: LayoutKey, out_key: LayoutKey):
        """Target layout for in_key -> out_key, or None if no restickify needed."""
        if self.has_no_stick:
            return None
        return self._target.get(in_key, {}).get(out_key)


class RestickNodeCost(abc.ABC):

    def __init__(self, edge_costs):
        self.edge_costs = edge_costs

    @abc.abstractmethod
    def cost(self, in_layouts: "list[LayoutKey]", out_key: LayoutKey) -> float: ...


class AllSameNode(RestickNodeCost):

    def cost(self, in_layouts: "list[LayoutKey]", out_key: LayoutKey) -> float:
        return sum(
            rc.cost(lk, out_key) for rc, lk in zip(self.edge_costs, in_layouts)
        )


class FixedInOutNode(RestickNodeCost):

    def __init__(
        self,
        edge_costs,
        required_out_key: LayoutKey,
        required_in_keys: "list[LayoutKey]",
    ):
        super().__init__(edge_costs)
        self.required_out_key = required_out_key
        self.required_in_keys = required_in_keys

    def cost(self, in_layouts: "list[LayoutKey]", out_key: LayoutKey) -> float:
        if out_key != self.required_out_key:
            return INF
        return sum(
            rc.cost(lk, req_key)
            for rc, lk, req_key in zip(
                self.edge_costs, in_layouts, self.required_in_keys
            )
        )


def record_stick_decisions(
    op, chosen_layout, out_key: LayoutKey, in_layouts: "list[LayoutKey]"
) -> None:
    op.stick_decisions = {
        "chosen_layout": chosen_layout,
        "out_key": out_key,
        "arg_in_layouts": in_layouts,
    }


def optimize_restickify_locations(operations: list) -> None:
    # Dumb implemntation for now
    always_choose_first_arg_stick(operations)


def always_choose_first_arg_stick(operations: list) -> None:
    """
        Choose where to put restickiy
        Replicate braindead algorithm of always using first arg's stick
    """

    from torch._inductor.ir import InputBuffer, StorageBox, TensorBox

    print()
    print("=== In Collapse Layouts ===")

    # Commit graph inputs first so all upstreams have committed_layout.
    for name in V.graph.graph_input_names:
        tb = V.graph.graph_inputs[name]
        if (
            isinstance(tb, TensorBox)
            and isinstance(tb.data, StorageBox)
            and isinstance(tb.data.data, InputBuffer)
            and hasattr(tb, "layouts")
        ):
            print(f"MRA input layouts: {name} -> {[list(LayoutKey.from_stl(l.device_layout).stride_map) for l in tb.layouts]}")
            chosen = next(iter(tb.layouts))
            tb.data.data.layout = chosen
            committed = LayoutKey.from_stl(chosen.device_layout)
            tb.data.data.committed_layout = committed
            tb.committed_layout = committed
            print(f"MRA input committed: {name} -> {list(committed.stride_map)}")
            del tb.layouts

    for op in operations:
        if not hasattr(op, "layouts"):
            continue  # FallbackKernel and other unhandled op types

        if not hasattr(op, "restick_cost_fn"):
            # Layout is fixed/inherited — no restickify decision needed.
            # Set chosen_layout so finalize_layouts assigns the tiled layout,
            # but only if the op isn't already a mutation op (those keep their
            # MutationLayoutSHOULDREMOVE for the scheduler to see).
            from torch._inductor.ir import MutationLayoutSHOULDREMOVE
            from torch_spyre._inductor.ir import FixedTiledLayout
            if not isinstance(op.layout, MutationLayoutSHOULDREMOVE):
                op.chosen_layout = op.layouts[0]
                op.committed_layout = LayoutKey.from_stl(op.layouts[0].device_layout)
            continue

        cost_fn = op.restick_cost_fn
        in_layouts = []
        for rc in cost_fn.edge_costs:
            buf = V.graph.get_buffer(rc.dep.name)
            has_cl = hasattr(buf, "committed_layout")
            lk = buf.committed_layout if has_cl else LayoutKey.from_stl(buf.get_layout().device_layout)
            print(f"MRA in_layout: arg={rc.dep.name} has_committed={has_cl} layout={list(lk.stride_map)}")
            in_layouts.append(lk)
        chosen = op.layouts[0]
        out_key = LayoutKey.from_stl(chosen.device_layout)
        cost = cost_fn.cost(in_layouts, out_key)
        print(
            f"MRA select_restickify_locations ({op.get_name()}): "
            f"stick={list(out_key.stride_map)} cost={cost} in_layouts={[list(lk.stride_map) for lk in in_layouts]}"
        )
        op.committed_layout = out_key
        record_stick_decisions(op, chosen, out_key, in_layouts)
