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
import torch

from .constants import MAX_RESTICK_COST as INF

from torch._inductor.virtualized import V


class EdgeCostMap:
    """Thin 2-D cost table for one input arg.

    Cost table indexed [in_iv][out_iv] where both iv indices are in THIS NODE's
    iteration variable namespace — NOT the upstream buffer's namespace.
      iv = numeric suffix N of loop variable dN whose Mod(dN,64) is the stick.
      e.g. Mod(d2,64) -> iv=2.
    This is NOT a tensor dimension index.

    """

    def __init__(self, dep: "MemoryDep", n: int):
        # dep is kept for IR passes (insert_restickify) that need the buffer name
        # and read index.  Cost optimization code does not and should not use it
        self.dep = dep
        self.has_no_stick = False  # True for scalar/broadcast args (no real stick)
        self._cost = [[INF] * n for _ in range(n)]
        self._target: list[list] = [[None] * n for _ in range(n)]

    def mark_no_stick(self) -> None:
        """Mark this arg as scalar/broadcast — compatible with any output at zero cost."""
        self.has_no_stick = True

    def set_cost_and_target(self, in_iv: int, out_iv: int, cost: int, target) -> None:
        """in_iv, out_iv are iteration variable indices (NOT tensor dim indices)."""
        self._cost[in_iv][out_iv] = cost
        self._target[in_iv][out_iv] = target

    def format_table(self) -> str:
        if self.has_no_stick:
            return "    (no stick — compatible with any output at zero cost)"
        lines = []
        for in_iv, (row, trow) in enumerate(zip(self._cost, self._target)):
            for out_iv, (cost, tgt) in enumerate(zip(row, trow)):
                if cost == INF:
                    lines.append(f"    iv{in_iv}->iv{out_iv} = MAX (infeasible)")
                else:
                    lines.append(
                        f"    iv{in_iv}->iv{out_iv} = {cost}"
                        f"  target_stride_map={list(tgt.device_layout.stride_map)}"
                    )
        return "\n".join(lines)

    def feasible_for_out(self, out_iv: int) -> bool:
        if self.has_no_stick:
            return True
        return any(row[out_iv] < INF for row in self._cost)

    def cost(self, in_iv: int, out_iv: int) -> int:
        """Cost for in_iv -> out_iv transition."""
        if self.has_no_stick:
            return 0
        if in_iv >= len(self._cost) or out_iv >= len(self._cost[in_iv]):
            return INF
        return self._cost[in_iv][out_iv]

    def target(self, in_iv: int, out_iv: int):
        """Target layout for in_iv -> out_iv, or None if no restickify needed."""
        if self.has_no_stick or in_iv == out_iv:
            return None
        if in_iv >= len(self._target) or out_iv >= len(self._target[in_iv]):
            return None
        return self._target[in_iv][out_iv]



class RestickNodeCost(abc.ABC):

    def __init__(self, edge_costs):
        self.edge_costs = edge_costs

    @abc.abstractmethod
    def cost_for_out(self, out_iv: int) -> int: ...


class AllSameNode(RestickNodeCost):

    def cost_for_out(self, out_iv: int) -> int:
        in_edge_costs = [rc.cost(out_iv, out_iv) for rc in self.edge_costs]
        return INF if INF in in_edge_costs else sum(in_edge_costs)

    def arg_in_ivs(self, out_iv: int) -> "list[int]":
        return [out_iv] * len(self.edge_costs)


class FixedInOutNode(RestickNodeCost):

    def __init__(self, edge_costs, required_out_iv: int, required_in_iv: "list[int]"):
        super().__init__(edge_costs)
        self.required_out_iv = required_out_iv
        self.required_in_iv = required_in_iv

    def cost_for_out(self, out_iv: int) -> int:
        if out_iv != self.required_out_iv:
            return INF
        in_edge_costs = [
            rc.cost(in_iv, out_iv)
            for rc, in_iv in zip(self.edge_costs, self.required_in_iv)
        ]
        return INF if INF in in_edge_costs else sum(in_edge_costs)

    def arg_in_ivs(self, out_iv: int) -> "list[int]":
        return list(self.required_in_iv)


class PassthroughNode(RestickNodeCost):

    def __init__(self):
        pass

    def cost_for_out(self, out_iv: int) -> int:
        return 0


def record_stick_decisions(op, cost_fn, chosen_layout, out_iv: int) -> None:
    op.stick_decisions = {
        "chosen_layout": chosen_layout,
        "out_iv": out_iv,
        "arg_in_ivs": cost_fn.arg_in_ivs(out_iv),
    }


def optimize_restickify_locations(operations: list) -> None:
    # Dumb implemntation for now
    always_choose_first_arg_stick(operations)


def propagate_passthrough(op) -> "FixedTiledLayout":
    """Pick the best layout for a node with no restick_cost_fn.

    If there is only one candidate, return it directly.  With multiple
    candidates, forward-propagate: choose the layout whose output stick iv
    matches the committed_out_iv of the first input arg.  Fall back to
    layouts[0] if no match is found.
    """
    if len(op.layouts) == 1:
        return op.layouts[0]

    for read_dep in op.get_read_writes().reads:
        buf = V.graph.get_buffer(read_dep.name)
        in_iv = getattr(buf, "committed_out_iv", None)
        if in_iv is None or in_iv == -1:
            continue
        for layout in op.layouts:
            if layout.out_iv == in_iv:
                return layout
        break  # only match against first valid input arg
    return op.layouts[0]


def always_choose_first_arg_stick(operations: list) -> None:
    """
        Choose where to put restickiy
        Replicate braindead algorithm of always using first arg's stick
    """

    print()
    print("=== In Collapse Layouts ===")

    for op in operations:
        if not hasattr(op, "layouts"):
            continue

        if not hasattr(op, "restick_cost_fn"):
            chosen = propagate_passthrough(op)
            print(
                f"MRA select_restickify_locations ({op.get_name()}): "
                f"no cost_fn, picked: {chosen}"
            )
            op.chosen_layout = chosen
            continue

        cost_fn = op.restick_cost_fn
        chosen = op.layouts[0]
        out_iv = chosen.out_iv
        cost = cost_fn.cost_for_out(out_iv)
        print(
            f"MRA select_restickify_locations ({op.get_name()}): "
            f"stick=iv{out_iv} cost={cost} (first-arg layout)"
        )
        record_stick_decisions(op, cost_fn, chosen, out_iv)
