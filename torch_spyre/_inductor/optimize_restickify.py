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
from torch._inductor.virtualized import V

INF = math.inf


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
        # Translates from the upstream buffer's output IV namespace into this
        # node's in_iv row index.  Indexed by upstream IV; value is the
        # corresponding local in_iv; None = unmapped.
        self.upstream_out_iv_to_local_in_iv: "list[int | None]" = [None] * n

    def mark_no_stick(self) -> None:
        """Mark this arg as scalar/broadcast — compatible with any output at zero cost."""
        self.has_no_stick = True

    def set_cost_and_target(self, in_iv: int, out_iv: int, cost: float, target) -> None:
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

    def local_in_iv(self, upstream_iv: int) -> int:
        """Translate upstream committed IV to this node's local in_iv."""
        assert not self.has_no_stick
        result = self.upstream_out_iv_to_local_in_iv[upstream_iv]
        assert result is not None, (
            f"upstream IV {upstream_iv} not in map {self.upstream_out_iv_to_local_in_iv}"
        )
        return result

    def cost(self, in_iv: int, out_iv: int) -> float:
        """Cost for local in_iv -> out_iv transition."""
        if self.has_no_stick:
            return 0
        return self._cost[in_iv][out_iv]

    def target(self, in_iv: int, out_iv: int):
        """Target layout for in_iv -> out_iv, or None if no restickify needed."""
        if self.has_no_stick or in_iv == out_iv:
            return None
        return self._target[in_iv][out_iv]


class RestickNodeCost(abc.ABC):

    def __init__(self, edge_costs):
        self.edge_costs = edge_costs

    @abc.abstractmethod
    def cost(self, upstream_ivs: "list[int]", out_iv: int) -> float: ...


class AllSameNode(RestickNodeCost):

    def cost(self, upstream_ivs: "list[int]", out_iv: int) -> float:
        return sum(
            rc.cost(rc.local_in_iv(uiv), out_iv)
            for rc, uiv in zip(self.edge_costs, upstream_ivs)
        )


class FixedInOutNode(RestickNodeCost):

    def __init__(self, edge_costs, required_out_iv: int, required_in_iv: "list[int]"):
        super().__init__(edge_costs)
        self.required_out_iv = required_out_iv
        self.required_in_iv = required_in_iv

    def cost(self, upstream_ivs: "list[int]", out_iv: int) -> float:
        if out_iv != self.required_out_iv:
            return INF
        return sum(
            rc.cost(rc.local_in_iv(uiv), req_iv)
            for rc, uiv, req_iv in zip(self.edge_costs, upstream_ivs, self.required_in_iv)
        )


def record_stick_decisions(op, chosen_layout, out_iv: int, upstream_ivs: "list[int]") -> None:
    op.stick_decisions = {
        "chosen_layout": chosen_layout,
        "out_iv": out_iv,
        "arg_upstream_ivs": upstream_ivs,
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

    # Commit graph inputs first so all upstreams have committed_out_iv.
    for name in V.graph.graph_input_names:
        tb = V.graph.graph_inputs[name]
        if (
            isinstance(tb, TensorBox)
            and isinstance(tb.data, StorageBox)
            and isinstance(tb.data.data, InputBuffer)
            and hasattr(tb, "layouts")
        ):
            chosen = next(iter(tb.layouts))
            tb.data.data.layout = chosen
            # get_buffer() returns the TensorBox for graph inputs, so set
            # committed_out_iv on both the TensorBox and inner InputBuffer.
            tb.data.data.committed_out_iv = chosen.out_iv
            tb.committed_out_iv = chosen.out_iv
            del tb.layouts

    for op in operations:
        assert hasattr(op, "layouts"), f"{op.get_name()} has no layouts - must handle"
        assert hasattr(op, "restick_cost_fn"), f"{op.get_name()} has layouts but no restick_cost_fn - must handle"

        cost_fn = op.restick_cost_fn
        upstream_ivs = [
            V.graph.get_buffer(rc.dep.name).committed_out_iv
            if not rc.has_no_stick else -1
            for rc in cost_fn.edge_costs
        ]
        chosen = op.layouts[0]
        out_iv = chosen.out_iv
        cost = cost_fn.cost(upstream_ivs, out_iv)
        print(
            f"MRA select_restickify_locations ({op.get_name()}): "
            f"stick=iv{out_iv} cost={cost} upstream_ivs={upstream_ivs}"
        )
        op.committed_out_iv = out_iv
        record_stick_decisions(op, chosen, out_iv, upstream_ivs)
