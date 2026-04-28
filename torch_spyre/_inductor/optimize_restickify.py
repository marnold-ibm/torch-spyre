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

    upstream_out_iv_to_local_in_iv translates from the upstream buffer's output
    IV namespace into this node's in_iv row index.  Indexed by upstream IV;
    value is the corresponding in_iv in this node's namespace; -1 = unmapped.

    required_out_iv (optional, set after construction):
      None = use op.chosen_stick_iv (decided by select_restickify_locations, for pointwise)
      int  = pinned to a specific iter var (for matmul, each arg has its own)
    """

    def __init__(self, dep: "MemoryDep", n: int):
        # dep is kept for IR passes (insert_restickify) that need the buffer name
        # and read index.  Cost optimization code does not and should not use it
        self.dep = dep
        self.required_out_iv: "int | None" = None
        self.has_no_stick = False  # True for scalar/broadcast args (no real stick)
        self._cost = [[INF] * n for _ in range(n)]
        self._target: list[list] = [[None] * n for _ in range(n)]
        # Indexed by upstream buffer's output IV (upstream's namespace).
        # Value is the in_iv row index in this node's namespace. -1 = unmapped.
        self.upstream_out_iv_to_local_in_iv: list[int] = [-1] * n

    def mark_no_stick(self) -> None:
        """Mark this arg as scalar/broadcast — compatible with any output at zero cost."""
        self.has_no_stick = True

    def set(self, in_iv: int, out_iv: int, cost: int, target) -> None:
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

    def min_cost_for_out(self, out_iv: int) -> int:
        """Minimum cost across all in iter vars to reach out_iv."""
        if self.has_no_stick:
            return 0
        return min(row[out_iv] for row in self._cost)

    def best_target_for_out(self, out_iv: int):
        """Returns (cost, target_layout) for cheapest transition to out_iv."""
        if self.has_no_stick:
            return 0, None
        best_cost = INF
        best_tgt = None
        for row, trow in zip(self._cost, self._target):
            if row[out_iv] < best_cost:
                best_cost = row[out_iv]
                best_tgt = trow[out_iv]
        return best_cost, best_tgt

    def cost_and_target(self, in_iv: "int | None", out_iv: int):
        """Returns (cost, target_layout) for in_iv -> out_iv.

        in_iv=None: input's committed iter var unknown, falls back to best_target_for_out.
        has_no_stick args always return (0, None).
        Both in_iv and out_iv are iteration variable indices (NOT tensor dim indices).
        """
        if self.has_no_stick:
            return 0, None
        if in_iv is None:
            return self.best_target_for_out(out_iv)
        if in_iv >= len(self._cost) or out_iv >= len(self._cost[in_iv]):
            return INF, None
        return self._cost[in_iv][out_iv], self._target[in_iv][out_iv]


def edge_cost_for_out(rc: "EdgeCostMap", upstream_committed_iv: int, out_iv: int) -> int:
    """Cost for rc's arg to reach out_iv.

    upstream_committed_iv: the chosen output IV of the upstream buffer, in the
        upstream buffer's IV namespace.  Pass -1 if unknown (falls back to
        minimum cost across all input IVs).
    out_iv: target output IV of this node, in this node's IV namespace.
    rc.upstream_out_iv_to_local_in_iv translates between the two namespaces.
    """
    if rc.has_no_stick:
        return 0
    if upstream_committed_iv < 0 or upstream_committed_iv >= len(rc.upstream_out_iv_to_local_in_iv):
        return rc.min_cost_for_out(out_iv)
    in_iv = rc.upstream_out_iv_to_local_in_iv[upstream_committed_iv]
    if in_iv == -1:
        return rc.min_cost_for_out(out_iv)
    return rc.cost_and_target(in_iv, out_iv)[0]


class RestickNodeCost(abc.ABC):

    def __init__(self, edge_costs):
        self.edge_costs = edge_costs

    @abc.abstractmethod
    def cost_for_out(self, out_iv: int) -> int: ...


class AllSameNode(RestickNodeCost):

    def cost_for_out(self, out_iv: int, upstream_committed_ivs: "list[int] | None" = None) -> int:
        """Cost to produce output at out_iv.

        upstream_committed_ivs: per-edge upstream output IV (upstream's namespace),
            or None/-1 entries when unknown (falls back to min cost across all IVs).
        """
        if upstream_committed_ivs is None:
            upstream_committed_ivs = [-1] * len(self.edge_costs)
        in_edge_costs = [
            edge_cost_for_out(rc, committed_iv, out_iv)
            for rc, committed_iv in zip(self.edge_costs, upstream_committed_ivs)
        ]
        return INF if INF in in_edge_costs else sum(in_edge_costs)


class FixedOutNode(RestickNodeCost):

    def __init__(self, edge_costs, required_out_iv: int):
        super().__init__(edge_costs)
        self.required_out_iv = required_out_iv

    def cost_for_out(self, out_iv: int, upstream_committed_ivs: "list[int] | None" = None) -> int:
        """Cost to produce output at out_iv.

        upstream_committed_ivs: per-edge upstream output IV (upstream's namespace),
            or None/-1 entries when unknown (falls back to min cost across all IVs).
        """
        if out_iv != self.required_out_iv:
            return INF
        if upstream_committed_ivs is None:
            upstream_committed_ivs = [-1] * len(self.edge_costs)
        in_edge_costs = [
            edge_cost_for_out(rc, committed_iv, out_iv)
            for rc, committed_iv in zip(self.edge_costs, upstream_committed_ivs)
        ]
        return INF if INF in in_edge_costs else sum(in_edge_costs)


class PassthroughNode(RestickNodeCost):

    def __init__(self):
        pass

    def cost_for_out(self, out_iv: int) -> int:
        return 0


def optimize_restickify_locations(operations: list) -> None:

    # Dumb, for now
    always_choose_first_arg_stick(operations)


def _pick_layout_for_skipped_node(op) -> "FixedTiledLayout":
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
            chosen = _pick_layout_for_skipped_node(op)
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
        op.chosen_layout = chosen

        # Mark chosen stick IV for finalize_layouts
        if hasattr(op, "arg_restick_costs"):
            op.chosen_stick_iv = out_iv
            print(
                f"MRA: Node ({op.get_name()}): "
                f"chosen_stick_iv=iv{op.chosen_stick_iv} "
                f"stride_map={list(chosen.device_layout.stride_map)}"
            )
