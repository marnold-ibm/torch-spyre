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


import torch
from .logging_utils import get_inductor_logger
from torch._inductor.ir import (
    InputBuffer,
    StorageBox,
    TensorBox,
)
from torch._inductor.virtualized import V

from .constants import MAX_RESTICK_COST
from .ir import FixedTiledLayout
from .pass_utils import (
    RestickCost,
    device_coordinates,
    iter_var_id,
)
# ---------------------------------------------------------------------------
# TODO(issue#1371): once SpyreTensorLayout is migrated to c10::SymInt, all
# concretize_expr calls in this file can be removed.
# ---------------------------------------------------------------------------

logger = get_inductor_logger("stickify")

aten = torch.ops.aten
spyreop = torch.ops.spyre


def cost_for_out(rc: "RestickCost", out_iv: int) -> float:
    """Cost for rc's arg to reach out_iv, using the actual read-dep stick iv."""
    if rc.has_no_stick:
        return 0
    buf = V.graph.get_buffer(rc.dep.name)
    if not hasattr(buf, "layout"):
        return rc.min_cost_for_out(out_iv)
    in_iv = iter_var_id(device_coordinates(buf.get_layout(), rc.dep)[-1])
    if in_iv == -1:
        return rc.min_cost_for_out(out_iv)
    if in_iv >= len(rc._cost) or out_iv >= len(rc._cost[in_iv]):
        return MAX_RESTICK_COST
    return rc._cost[in_iv][out_iv]


def select_restickify_locations(operations: list) -> None:
    """Resolve each op's layouts set to a single op.layout.

    Called after propagate_spyre_tensor_layouts has finished so that downstream
    passes (core_division, insert_restickify) see a single FixedTiledLayout on
    each op and each graph input buffer.
    """

    print()
    print("=== In Collapse Layouts ===")

    for name in V.graph.graph_input_names:
        tb = V.graph.graph_inputs[name]
        if (
            isinstance(tb, TensorBox)
            and isinstance(tb.data, StorageBox)
            and isinstance(tb.data.data, InputBuffer)
            and hasattr(tb, "layouts")
        ):
            tb.data.data.layout = next(iter(tb.layouts))
            del tb.layouts

    for op in operations:
        if not hasattr(op, "layouts"):
            continue

        costs = getattr(op, "arg_restick_costs", None)
        if costs and len(op.layouts) > 1:
            rw = op.get_read_writes()
            output_dep = next(iter(rw.writes))
            best_layout = None
            best_total = MAX_RESTICK_COST * len(costs) + 1
            for layout in op.layouts:
                idc = device_coordinates(layout, output_dep)
                out_iv = iter_var_id(idc[-1])
                total = sum(cost_for_out(rc, out_iv) for rc in costs)
                print(
                    f"MRA select_restickify_locations ({op.get_name()}): "
                    f"stick=iv{out_iv} total_cost={total}"
                )
                if total < best_total:
                    best_total = total
                    best_layout = layout
            chosen = best_layout
        else:
            chosen = op.layouts[0]
            if len(op.layouts) > 1:
                print(
                    f"MRA select_restickify_locations ({op.get_name()}): "
                    f"{len(op.layouts)} layouts, no RestickCost, collapsing to first: {chosen}"
                )
            else:
                print(
                    f"MRA select_restickify_locations ({op.get_name()}): single layout: {chosen}"
                )

        op.layout = chosen
        del op.layouts

        # Stamp chosen_stick_iv for any op that carries arg_restick_costs
        # (both pointwise multi-layout and matmul single-layout cases).
        if hasattr(op, "arg_restick_costs"):
            assert isinstance(op.layout, FixedTiledLayout)
            rw = op.get_read_writes()
            output_dep = next(iter(rw.writes))
            op.chosen_stick_iv = iter_var_id(
                device_coordinates(op.layout, output_dep)[-1]
            )
            print(
                f"MRA select_restickify_locations ({op.get_name()}): "
                f"stamped chosen_stick_iv=iv{op.chosen_stick_iv} "
                f"stride_map={list(op.layout.device_layout.stride_map)}"
            )
