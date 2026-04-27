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

import logging
from typing import Any
import math

import sympy
import torch
from .logging_utils import get_inductor_logger
from torch._inductor.ir import (
    ComputedBuffer,
    ExternKernel,
    FallbackKernel,
    FixedLayout,
    InputBuffer,
    MutationLayoutSHOULDREMOVE,
    MultiOutput,
    Operation,
    Pointwise,
    Reduction,
    StorageBox,
    TensorBox,
)
from torch._inductor.dependencies import MemoryDep
from torch._inductor.scheduler import SchedulerNode
from torch._inductor.virtualized import V

from torch_spyre._C import (
    SpyreTensorLayout,
    get_device_dtype,
    get_elem_in_stick,
)
from .errors import Unsupported
from .constants import MATMUL_REDUCTION_OP, BATCH_MATMUL_OP
from .ir import FixedTiledLayout
from .pass_utils import (
    SchedNodeArg,
    concretize_expr,
    get_mem_deps_from_rw,
    host_coordinates,
    device_coordinates,
)
from .views import matching_dim
# ---------------------------------------------------------------------------
# TODO(issue#1371): once SpyreTensorLayout is migrated to c10::SymInt, all
# concretize_expr calls in this file can be removed.
# ---------------------------------------------------------------------------

logger = get_inductor_logger("stickify")

aten = torch.ops.aten
spyreop = torch.ops.spyre



def same_device_size(t1: torch.dtype, t2: torch.dtype) -> bool:
    return get_elem_in_stick(t1) == get_elem_in_stick(t2)


def restickify_device_size(
    old_device_size: list,
    old_sd_outer_dim: int,
    old_sd_host_size: int,
    new_sd_outer_dim: int,
    new_sd_host_size: int,
    stick_size: int = 64,
) -> list:
    """Compute device_size after moving the stick from old_sd to new_sd."""
    assert new_sd_host_size % stick_size == 0, (
        f"Cannot move stick to dimension with size {new_sd_host_size}: "
        f"not a multiple of stick_size={stick_size}"
    )
    new_device_size = list(old_device_size)
    new_device_size[-1] = stick_size
    new_device_size[old_sd_outer_dim] = new_sd_host_size // stick_size
    new_device_size[new_sd_outer_dim] = old_sd_host_size
    return new_device_size


def restickify_stride_map(
    old_stride_map: list,
    old_sd_outer_dim: int,
    old_sd_host_stride: int,
    new_sd_outer_dim: int,
    new_sd_host_stride: int,
    stick_size: int = 64,
) -> list:
    """Compute stride_map after moving the stick from old_sd to new_sd."""
    new_stride_map = list(old_stride_map)
    new_stride_map[-1] = new_sd_host_stride
    new_stride_map[old_sd_outer_dim] = new_sd_host_stride * stick_size
    new_stride_map[new_sd_outer_dim] = old_sd_host_stride
    return new_stride_map


def _iter_var_id(stick_expr) -> int:
    """Iteration variable index from a stick expr: Mod(d2,64) -> 2, d2 -> 2.
    Returns -1 for constant-zero (scalar/broadcast, no real stick).
    NOTE: this is the loop variable index (suffix of dN), NOT a tensor dimension index."""
    if stick_expr == sympy.S.Zero or not stick_expr.free_symbols:
        return -1
    sym = next(iter(stick_expr.free_symbols))
    name = str(sym)
    i = len(name) - 1
    while i >= 0 and name[i].isdigit():
        i -= 1
    return int(name[i + 1:])


_MAX_COST = math.inf


class RestickCost:
    """Thin 2-D cost table for one input arg.

    Indexed [in_iv][out_iv] where iv = iteration variable index:
      the numeric suffix N of the loop variable dN whose Mod(dN,64) is the stick.
      e.g. Mod(d2,64) -> iv=2.
    This is NOT a tensor dimension index.

    required_out_iv:
      None = use op.chosen_stick_iv (decided by collapse_layouts, for pointwise)
      int  = pinned to a specific iter var (for matmul, each arg has its own)
    """

    def __init__(self, dep: MemoryDep, n: int, required_out_iv: "int | None" = None):
        self.dep = dep
        self.required_out_iv = required_out_iv
        self.has_no_stick = False  # True for scalar/broadcast args (no real stick)
        self._cost = [[_MAX_COST] * n for _ in range(n)]
        self._target: list[list] = [[None] * n for _ in range(n)]

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
                if cost == _MAX_COST:
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
        best_cost = _MAX_COST
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
            return _MAX_COST, None
        return self._cost[in_iv][out_iv], self._target[in_iv][out_iv]


def _build_restick_costs(
    args: "list[SchedNodeArg]",
    stick_exprs: set,
) -> "list[RestickCost]":
    """Build one RestickCost per arg for the given candidate output sticks."""
    all_iter_var_ids: set[int] = {_iter_var_id(e) for e in stick_exprs}
    for arg in args:
        for layout in arg.layouts:
            idc = device_coordinates(layout, arg.dep)
            all_iter_var_ids.add(_iter_var_id(idc[-1]))
    n = max(all_iter_var_ids) + 1 if all_iter_var_ids else 1

    result: list[RestickCost] = []
    for arg in args:
        rc = RestickCost(arg.dep, n)
        for layout in arg.layouts:
            ic = host_coordinates(layout, arg.dep)
            idc = device_coordinates(layout, arg.dep)
            in_iv = _iter_var_id(idc[-1])
            if in_iv == -1:
                rc.mark_no_stick()
                continue
            rc.set(in_iv, in_iv, 0, layout)  # staying on same iter var is always free
            for out_expr in stick_exprs:
                out_iv = _iter_var_id(out_expr)
                if in_iv == out_iv:
                    rc.set(in_iv, out_iv, 0, layout)
                else:
                    tgt = compute_restickify_target_layout(
                        arg.dep, layout, out_expr, ic, idc
                    )
                    if tgt is not None:
                        cost = 1
                        for s in layout.size:
                            cost *= concretize_expr(s)
                        rc.set(in_iv, out_iv, cost, tgt)
        result.append(rc)
    return result


def compute_restickify_target_layout(
    dep: MemoryDep,
    layout: FixedTiledLayout,
    target_stick_expr,
    ic: list,
    idc: list,
) -> "FixedTiledLayout | None":
    """Pure. Returns target layout, or None if restickify is infeasible."""
    dl = layout.device_layout
    new_sd = matching_dim(ic, target_stick_expr)
    if new_sd is None:
        return None
    host_size = [concretize_expr(s) for s in layout.size]
    host_stride = [concretize_expr(s) for s in layout.stride]
    old_sd = matching_dim(ic, idc[-1])
    if old_sd is None:
        return None
    old_stick_expr = idc[-1]
    old_stride_map = list(dl.stride_map)
    old_var = next(iter(old_stick_expr.free_symbols))
    new_var = next(iter(target_stick_expr.free_symbols))
    stick_size = 64
    old_sd_outer_dim = next(
        (j for j in range(len(idc) - 1) if old_var in idc[j].free_symbols),
        next((j for j in range(len(idc) - 1) if idc[j] == sympy.S.Zero), None),
    )
    if old_sd_outer_dim is None:
        return None
    candidates = [j for j in range(len(idc) - 1) if new_var in idc[j].free_symbols]
    if not candidates:
        return None
    new_sd_outer_dim = candidates[0]
    if host_size[new_sd] % stick_size != 0:
        return None
    device_size = restickify_device_size(
        list(dl.device_size), old_sd_outer_dim, host_size[old_sd],
        new_sd_outer_dim, host_size[new_sd],
    )
    stride_map = restickify_stride_map(
        old_stride_map, old_sd_outer_dim, host_stride[old_sd],
        new_sd_outer_dim, host_stride[new_sd],
    )
    stl = SpyreTensorLayout(device_size, stride_map, dl.device_dtype)
    return FixedTiledLayout(layout.device, layout.dtype, layout.size, layout.stride, stl)


def _record_restickify(
    op: Operation,
    dep_name: str,
    target_layout: FixedTiledLayout,
    restickify_plan: dict,
) -> None:
    """Append a restickify entry to restickify_plan."""
    restickify_plan.setdefault(op.get_name(), []).append(
        {"arg_name": dep_name, "target_layout": target_layout}
    )


def schedule_restickify(
    op: Operation,
    dep: MemoryDep,
    layout: FixedTiledLayout,
    target_stick_expr,
    ic: list,
    idc: list,
    restickify_plan: dict,
) -> FixedTiledLayout:
    """Schedule a restickify of arg so that its stick aligns with target_stick_expr for op."""
    target_layout = compute_restickify_target_layout(dep, layout, target_stick_expr, ic, idc)
    assert target_layout is not None, (
        f"schedule_restickify: infeasible restickify to {target_stick_expr} "
        f"ic={ic} idc={idc}"
    )
    _record_restickify(op, dep.name, target_layout, restickify_plan)
    return target_layout


def first_arg_pointwise_layout(
    op: Operation,
    output: FixedLayout,
    output_dep: MemoryDep,
    dep: MemoryDep,
    layout: FixedTiledLayout,
    restickify_plan: dict[str, list[dict[str, Any]]],
) -> FixedTiledLayout:
    data = op.data
    origin_node = next(iter(data.origins))
    aten_op = origin_node.target
    match aten_op:
        case aten.clone.default:
            # Clone is generated by an explicit `contiguous()`; on spyre that means use the default row major tiling.
            # Concretize for C++ SpyreTensorLayout constructor.
            c_size = [concretize_expr(s) for s in output.size]
            c_stride = [concretize_expr(s) for s in output.stride]
            stl = SpyreTensorLayout(
                c_size,
                c_stride,
                output.dtype,
                list(range(len(output.size))),
            )
            return FixedTiledLayout(
                output.device, output.dtype, output.size, output.stride, stl
            )

        case spyreop.overwrite.default:
            stl = SpyreTensorLayout(output.size, output.dtype)
            return FixedTiledLayout(
                output.device, output.dtype, output.size, output.stride, stl
            )

        case spyreop.layernormnorm.default:
            # Output layout is determined by layout of first argument only
            x_stl = layout.device_layout
            if layout.size != output.size or layout.stride != output.stride:
                raise Unsupported(
                    f"views not supported for spyre.layernormnorm({layout.size})=>{output.size}) "
                )
            stl = SpyreTensorLayout(x_stl.device_size, x_stl.stride_map, x_stl.device_dtype)
            return FixedTiledLayout(
                output.device, output.dtype, output.size, output.stride, stl
            )

        case _:
            x_stl = layout.device_layout
            in_coords = host_coordinates(layout, dep)
            out_coords = host_coordinates(output, output_dep)
            if (
                in_coords == out_coords
                and dep.index == output_dep.index
                and same_device_size(layout.dtype, output.dtype)
            ):
                # Input and output tensors are being accessed identically and elem size is the same.
                # We can simply propagate the device_layout.
                stl = SpyreTensorLayout(
                    x_stl.device_size,
                    x_stl.stride_map,
                    get_device_dtype(output.dtype),
                )
            else:
                # TODO: We should be able to preserve the input stride_map
                #       unless the operation is changing elems_per_stick.
                #       For now, use the default layout for a mostly row major dimension
                #       ordering, adjusted to put the stick dimension last and move all
                #       non-stick size one dimensions to the right to avoid tiling them.
                in_device_coords = device_coordinates(layout, dep)
                stick_expr = in_device_coords[-1]
                maybe_stick_dim = matching_dim(out_coords, stick_expr)
                out_stick_dim = -1 if maybe_stick_dim is None else maybe_stick_dim
                dim_order = [
                    d
                    for d in range(len(output.size))
                    if d != out_stick_dim and out_coords[d] != 0
                ]
                dim_order += [
                    d
                    for d in range(len(output.size))
                    if d != out_stick_dim and out_coords[d] == 0
                ]
                dim_order += [out_stick_dim]
                # Concretize for C++ SpyreTensorLayout constructor.
                c_size = [concretize_expr(s) for s in output.size]
                c_stride = [concretize_expr(s) for s in output.stride]
                stl = SpyreTensorLayout(c_size, c_stride, output.dtype, dim_order)

            # FixedTiledLayout keeps original (possibly symbolic) size/stride.
            return FixedTiledLayout(
                output.device, output.dtype, output.size, output.stride, stl
            )


def pointwise_layouts(
    op: Operation,
    output: FixedLayout,
    output_dep: MemoryDep,
    args: list[SchedNodeArg],
    restickify_plan: dict[str, list[dict[str, Any]]],
) -> list[FixedTiledLayout]:
    data = op.data
    origin_node = next(iter(data.origins))
    aten_op = origin_node.target
    print()
    print (f"MRA:  ====== In Pointwise ({op.get_name()})  ======")

    if len(args) == 1 or aten_op == spyreop.layernormnorm.default:
        return [
            first_arg_pointwise_layout(
                op, output, output_dep, args[0].dep, layout, restickify_plan
            )
            for layout in args[0].layouts
        ]
    else:
        # len(args) > 1, not layernormnorm

        print("MRA: ARGS:")
        for i, arg in enumerate(args):
            for layout in arg.layouts:
                ic = host_coordinates(layout, arg.dep)
                idc = device_coordinates(layout, arg.dep)
                print(f"MRA: arg {i} layout: host_coords={ic} device_coords={idc} stick_dim={matching_dim(ic, idc[-1])}")
        print("MRA: out_coords:", host_coordinates(output, output_dep))
        print()

        # Stick compatability check.
        # For all tensors whose stick dimension is being iterated over,
        # the indexing expression must be identical.
        stick_exprs = set()
        for arg in args:
            for layout in arg.layouts:
                idc = device_coordinates(layout, arg.dep)
                if idc[-1] != 0:
                    stick_exprs.add(idc[-1])
        print("MRA: stick_exprs (from all layouts):", stick_exprs)
        stick_expr = next(iter(stick_exprs)) if stick_exprs else None

        if len(stick_exprs) > 1:
            logger.warning(
                f"Multi-stick pointwise ({op.get_name()}): producing {len(stick_exprs)} output layouts."
            )

        # Build RestickCost tables: one per arg, indexed [in_iter_var_id][out_iter_var_id].
        if stick_exprs:
            arg_restick_costs = _build_restick_costs(args, stick_exprs)

            print(f"MRA RestickCost tables for {op.get_name()}:")
            for i, (rc, arg) in enumerate(zip(arg_restick_costs, args)):
                print(f"  arg {i} ({rc.dep.name}):")
                print(rc.format_table())

            viable_sticks = [
                e for e in stick_exprs
                if all(rc.min_cost_for_out(_iter_var_id(e)) < _MAX_COST for rc in arg_restick_costs)
            ]
            if not viable_sticks:
                raise Unsupported(
                    f"pointwise_layouts ({op.get_name()}): no viable stick — "
                    f"every candidate stick is infeasible for at least one arg. "
                    f"stick_exprs={stick_exprs}"
                )

            op.arg_restick_costs = arg_restick_costs

        # If the indexing and device element size are identical
        # across all inputs and the output we can just propagate the device layout.
        in_coords = [host_coordinates(next(iter(arg.layouts)), arg.dep) for arg in args]
        out_coords = host_coordinates(output, output_dep)
        can_use_same_layout = True

        if len(stick_exprs) > 1 or any(len(arg.layouts) > 1 for arg in args):
            can_use_same_layout = False
        else:
            for arg, arg_coors in zip(args, in_coords):
                if (
                    arg_coors != out_coords
                    or arg.dep.index != output_dep.index
                    or not same_device_size(next(iter(arg.layouts)).dtype, output.dtype)
                ):
                    can_use_same_layout = False
                    break
            if stick_expr not in stick_exprs:
                can_use_same_layout = False

        results: list[FixedTiledLayout] = []
        for stick_expr in (stick_exprs if stick_exprs else {None}):
            if can_use_same_layout:
                template_stl = next(iter(args[0].layouts)).device_layout
                stl = SpyreTensorLayout(
                    template_stl.device_size,
                    template_stl.stride_map,
                    get_device_dtype(output.dtype),
                )
            else:
                if stick_expr is None:
                    out_stick_dim = -1
                else:
                    maybe_stick_dim = matching_dim(out_coords, stick_expr)
                    out_stick_dim = -1 if maybe_stick_dim is None else maybe_stick_dim
                dim_order = [d for d in range(len(output.size)) if d != out_stick_dim and out_coords[d] != 0]
                dim_order += [d for d in range(len(output.size)) if d != out_stick_dim and out_coords[d] == 0]
                dim_order += [out_stick_dim]
                c_size = [concretize_expr(s) for s in output.size]
                c_stride = [concretize_expr(s) for s in output.stride]
                stl = SpyreTensorLayout(c_size, c_stride, output.dtype, dim_order)
                print(f"MRA: stick_expr={stick_expr} out_stick_dim={out_stick_dim} dim_order={dim_order} stride_map={list(stl.stride_map)}")
            results.append(FixedTiledLayout(output.device, output.dtype, output.size, output.stride, stl))

        return results


def first_arg_reduction_layout(
    op: Operation,
    output: FixedLayout,
    output_dep: MemoryDep,
    dep: MemoryDep,
    layout: FixedTiledLayout,
    restickify_plan: dict[str, list[dict[str, Any]]],
) -> FixedTiledLayout:
    data = op.data
    if data.reduction_type == "exx2":
        x_coords = host_coordinates(layout, dep)
        x_dev_coords = device_coordinates(layout, dep)
        x_stick_expr = x_dev_coords[-1]
        x_stick_dim = matching_dim(x_coords, x_stick_expr)
        if x_stick_dim is None or x_stick_dim != len(layout.size) - 1:
            # TODO: Insert a restickify to enable the operation to be performed
            raise Unsupported(f"exx2: illegal device layout {layout}")

        dim_order = list(range(len(output.size))) + [-1]
        # Concretize for C++ SpyreTensorLayout constructor.
        c_size = [concretize_expr(s) for s in output.size]
        c_stride = [concretize_expr(s) for s in output.stride]
        stl = SpyreTensorLayout(c_size, c_stride, output.dtype, dim_order)
        return FixedTiledLayout(
            output.device, output.dtype, output.size, output.stride, stl
        )
    else:
        x_coords = host_coordinates(layout, dep)
        x_dev_coords = device_coordinates(layout, dep)
        out_coords = host_coordinates(output, output_dep)
        x_stick_expr = x_dev_coords[-1]
        out_stick_dim = matching_dim(out_coords, x_stick_expr)
        if out_stick_dim is None:
            out_dim_order = list(range(len(output.size))) + [-1]
        else:
            out_dim_order = [
                d for d in list(range(len(output.size))) if d != out_stick_dim
            ]
            out_dim_order = out_dim_order + [out_stick_dim]
        # Concretize for C++ SpyreTensorLayout constructor.
        c_size = [concretize_expr(s) for s in output.size]
        c_stride = [concretize_expr(s) for s in output.stride]
        stl = SpyreTensorLayout(c_size, c_stride, output.dtype, out_dim_order)
        result = FixedTiledLayout(
            output.device, output.dtype, output.size, output.stride, stl
        )

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                f"{data.reduction_type} layout: in:{list(layout.size)} -> out:{list(result.size)}, "
                f"device_size={list(result.device_layout.device_size)}"
            )

        return result


def reduction_layouts(
    op: Operation,
    output: FixedLayout,
    output_dep: MemoryDep,
    args: list[SchedNodeArg],
    restickify_plan: dict[str, list[dict[str, Any]]],
) -> list[FixedTiledLayout]:
    data = op.data

    if len(args) == 1:
        return [
            first_arg_reduction_layout(
                op, output, output_dep, args[0].dep, layout, restickify_plan
            )
            for layout in args[0].layouts
        ]
    else:
        # matmul/bmm
        assert data.reduction_type in (MATMUL_REDUCTION_OP, BATCH_MATMUL_OP), (
            f"unexpected multi-arg reduction type: {data.reduction_type}"
        )
        print(f"MRA:  ====== In MatMul ({op.get_name()})  ======")
        out_coords = host_coordinates(output, output_dep)

        print("MRA: ARGS:")
        for i, arg in enumerate(args):
            print("MRA: arg:", i, arg)
            _hc = host_coordinates(next(iter(arg.layouts)), arg.dep)
            _dc = device_coordinates(next(iter(arg.layouts)), arg.dep)
            print("MRA: host_coords:", _hc)
            print("MRA: device_coords:", _dc)
            print("MRA: Matching host stick dim:", matching_dim(_hc, _dc[-1]))
        print("MRA: out_coords:", out_coords)
        print()

        x = args[0]
        y = args[1]
        x_coords = host_coordinates(next(iter(x.layouts)), x.dep)
        x_dev_coords = device_coordinates(next(iter(x.layouts)), x.dep)
        y_coords = host_coordinates(next(iter(y.layouts)), y.dep)
        y_dev_coords = device_coordinates(next(iter(y.layouts)), y.dep)

        x_stick_expr = x_dev_coords[-1]
        y_stick_expr = y_dev_coords[-1]
        x_stick_dim = matching_dim(x_coords, x_stick_expr)
        y_stick_dim = matching_dim(y_coords, y_stick_expr)
        print(f"MRA: x_stick_expr={x_stick_expr} x_stick_dim={x_stick_dim} x_stick_iv=iv{_iter_var_id(x_stick_expr)}")
        print(f"MRA: y_stick_expr={y_stick_expr} y_stick_dim={y_stick_dim} y_stick_iv=iv{_iter_var_id(y_stick_expr)}")
        if x_stick_dim is None or y_stick_dim is None:
            raise Unsupported(
                f"{data.reduction_type}: failed to map stick_dims to host coords"
            )

        # Hardware stick constraints (DF16):
        #   Input1 (x): stick on reduction_dim (the x coord that does NOT appear in output)
        #   Input2 (y): stick on generated_dim (the y coord that appears in output)
        #   Output:     stick on generated_dim
        if matching_dim(out_coords, x_stick_expr) is not None:
            reduction_coord = next(
                c
                for c in x_coords
                if len(c.free_symbols) > 0 and matching_dim(out_coords, c) is None
            )
            print(f"MRA: x stick iv{_iter_var_id(x_stick_expr)} is on output dim -> needs restickify to reduction_coord={reduction_coord} iv{_iter_var_id(reduction_coord)}")
        else:
            reduction_coord = x_stick_expr
            print(f"MRA: x stick iv{_iter_var_id(x_stick_expr)} already on reduction dim -> reduction_coord={reduction_coord}")

        if matching_dim(out_coords, y_stick_expr) is None:
            generated_coord = next(
                c
                for c in y_coords
                if len(c.free_symbols) > 0
                and matching_dim(out_coords, c) is not None
                and matching_dim(x_coords, c) is None
            )
            print(f"MRA: y stick iv{_iter_var_id(y_stick_expr)} not on output dim -> needs restickify to generated_coord={generated_coord} iv{_iter_var_id(generated_coord)}")
        else:
            generated_coord = y_stick_expr
            print(f"MRA: y stick iv{_iter_var_id(y_stick_expr)} already on generated dim -> generated_coord={generated_coord}")

        x_rc = _build_restick_costs([x], {reduction_coord})[0]
        x_rc.required_out_iv = _iter_var_id(reduction_coord)
        y_rc = _build_restick_costs([y], {generated_coord})[0]
        y_rc.required_out_iv = _iter_var_id(generated_coord)
        op.arg_restick_costs = [x_rc, y_rc]

        print(f"MRA RestickCost tables for {op.get_name()}:")
        print(f"  x ({x_rc.dep.name}) required_out_iv=iv{x_rc.required_out_iv}:")
        print(x_rc.format_table())
        print(f"  y ({y_rc.dep.name}) required_out_iv=iv{y_rc.required_out_iv}:")
        print(y_rc.format_table())

        x_req_iv = x_rc.required_out_iv
        y_req_iv = y_rc.required_out_iv
        if x_rc.min_cost_for_out(x_req_iv) >= _MAX_COST:
            raise Unsupported(
                f"{data.reduction_type}: x arg cannot reach required stick iv{x_req_iv}"
            )
        if y_rc.min_cost_for_out(y_req_iv) >= _MAX_COST:
            raise Unsupported(
                f"{data.reduction_type}: y arg cannot reach required stick iv{y_req_iv}"
            )

        out_stick_dim = matching_dim(out_coords, generated_coord)
        print(f"MRA: out_stick_dim={out_stick_dim} from generated_coord={generated_coord}")
        if out_stick_dim is None:
            raise Unsupported(
                f"{data.reduction_type}: failed to map output stick_dim to host coords {out_coords} {generated_coord}"
            )

        out_dims = len(output.size)
        out_dim_order = list(range(out_dims - 2))
        if out_stick_dim == out_dims - 1:
            out_dim_order = out_dim_order + [out_dims - 2, out_dims - 1]
        else:
            out_dim_order = out_dim_order + [out_dims - 1, out_dims - 2]
        print(f"MRA: out_dim_order={out_dim_order}")
        # Concretize for C++ SpyreTensorLayout constructor.
        c_size = [concretize_expr(s) for s in output.size]
        c_stride = [concretize_expr(s) for s in output.stride]
        stl = SpyreTensorLayout(c_size, c_stride, output.dtype, out_dim_order)
        result_layout = FixedTiledLayout(output.device, output.dtype, output.size, output.stride, stl)
        print(f"MRA: matmul output layout: {result_layout}")
        return [result_layout]


def generic_layout(op: Operation) -> FixedTiledLayout:
    output: FixedLayout = op.get_layout()
    # Concretize for C++ SpyreTensorLayout constructor.
    c_size = [concretize_expr(s) for s in output.size]
    # Use the generic stick format
    stl = SpyreTensorLayout(c_size, output.dtype)
    return FixedTiledLayout(
        output.device, output.dtype, output.size, output.stride, stl
    )


def propagate_spyre_tensor_layouts(
    operations: list[Operation],
) -> None:
    # Convert InputBuffers from FixedLayout to FixedTiledLayouts
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
                ftl = FixedTiledLayout(ptl.device, ptl.dtype, ptl.size, ptl.stride, stl)
                print ("Created FixedTiledLayout for Input:", name)
                print (ftl)
                tb.layouts = {ftl}

    # Operations are in topological order (guaranteed by GraphLowering).
    # Visit them and use the inputs' FixedTiledLayouts and the operation being
    # performed to convert each output FixedLayout to a FixedTiledLayout.
    restickify_plan: dict[str, list[dict[str, Any]]] = {}

    it = iter(operations)
    for op in it:
        if op.is_no_op():
            op.layouts = [generic_layout(op)]
        elif isinstance(op, ComputedBuffer):
            if isinstance(op.layout, MutationLayoutSHOULDREMOVE):
                # Mutation ops (e.g. spyre.overwrite) must keep their
                # MutationLayoutSHOULDREMOVE so the scheduler correctly
                # treats them as in-place writes to the target buffer.
                # Their FixedTiledLayout is assigned later in
                # propagate_mutation_layouts, after the scheduler has
                # set up mutation tracking.
                continue
            op.decide_layout()
            rw = op.get_read_writes()
            output_dep = next(iter(rw.writes))
            args = get_mem_deps_from_rw(rw)
            output = op.get_layout()
            if isinstance(op.data, Pointwise):
                op.layouts = pointwise_layouts(
                    op, output, output_dep, args, restickify_plan
                )
            elif isinstance(op.data, Reduction):
                op.layouts = reduction_layouts(
                    op, output, output_dep, args, restickify_plan
                )
            else:
                logger.warning(f"Warning: unhandled node type {type(op.data)}")
        elif isinstance(op, FallbackKernel):
            op = next(it, None)
            if not isinstance(op, MultiOutput):
                raise RuntimeError("FallbackKernel must be followed by MultiOutput")
            op.layouts = [generic_layout(op)]
        elif isinstance(op, ExternKernel):
            logger.warning(f"unhandled node type {type(op)}")
        else:
            logger.warning(f"unhandled operation type {type(op)}")

    V.graph.restickify_plan = restickify_plan


def propagate_mutation_layouts(
    nodes: list,
) -> list:
    """
    Second phase of layout propagation for mutation ops.

    ComputedBuffers with MutationLayoutSHOULDREMOVE are skipped in
    propagate_spyre_tensor_layouts because the scheduler needs to see the
    mutation layout during its initialisation to set up mutation tracking.
    This pass runs as a _pre_fusion_custom_pass (after scheduler init) to
    assign FixedTiledLayout to those remaining mutation ops.
    """
    from .pass_utils import get_mem_deps

    for n in nodes:
        if not (isinstance(n, SchedulerNode) and isinstance(n.node, ComputedBuffer)):
            continue
        if not isinstance(n.node.layout, MutationLayoutSHOULDREMOVE):
            continue
        if isinstance(n.node.data, Pointwise):
            rw = n.read_writes
            output_dep = next(iter(rw.writes))
            args = get_mem_deps(n)
            output = n.node.get_layout()
            n.node.layout = next(iter(pointwise_layouts(n.node, output, output_dep, args, {})))
        else:
            logger.warning(
                f"propagate_mutation_layouts: unhandled mutation op {type(n.node.data)}"
            )

    return nodes


def _cost_for_out(rc: "RestickCost", out_iv: int) -> int:
    """Cost for rc's arg to reach out_iv, using the actual read-dep stick iv."""
    if rc.has_no_stick:
        return 0
    buf = V.graph.get_buffer(rc.dep.name)
    if not hasattr(buf, "layout"):
        return rc.min_cost_for_out(out_iv)
    in_iv = _iter_var_id(device_coordinates(buf.get_layout(), rc.dep)[-1])
    if in_iv == -1:
        return rc.min_cost_for_out(out_iv)
    if in_iv >= len(rc._cost) or out_iv >= len(rc._cost[in_iv]):
        return _MAX_COST
    return rc._cost[in_iv][out_iv]


def collapse_layouts(operations: list) -> None:
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
            best_total = _MAX_COST * len(costs) + 1
            for layout in op.layouts:
                idc = device_coordinates(layout, output_dep)
                out_iv = _iter_var_id(idc[-1])
                total = sum(_cost_for_out(rc, out_iv) for rc in costs)
                print(
                    f"MRA collapse_layouts ({op.get_name()}): "
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
                    f"MRA collapse_layouts ({op.get_name()}): "
                    f"{len(op.layouts)} layouts, no RestickCost, collapsing to first: {chosen}"
                )
            else:
                print(
                    f"MRA collapse_layouts ({op.get_name()}): single layout: {chosen}"
                )

        op.layout = chosen
        del op.layouts

        # Stamp chosen_stick_iv for any op that carries arg_restick_costs
        # (both pointwise multi-layout and matmul single-layout cases).
        if hasattr(op, "arg_restick_costs"):
            rw = op.get_read_writes()
            output_dep = next(iter(rw.writes))
            op.chosen_stick_iv = _iter_var_id(device_coordinates(op.layout, output_dep)[-1])
            print(
                f"MRA collapse_layouts ({op.get_name()}): "
                f"stamped chosen_stick_iv=iv{op.chosen_stick_iv} "
                f"stride_map={list(op.layout.device_layout.stride_map)}"
            )


def schedule_restickify_pass(operations: list) -> None:
    """Populate V.graph.restickify_plan from op.arg_restick_costs.

    Called after collapse_layouts has set op.layout and op.chosen_stick_iv on
    every op.  For each arg whose committed stick differs from the required
    output stick, records a restickify entry.
    """
    restickify_plan = getattr(V.graph, "restickify_plan", {})
    print()
    print("=== In schedule_restickify_pass ===")

    for op in operations:
        costs = getattr(op, "arg_restick_costs", None)
        if not costs:
            continue
        print(f"  op={op.get_name()} chosen_stick_iv=iv{getattr(op, 'chosen_stick_iv', None)}")
        for rc in costs:
            out_iv = (
                rc.required_out_iv
                if rc.required_out_iv is not None
                else getattr(op, "chosen_stick_iv", None)
            )
            buf = V.graph.get_buffer(rc.dep.name)
            in_iv = _iter_var_id(device_coordinates(buf.get_layout(), rc.dep)[-1])
            cost, tgt = rc.cost_and_target(in_iv, out_iv)
            print(
                f"    arg={rc.dep.name} in_iv=iv{in_iv} required_out_iv=iv{rc.required_out_iv} "
                f"out_iv=iv{out_iv} cost={cost} tgt={'None' if tgt is None else list(tgt.device_layout.stride_map)}"
            )
            if out_iv is None:
                print(f"    -> skip (out_iv is None)")
                continue
            if cost == 0:
                print(f"    -> no restickify needed (cost=0)")
                continue
            assert tgt is not None and cost < _MAX_COST, (
                f"schedule_restickify_pass: inviable restickify for "
                f"arg={rc.dep.name} in_iv={in_iv} -> out_iv={out_iv} "
                f"op={op.get_name()} cost={cost}"
            )
            print(f"    -> scheduling restickify iv{in_iv} -> iv{out_iv}")
            logger.warning(
                f"Injecting restickify on {op.get_name()} input {rc.dep.name}: "
                f"iv{in_iv} -> iv{out_iv} "
                f"target_stride_map={list(tgt.device_layout.stride_map)}"
            )
            _record_restickify(op, rc.dep.name, tgt, restickify_plan)

    V.graph.restickify_plan = restickify_plan


def _format_restickify_plan(restickify_plan: dict) -> None:
    print()
    print("=== restickify_plan (entering insert_restickify) ===")
    if not restickify_plan:
        print("  (empty)")
        return
    for consumer_name, entries in restickify_plan.items():
        for e in entries:
            print(
                f"  consumer={consumer_name} arg={e['arg_name']} "
                f"target_stride_map={list(e['target_layout'].device_layout.stride_map)} "
                f"target_device_size={list(e['target_layout'].device_layout.device_size)}"
            )
