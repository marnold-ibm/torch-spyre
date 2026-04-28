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
from .constants import MATMUL_REDUCTION_OP, BATCH_MATMUL_OP, MAX_RESTICK_COST
from .ir import FixedTiledLayout
from .pass_utils import (
    SchedNodeArg,
    concretize_expr,
    get_mem_deps_from_rw,
    host_coordinates,
    device_coordinates,
    iter_var_id,
)
from .optimize_restickify import EdgeCostMap, AllSameNode, FixedInOutNode
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


def _make_identity_write_dep(layout: FixedTiledLayout) -> MemoryDep:
    """Synthetic write dep with identity index in the layout's device coordinates.

    device_coordinates only needs dep.index and dep.ranges.  This constructs
    the identity index — sum(sym_i * stride_map_i) with ranges {sym_i: device_size_i}
    — which is what the upstream node's write dep looks like for its output buffer.
    Used for InputBuffers which do not implement get_read_writes().
    """
    dl = layout.device_layout
    syms = tuple(sympy.Symbol(f"_w{i}") for i in range(len(dl.device_size)))
    index = sum(s * int(m) for s, m in zip(syms, dl.stride_map))
    return MemoryDep("_synthetic", index, syms, tuple(int(sz) for sz in dl.device_size))


def build_edge_restick_costs(
    args: "list[SchedNodeArg]",
    stick_exprs: set,
) -> "list[EdgeCostMap]":
    """Build one EdgeCostMap per arg for the given candidate output sticks.

    All IV indices in the cost table are in THIS NODE's iteration variable
    namespace.
    """
    alliter_var_ids: set[int] = {iter_var_id(e) for e in stick_exprs}
    for arg in args:
        for layout in arg.layouts:
            idc = device_coordinates(layout, arg.dep)
            alliter_var_ids.add(iter_var_id(idc[-1]))
    n = max(alliter_var_ids) + 1 if alliter_var_ids else 1

    result: list[EdgeCostMap] = []
    for arg in args:
        rc = EdgeCostMap(arg.dep, n)
        for layout in arg.layouts:
            ic = host_coordinates(layout, arg.dep)
            idc = device_coordinates(layout, arg.dep)
            local_in_iv = iter_var_id(idc[-1])
            if local_in_iv == -1:
                rc.mark_no_stick()
                continue

            in_iv = local_in_iv
            rc.set(in_iv, in_iv, 0, layout)  # staying on same iter var is always free
            for out_expr in stick_exprs:
                out_iv = iter_var_id(out_expr)
                if in_iv == out_iv:
                    rc.set(in_iv, out_iv, 0, layout)
                else:
                    tgt = compute_restickify_target_layout(layout, out_expr, ic, idc)
                    if tgt is not None:
                        cost = 1
                        for s in layout.size:
                            cost *= concretize_expr(s)
                        rc.set(in_iv, out_iv, cost, tgt)
        result.append(rc)
    return result


def compute_restickify_target_layout(
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
        list(dl.device_size),
        old_sd_outer_dim,
        host_size[old_sd],
        new_sd_outer_dim,
        host_size[new_sd],
    )
    stride_map = restickify_stride_map(
        old_stride_map,
        old_sd_outer_dim,
        host_stride[old_sd],
        new_sd_outer_dim,
        host_stride[new_sd],
    )
    stl = SpyreTensorLayout(device_size, stride_map, dl.device_dtype)
    return FixedTiledLayout(
        layout.device, layout.dtype, layout.size, layout.stride, stl
    )


def first_arg_pointwise_layout(
    op: Operation,
    output: FixedLayout,
    output_dep: MemoryDep,
    dep: MemoryDep,
    layout: FixedTiledLayout,
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
            stl = SpyreTensorLayout(
                x_stl.device_size, x_stl.stride_map, x_stl.device_dtype
            )
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
) -> list[FixedTiledLayout]:
    data = op.data
    origin_node = next(iter(data.origins))
    aten_op = origin_node.target
    print()
    print(f"MRA:  ====== In Pointwise ({op.get_name()})  ======")

    if len(args) == 1 or aten_op == spyreop.layernormnorm.default:
        return [
            first_arg_pointwise_layout(op, output, output_dep, args[0].dep, layout)
            for layout in args[0].layouts
        ]
    else:
        # Standard multi-input pointwise 

        print("MRA: ARGS:")
        for i, arg in enumerate(args):
            for layout in arg.layouts:
                ic = host_coordinates(layout, arg.dep)
                idc = device_coordinates(layout, arg.dep)
                print(
                    f"MRA: arg {i} layout: host_coords={ic} device_coords={idc} stick_dim={matching_dim(ic, idc[-1])}"
                )
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

        # Build EdgeCostMap tables: one per arg, indexed [initer_var_id][outiter_var_id].
        if stick_exprs:
            edge_costs = build_edge_restick_costs(args, stick_exprs)

            print(f"MRA EdgeCostMap tables for {op.get_name()}:")
            for i, (rc, arg) in enumerate(zip(edge_costs, args)):
                print(f"  arg {i} ({arg.dep.name}):")
                print(rc.format_table())

            viable_sticks = [
                e
                for e in stick_exprs
                if all(
                    rc.min_cost_for_out(iter_var_id(e)) < MAX_RESTICK_COST
                    for rc in edge_costs
                )
            ]
            if not viable_sticks:
                raise Unsupported(
                    f"pointwise_layouts ({op.get_name()}): no viable stick — "
                    f"every candidate stick is infeasible for at least one arg. "
                    f"stick_exprs={stick_exprs}"
                )

            op.arg_restick_costs = edge_costs
            op.restick_cost_fn = AllSameNode(edge_costs)

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
        for stick_expr in stick_exprs if stick_exprs else {None}:
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
                c_size = [concretize_expr(s) for s in output.size]
                c_stride = [concretize_expr(s) for s in output.stride]
                stl = SpyreTensorLayout(c_size, c_stride, output.dtype, dim_order)
                print(
                    f"MRA: stick_expr={stick_expr} out_stick_dim={out_stick_dim} dim_order={dim_order} stride_map={list(stl.stride_map)}"
                )
            results.append(
                FixedTiledLayout(
                    output.device, output.dtype, output.size, output.stride, stl
                )
            )

        return results


def first_arg_reduction_layout(
    op: Operation,
    output: FixedLayout,
    output_dep: MemoryDep,
    dep: MemoryDep,
    layout: FixedTiledLayout,
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
) -> list[FixedTiledLayout]:
    data = op.data

    if len(args) == 1:
        return [
            first_arg_reduction_layout(op, output, output_dep, args[0].dep, layout)
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
        print(
            f"MRA: x_stick_expr={x_stick_expr} x_stick_dim={x_stick_dim} x_stick_iv=iv{iter_var_id(x_stick_expr)}"
        )
        print(
            f"MRA: y_stick_expr={y_stick_expr} y_stick_dim={y_stick_dim} y_stick_iv=iv{iter_var_id(y_stick_expr)}"
        )
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
            print(
                f"MRA: x stick iv{iter_var_id(x_stick_expr)} is on output dim -> needs restickify to reduction_coord={reduction_coord} iv{iter_var_id(reduction_coord)}"
            )
        else:
            reduction_coord = x_stick_expr
            print(
                f"MRA: x stick iv{iter_var_id(x_stick_expr)} already on reduction dim -> reduction_coord={reduction_coord}"
            )

        if matching_dim(out_coords, y_stick_expr) is None:
            generated_coord = next(
                c
                for c in y_coords
                if len(c.free_symbols) > 0
                and matching_dim(out_coords, c) is not None
                and matching_dim(x_coords, c) is None
            )
            print(
                f"MRA: y stick iv{iter_var_id(y_stick_expr)} not on output dim -> needs restickify to generated_coord={generated_coord} iv{iter_var_id(generated_coord)}"
            )
        else:
            generated_coord = y_stick_expr
            print(
                f"MRA: y stick iv{iter_var_id(y_stick_expr)} already on generated dim -> generated_coord={generated_coord}"
            )

        x_rc = build_edge_restick_costs([x], {reduction_coord})[0]
        y_rc = build_edge_restick_costs([y], {generated_coord})[0]
        x_req_iv = iter_var_id(reduction_coord)
        y_req_iv = iter_var_id(generated_coord)
        op.arg_restick_costs = [x_rc, y_rc]
        op.restick_cost_fn = FixedInOutNode(
            [x_rc, y_rc],
            required_out_iv=y_req_iv,
            required_in_iv=[x_req_iv, y_req_iv],
        )

        print(f"MRA EdgeCostmap tables for {op.get_name()}:")
        print(f"  x ({x.dep.name}) required_in_iv=iv{x_req_iv}:")
        print(x_rc.format_table())
        print(f"  y ({y.dep.name}) required_in_iv=iv{y_req_iv}:")
        print(y_rc.format_table())
        if x_rc.min_cost_for_out(x_req_iv) >= MAX_RESTICK_COST:
            raise Unsupported(
                f"{data.reduction_type}: x arg cannot reach required stick iv{x_req_iv}"
            )
        if y_rc.min_cost_for_out(y_req_iv) >= MAX_RESTICK_COST:
            raise Unsupported(
                f"{data.reduction_type}: y arg cannot reach required stick iv{y_req_iv}"
            )

        out_stick_dim = matching_dim(out_coords, generated_coord)
        print(
            f"MRA: out_stick_dim={out_stick_dim} from generated_coord={generated_coord}"
        )
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
        result_layout = FixedTiledLayout(
            output.device, output.dtype, output.size, output.stride, stl
        )
        print(f"MRA: matmul output layout: {result_layout}")
        return [result_layout]


def _stamp_out_ivs(layouts: "list[FixedTiledLayout]", output_dep: "MemoryDep") -> None:
    """Stamp layout.out_iv on each candidate layout.

    out_iv is the iteration variable index of the stick in this node's output,
    in this node's IV namespace.  Pre-computed here so optimize_restickify.py
    can read it without calling device_coordinates.
    Skips layouts where output_dep is not a MemoryDep (e.g. StarDep).
    """
    if not isinstance(output_dep, MemoryDep):
        for layout in layouts:
            layout.out_iv = -1
        return
    for layout in layouts:
        idc = device_coordinates(layout, output_dep)
        layout.out_iv = iter_var_id(idc[-1])


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
                write_dep = _make_identity_write_dep(ftl)
                ftl.out_iv = iter_var_id(device_coordinates(ftl, write_dep)[-1])
                print("Created FixedTiledLayout for Input:", name)
                print(ftl)
                tb.layouts = {ftl}

    # Operations are in topological order (guaranteed by GraphLowering).
    # Visit them and use the inputs' FixedTiledLayouts and the operation being
    # performed to convert each output FixedLayout to a FixedTiledLayout.
    it = iter(operations)
    for op in it:
        if op.is_no_op():
            op.layouts = [generic_layout(op)]
            _stamp_out_ivs(op.layouts, next(iter(op.get_read_writes().writes)))
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
                op.layouts = pointwise_layouts(op, output, output_dep, args)
            elif isinstance(op.data, Reduction):
                op.layouts = reduction_layouts(op, output, output_dep, args)
            else:
                logger.warning(f"Warning: unhandled node type {type(op.data)}")
            if hasattr(op, "layouts"):
                _stamp_out_ivs(op.layouts, output_dep)
        elif isinstance(op, FallbackKernel):
            op = next(it, None)
            if not isinstance(op, MultiOutput):
                raise RuntimeError("FallbackKernel must be followed by MultiOutput")
            op.layouts = [generic_layout(op)]
            _stamp_out_ivs(op.layouts, next(iter(op.get_read_writes().writes)))
        elif isinstance(op, ExternKernel):
            logger.warning(f"unhandled node type {type(op)}")
        else:
            logger.warning(f"unhandled operation type {type(op)}")


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
            n.node.layout = next(
                iter(pointwise_layouts(n.node, output, output_dep, args))
            )
        else:
            logger.warning(
                f"propagate_mutation_layouts: unhandled mutation op {type(n.node.data)}"
            )

    return nodes
