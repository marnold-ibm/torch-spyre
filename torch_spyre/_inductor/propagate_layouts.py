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
from .constants import MATMUL_REDUCTION_OP, BATCH_MATMUL_OP
from .ir import FixedTiledLayout
from .pass_utils import (
    SchedNodeArg,
    concretize_expr,
    get_mem_deps_from_rw,
    host_coordinates,
    device_coordinates,
    iter_var_id,
)
from .optimize_restickify import EdgeCostMap, AllSameNode, FixedInOutNode, LayoutKey
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


def build_edge_restick_costs(
    args: "list[SchedNodeArg]",
    out_key_by_expr: "dict",
) -> "list[EdgeCostMap]":
    """Build one EdgeCostMap per arg for the given candidate output sticks.

    out_key_by_expr maps each stick_expr to the LayoutKey of the output layout
    that uses that stick.  This ensures out_key is the output buffer's key,
    not the input's post-restickify key — which would be ambiguous when two
    inputs have the same device_layout but different sticks.
    """
    result: list[EdgeCostMap] = []
    for arg in args:
        rc = EdgeCostMap(arg.dep)
        for layout in arg.layouts:
            ic = host_coordinates(layout, arg.dep)
            idc = device_coordinates(layout, arg.dep)
            if iter_var_id(idc[-1]) == -1:
                rc.mark_no_stick()
                continue

            in_key = LayoutKey.from_stl(layout.device_layout)
            print(f"MRA build_edge_restick_costs: arg={arg.dep.name} idc={idc} stick=idc[-1]={idc[-1]} in_key={list(in_key.stride_map)}")
            for out_expr, out_key in out_key_by_expr.items():
                if out_expr == idc[-1]:
                    print(f"MRA build_edge_restick_costs:   out_expr={out_expr} same stick -> out_key={list(out_key.stride_map)} cost=0")
                    rc.set_cost_and_target(in_key, out_key, 0, None)
                else:
                    tgt = compute_restickify_target_layout(layout, out_expr, ic, idc)
                    print(f"MRA build_edge_restick_costs:   out_expr={out_expr} tgt={None if tgt is None else list(tgt.device_layout.stride_map)}")
                    if tgt is not None:
                        cost = 1
                        for s in layout.size:
                            cost *= concretize_expr(s)
                        rc.set_cost_and_target(in_key, out_key, cost, tgt)
        result.append(rc)
    return result


def _collect_stick_exprs(args: "list[SchedNodeArg]") -> set:
    exprs = set()
    for arg in args:
        for layout in arg.layouts:
            idc = device_coordinates(layout, arg.dep)
            if idc[-1] != 0:
                exprs.add(idc[-1])
    return exprs


def _attach_all_same_cost_fn(
    op: Operation,
    args: "list[SchedNodeArg]",
    stick_exprs: set,
    out_key_by_expr: "dict",
) -> None:
    """Build and attach an AllSameNode cost function to op.

    out_key_by_expr maps each stick_expr to the LayoutKey of the output layout
    for that stick.

    No-op when stick_exprs is empty (scalar/broadcast-only args).
    Raises Unsupported if no stick is viable for all args.
    """
    print(f"MRA _attach_all_same_cost_fn ({op.get_name()}): stick_exprs={stick_exprs}")
    if not stick_exprs:
        print(f"MRA _attach_all_same_cost_fn ({op.get_name()}): no stick exprs, skipping")
        return
    for i, arg in enumerate(args):
        for layout in arg.layouts:
            ic = host_coordinates(layout, arg.dep)
            idc = device_coordinates(layout, arg.dep)
            dl = layout.device_layout
            print(
                f"MRA   arg {i} ({arg.dep.name}): host_coords={ic} device_coords={idc}"
                f" stick_iv=iv{iter_var_id(idc[-1])}"
                f" device_size={list(dl.device_size)} stride_map={list(dl.stride_map)}"
            )
    edge_costs = build_edge_restick_costs(args, out_key_by_expr)
    print(f"MRA EdgeCostMap tables for {op.get_name()}:")
    for i, (rc, arg) in enumerate(zip(edge_costs, args)):
        print(f"  arg {i} ({arg.dep.name}):")
        print(rc.format_table())
    # Collect all out_keys that appear in at least one non-no_stick cost map,
    # then keep only those feasible for every arg.
    all_out_keys: set[LayoutKey] = set()
    for rc in edge_costs:
        if not rc.has_no_stick:
            for row in rc._cost.values():
                all_out_keys.update(row.keys())
    viable_keys = [k for k in all_out_keys if all(rc.feasible_for_out(k) for rc in edge_costs)]
    print(f"MRA   viable_out_keys={[list(k.stride_map) for k in viable_keys]}")
    if not viable_keys:
        raise Unsupported(
            f"_attach_all_same_cost_fn ({op.get_name()}): no viable stick — "
            f"every candidate stick is infeasible for at least one arg. "
            f"stick_exprs={stick_exprs}"
        )
    op.restick_cost_fn = AllSameNode(edge_costs)
    print(f"MRA   attached AllSameNode to {op.get_name()}")


def _single_arg_layouts_and_cost(op, output, output_dep, arg, cost_args, layout_fn):
    layouts = [
        layout_fn(op, output, output_dep, arg.dep, layout)
        for layout in arg.layouts
    ]
    out_key_by_expr = {
        device_coordinates(in_layout, arg.dep)[-1]: LayoutKey.from_stl(out_layout.device_layout)
        for in_layout, out_layout in zip(arg.layouts, layouts)
    }
    _attach_all_same_cost_fn(op, cost_args, _collect_stick_exprs(cost_args), out_key_by_expr)
    return layouts


def _single_arg_layout(
    op: Operation,
    output: FixedLayout,
    output_dep: MemoryDep,
    dep: MemoryDep,
    layout: FixedTiledLayout,
) -> FixedTiledLayout:
    data = op.data

    # exx2: stick must be on last host dim; output gets default dim_order with stick last.
    if isinstance(data, Reduction) and data.reduction_type == "exx2":
        x_coords = host_coordinates(layout, dep)
        x_dev_coords = device_coordinates(layout, dep)
        x_stick_expr = x_dev_coords[-1]
        x_stick_dim = matching_dim(x_coords, x_stick_expr)
        if x_stick_dim is None or x_stick_dim != len(layout.size) - 1:
            # TODO: Insert a restickify to enable the operation to be performed
            raise Unsupported(f"exx2: illegal device layout {layout}")
        dim_order = list(range(len(output.size))) + [-1]
        c_size = [concretize_expr(s) for s in output.size]
        c_stride = [concretize_expr(s) for s in output.stride]
        stl = SpyreTensorLayout(c_size, c_stride, output.dtype, dim_order)
        return FixedTiledLayout(output.device, output.dtype, output.size, output.stride, stl)

    # Non-exx2 reduction: propagate input stick to output if the dim survives, else put stick last.
    if isinstance(data, Reduction):
        x_coords = host_coordinates(layout, dep)
        x_dev_coords = device_coordinates(layout, dep)
        out_coords = host_coordinates(output, output_dep)
        x_stick_expr = x_dev_coords[-1]
        out_stick_dim = matching_dim(out_coords, x_stick_expr)
        if out_stick_dim is None:
            out_dim_order = list(range(len(output.size))) + [-1]
        else:
            out_dim_order = [d for d in range(len(output.size)) if d != out_stick_dim]
            out_dim_order = out_dim_order + [out_stick_dim]
        c_size = [concretize_expr(s) for s in output.size]
        c_stride = [concretize_expr(s) for s in output.stride]
        stl = SpyreTensorLayout(c_size, c_stride, output.dtype, out_dim_order)
        return FixedTiledLayout(output.device, output.dtype, output.size, output.stride, stl)

    # Pointwise: handle special ops, then propagate stick from input to output.
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


def _matmul_layouts(
    op: Operation,
    output: FixedLayout,
    output_dep: MemoryDep,
    args: list[SchedNodeArg],
) -> list[FixedTiledLayout]:
    data = op.data
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

    x_rc = build_edge_restick_costs([x], _single_out_key_by_expr(x, reduction_coord))[0]
    y_rc = build_edge_restick_costs([y], _single_out_key_by_expr(y, generated_coord))[0]
    x_req_key = _required_key_from_single_stick_rc(x_rc)
    y_req_key = _required_key_from_single_stick_rc(y_rc)

    print(f"MRA EdgeCostmap tables for {op.get_name()}:")
    print(f"  x ({x.dep.name}) required_in_key={list(x_req_key.stride_map)}:")
    print(x_rc.format_table())
    print(f"  y ({y.dep.name}) required_in_key={list(y_req_key.stride_map)}:")
    print(y_rc.format_table())
    if not x_rc.feasible_for_out(x_req_key):
        raise Unsupported(
            f"{data.reduction_type}: x arg cannot reach required stick {list(x_req_key.stride_map)}"
        )
    if not y_rc.feasible_for_out(y_req_key):
        raise Unsupported(
            f"{data.reduction_type}: y arg cannot reach required stick {list(y_req_key.stride_map)}"
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
    required_out_key = LayoutKey.from_stl(stl)
    op.restick_cost_fn = FixedInOutNode(
        [x_rc, y_rc],
        required_out_key=required_out_key,
        required_in_keys=[x_req_key, y_req_key],
    )
    print(f"MRA: matmul output layout: {result_layout}")
    return [result_layout]


def _multi_arg_pointwise_layouts(
    op: Operation,
    output: FixedLayout,
    output_dep: MemoryDep,
    args: list[SchedNodeArg],
) -> list[FixedTiledLayout]:
    stick_exprs = _collect_stick_exprs(args)
    print("MRA: stick_exprs (from all layouts):", stick_exprs)
    stick_expr = next(iter(stick_exprs)) if stick_exprs else None

    if len(stick_exprs) > 1:
        logger.warning(
            f"Multi-stick pointwise ({op.get_name()}): producing {len(stick_exprs)} output layouts."
        )

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
    out_key_by_expr: dict = {}
    # Sort stick exprs for determinism
    for stick_expr in sorted(stick_exprs, key=str) if stick_exprs else [None]:
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
        if stick_expr is not None:
            out_key_by_expr[stick_expr] = LayoutKey.from_stl(stl)
        results.append(
            FixedTiledLayout(
                output.device, output.dtype, output.size, output.stride, stl
            )
        )

    _attach_all_same_cost_fn(op, args, stick_exprs, out_key_by_expr)

    return results


def compute_layouts(
    op: Operation,
    output: FixedLayout,
    output_dep: MemoryDep,
    args: list[SchedNodeArg],
) -> list[FixedTiledLayout]:
    data = op.data
    origin_node = next(iter(data.origins)) if data.origins else None
    aten_op = origin_node.target if origin_node is not None else None
    print()
    print(f"MRA:  ====== In compute_layouts ({op.get_name()})  ======")

    if len(args) > 1 and isinstance(data, Pointwise):
        return _multi_arg_pointwise_layouts(op, output, output_dep, args)

    if isinstance(data, Reduction) and data.reduction_type in (MATMUL_REDUCTION_OP, BATCH_MATMUL_OP):
        return _matmul_layouts(op, output, output_dep, args)

    if aten_op == spyreop.layernormnorm.default:
        first_layout = next(iter(args[0].layouts))
        if first_layout.size != output.size or first_layout.stride != output.stride:
            raise Unsupported(
                f"views not supported for spyre.layernormnorm({first_layout.size})=>{output.size})"
            )
        return _single_arg_layouts_and_cost(
            op, output, output_dep, args[0], args[:1], _single_arg_layout
        )

    return _single_arg_layouts_and_cost(
        op, output, output_dep, args[0], args, _single_arg_layout
    )



def _single_out_key_by_expr(arg: "SchedNodeArg", out_expr) -> "dict":
    """Build a single-entry out_key_by_expr for one stick expression from one arg."""
    layout = next(iter(arg.layouts))
    ic = host_coordinates(layout, arg.dep)
    idc = device_coordinates(layout, arg.dep)
    if out_expr == idc[-1]:
        return {out_expr: LayoutKey.from_stl(layout.device_layout)}
    tgt = compute_restickify_target_layout(layout, out_expr, ic, idc)
    if tgt is not None:
        return {out_expr: LayoutKey.from_stl(tgt.device_layout)}
    return {out_expr: LayoutKey.from_stl(layout.device_layout)}


def _required_key_from_single_stick_rc(rc: EdgeCostMap) -> LayoutKey:
    """Extract the target LayoutKey from a cost map built with exactly one stick_expr.

    Returns the non-identity out_key if a restickify target exists, otherwise
    the identity (the input is already on the required stick).
    """
    for in_key, row in rc._cost.items():
        for out_key in row:
            if in_key != out_key:
                return out_key
    # All entries are identity transitions — input is already on the required stick.
    for in_key in rc._cost:
        return in_key
    raise AssertionError("EdgeCostMap has no entries")


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
        elif isinstance(op, ComputedBuffer):
            if isinstance(op.layout, MutationLayoutSHOULDREMOVE):
                # # Mutation ops write into an existing buffer. Give them an
                # # AllSameNode so the stick propagates through unchanged.
                # rw = op.get_read_writes()
                # args = get_mem_deps_from_rw(rw)
                # print(f"MRA mutation op ({op.get_name()}) args:")
                # for arg in args:
                #     print(f"  dep={arg.dep.name} layouts={list(arg.layouts)}")
                # stick_exprs = _collect_stick_exprs(args)
                # if stick_exprs:
                #     _attach_all_same_cost_fn(op, args, stick_exprs)
                # target_buf = op.layout.target
                # op.layouts = list(target_buf.layouts) if hasattr(target_buf, "layouts") else [generic_layout(op)]
                continue
            op.decide_layout()
            rw = op.get_read_writes()
            output_dep = next(iter(rw.writes))
            args = get_mem_deps_from_rw(rw)
            output = op.get_layout()
            if isinstance(op.data, (Pointwise, Reduction)):
                op.layouts = compute_layouts(op, output, output_dep, args)
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
        print(f"MRA propagate_mutation_layouts: node={n.node.get_name()} layout={type(n.node.layout).__name__}")
        if not isinstance(n.node.layout, MutationLayoutSHOULDREMOVE):
            continue
        print(f"MRA propagate_mutation_layouts: processing mutation op {n.node.get_name()}")
        if isinstance(n.node.data, Pointwise):
            rw = n.read_writes
            output_dep = next(iter(rw.writes))
            args = get_mem_deps(n)
            print(f"MRA propagate_mutation_layouts: args={[a.dep.name for a in args]}")
            output = n.node.get_layout()
            layouts = list(compute_layouts(n.node, output, output_dep, args))
            print(f"MRA propagate_mutation_layouts: got {len(layouts)} layouts, setting layout={layouts[0]}")
            n.node.layout = layouts[0]
            print(f"MRA propagate_mutation_layouts: n.node id={id(n.node)} layout now={type(n.node.layout).__name__}")
            buf = V.graph.get_buffer(n.node.get_name())
            print(f"MRA propagate_mutation_layouts: name_to_buffer[{n.node.get_name()}] id={id(buf)} layout={type(buf.get_layout()).__name__}")
        else:
            logger.warning(
                f"propagate_mutation_layouts: unhandled mutation op {type(n.node.data)}"
            )

    return nodes
