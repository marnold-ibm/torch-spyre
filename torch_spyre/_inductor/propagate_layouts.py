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
    compute_restickify_target_layout,
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


def _attach_all_same_cost_fn(
    op: Operation,
    args: "list[SchedNodeArg]",
    out_layouts: "list[FixedTiledLayout]",
    out_dep: "MemoryDep",
) -> None:
    """Build and attach an AllSameNode cost function to op.

    out_layouts are the candidate output layouts for this op.
    out_dep is the MemoryDep for the output buffer, which may differ from each
    arg's dep when inputs are accessed with a different index (e.g. transposed).
    No-op when out_layouts is empty (scalar/broadcast-only args).
    """
    if not out_layouts:
        return
    edge_costs = [EdgeCostMap(arg.dep, arg.layouts, out_layouts, out_dep) for arg in args]
    op.restick_cost_fn = AllSameNode(edge_costs)


def _single_arg_layouts_and_cost(op, output, output_dep, arg, cost_args, layout_fn):
    layouts = [
        layout_fn(op, output, output_dep, arg.dep, layout) for layout in arg.layouts
    ]
    _attach_all_same_cost_fn(op, cost_args, layouts, output_dep)
    return layouts


def _single_arg_op_layout(
    op: Operation,
    output: FixedLayout,
    output_dep: MemoryDep,
    dep: MemoryDep,
    layout: FixedTiledLayout,
) -> FixedTiledLayout:
    data = op.data

    if isinstance(data, Reduction):
        if data.reduction_type == "exx2":
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
        else:
            # Propagate input stick to output if the dim survives, else put stick last.
            x_coords = host_coordinates(layout, dep)
            x_dev_coords = device_coordinates(layout, dep)
            out_coords = host_coordinates(output, output_dep)
            x_stick_expr = x_dev_coords[-1]
            out_stick_dim = matching_dim(out_coords, x_stick_expr)
            if out_stick_dim is None:
                out_dim_order = list(range(len(output.size))) + [-1]
            else:
                out_dim_order = [
                    d for d in range(len(output.size)) if d != out_stick_dim
                ]
                out_dim_order = out_dim_order + [out_stick_dim]
            c_size = [concretize_expr(s) for s in output.size]
            c_stride = [concretize_expr(s) for s in output.stride]
            stl = SpyreTensorLayout(c_size, c_stride, output.dtype, out_dim_order)
        return FixedTiledLayout(
            output.device, output.dtype, output.size, output.stride, stl
        )

    # Single-arg pointwise
    assert isinstance(data, Pointwise)
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

    x_layout = next(iter(x.layouts))
    x_req_layout = x_layout if reduction_coord == x_dev_coords[-1] else compute_restickify_target_layout(x_layout, reduction_coord, x_coords, x_dev_coords)
    if x_req_layout is None:
        raise Unsupported(f"{data.reduction_type}: cannot restickify x to reduction_coord={reduction_coord}")

    y_layout = next(iter(y.layouts))
    y_req_layout = y_layout if generated_coord == y_dev_coords[-1] else compute_restickify_target_layout(y_layout, generated_coord, y_coords, y_dev_coords)
    if y_req_layout is None:
        raise Unsupported(f"{data.reduction_type}: cannot restickify y to generated_coord={generated_coord}")

    x_req_key = LayoutKey.from_stl(x_req_layout.device_layout)
    y_req_key = LayoutKey.from_stl(y_req_layout.device_layout)

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
    result_layout = FixedTiledLayout(
        output.device, output.dtype, output.size, output.stride, stl
    )
    required_out_key = LayoutKey.from_stl(stl)
    x_rc = EdgeCostMap(x.dep, x.layouts, [x_req_layout], x.dep)
    y_rc = EdgeCostMap(y.dep, y.layouts, [y_req_layout], y.dep)
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
    stick_exprs = {
        device_coordinates(layout, arg.dep)[-1]
        for arg in args
        for layout in arg.layouts
        if device_coordinates(layout, arg.dep)[-1] != 0
    }
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
        results.append(
            FixedTiledLayout(
                output.device, output.dtype, output.size, output.stride, stl
            )
        )

    _attach_all_same_cost_fn(op, args, results, output_dep)

    return results


def compute_layouts(
    op: Operation,
    output: FixedLayout,
    output_dep: MemoryDep,
    args: list[SchedNodeArg],
) -> list[FixedTiledLayout]:
    data = op.data
    print()
    print(f"MRA:  ====== In compute_layouts ({op.get_name()})  ======")

    if len(args) > 1 and isinstance(data, Pointwise):
        return _multi_arg_pointwise_layouts(op, output, output_dep, args)

    if isinstance(data, Reduction) and data.reduction_type in (
        MATMUL_REDUCTION_OP,
        BATCH_MATMUL_OP,
    ):
        return _matmul_layouts(op, output, output_dep, args)

    aten_op = next(iter(data.origins)).target if data.origins else None
    if aten_op == spyreop.layernormnorm.default:
        first_layout = next(iter(args[0].layouts))
        if first_layout.size != output.size or first_layout.stride != output.stride:
            raise Unsupported(
                f"views not supported for spyre.layernormnorm({first_layout.size})=>{output.size})"
            )
        return _single_arg_layouts_and_cost(
            op, output, output_dep, args[0], args[:1], _single_arg_op_layout
        )

    # All other single arg ops
    return _single_arg_layouts_and_cost(
        op, output, output_dep, args[0], args, _single_arg_op_layout
    )



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
        if not isinstance(n.node.layout, MutationLayoutSHOULDREMOVE):
            continue
        if isinstance(n.node.data, Pointwise):
            rw = n.read_writes
            output_dep = next(iter(rw.writes))
            args = get_mem_deps(n)
            output = n.node.get_layout()
            layouts = list(compute_layouts(n.node, output, output_dep, args))
            n.node.layout = layouts[0]
        else:
            logger.warning(
                f"propagate_mutation_layouts: unhandled mutation op {type(n.node.data)}"
            )

    return nodes
