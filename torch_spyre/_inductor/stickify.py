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


import torch
from .logging_utils import get_inductor_logger
from torch._inductor.ir import (
    ComputedBuffer,
    FallbackKernel,
    FixedLayout,
    InputBuffer,
    MultiOutput,
    Pointwise,
    Reduction,
    StorageBox,
    TensorBox,
)
from torch._inductor.scheduler import (
    BaseSchedulerNode,
    SchedulerNode,
    ExternKernelSchedulerNode,
    NopKernelSchedulerNode,
)
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
    get_mem_deps,
    host_coordinates,
    device_coordinates,
)
from .views import matching_dim


logger = get_inductor_logger("stickify")

aten = torch.ops.aten
spyreop = torch.ops.spyre


def same_device_size(t1: torch.dtype, t2: torch.dtype) -> bool:
    return get_elem_in_stick(t1) == get_elem_in_stick(t2)


def compute_device_size(host_size: list, dim_map: list, stick_size: int = 64) -> list:
    """Compute device_size from host_size and a target dim_map.

    The C++ SpyreTensorLayout constructor only produces a correct stride_map
    when the host is contiguous in the given dim_order.  We replicate the logic
    here to handle non-contiguous strides (e.g. after a transpose).

    Each non-stick device dim gets the full size of its host dim, except when
    it shares a host dim with the stick — then it gets host_size // stick_size.
    The last (stick) device dim is always stick_size (64 at fp16).
    """
    stick_host_dim = dim_map[-1]
    result = []
    for j in range(len(dim_map) - 1):
        h = dim_map[j]
        if h == stick_host_dim:
            result.append(host_size[h] // stick_size)
        else:
            result.append(host_size[h])
    result.append(stick_size)
    return result


def dim_map_to_stride_map(host_stride: list, device_size: list, dim_map: list) -> list:
    """Compute stride_map from host_stride and tiling geometry (device_size, dim_map).

    TODO: Remove this and make C++ version available?

    Python port of the C++ static function dim_map_to_stride_map, which is not
    exposed to Python.  Iterates backwards, accumulating the product of inner
    device sizes for each host dim so that higher-dimensional cases with a host
    dim appearing 3+ times in dim_map are handled correctly.
    """
    n = len(device_size)
    stride_map = [-1] * n
    last_stride: dict[int, int] = {}
    for j in range(n - 1, -1, -1):
        d = dim_map[j]
        if d == -1:
            stride_map[j] = -1
        elif d not in last_stride:
            stride_map[j] = host_stride[d]
            last_stride[d] = stride_map[j] * device_size[j]
        else:
            stride_map[j] = last_stride[d]
            last_stride[d] = stride_map[j] * device_size[j]
    return stride_map


def schedule_restickify(
    n: SchedulerNode,
    arg: SchedNodeArg,
    arg_i: int,
    target_stick_expr,
    ic: list,
    restick_needed: dict,
) -> None:
    """Record a restickify needed for arg to match target_stick_expr.

    Computes the target FixedTiledLayout by swapping the current stick dim with
    the host dim that matches target_stick_expr, then appends an entry to
    restick_needed[n] for the insert_restickify pass to act on.
    """
    dl = arg.layout.device_layout
    new_sd = matching_dim(ic, target_stick_expr)
    assert new_sd is not None, (
        f"Could not find a host dimension matching stick expr {target_stick_expr} in {ic}"
    )
    orig_dim_map = dl.dim_map
    old_sd = orig_dim_map[-1]
    new_dim_map = [
        new_sd if x == old_sd else old_sd if x == new_sd else x
        for x in orig_dim_map
    ]
    device_size = compute_device_size(list(arg.layout.size), new_dim_map)
    stride_map = dim_map_to_stride_map(
        list(arg.layout.stride), device_size, new_dim_map
    )
    stl = SpyreTensorLayout(device_size, new_dim_map, stride_map, dl.device_dtype)
    target_layout = FixedTiledLayout(
        arg.layout.device, arg.layout.dtype, arg.layout.size, arg.layout.stride, stl
    )
    restick_needed.setdefault(n, []).append(
        {"arg_index": arg_i, "target_layout": target_layout}
    )
    return target_layout


def pointwise_layout(
    n: SchedulerNode, args: list[SchedNodeArg], restick_needed: dict
) -> FixedTiledLayout:
    pw: Pointwise = n.node.data
    output: FixedLayout = n.node.get_layout()
    output_dep = next(iter(n.read_writes.writes))
    origin_node = next(iter(pw.origins))
    op = origin_node.target

    if len(args) == 1:
        x = args[0]
        match op:
            case aten.clone.default:
                # Clone is generated by an explicit `contiguous()`; on spyre that means use the default row major tiling.
                stl = SpyreTensorLayout(
                    output.size,
                    output.stride,
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
                x_stl = x.layout.device_layout
                in_coords = host_coordinates(x.layout, x.dep)
                out_coords = host_coordinates(output, output_dep)
                if (
                    in_coords == out_coords
                    and x.dep.index == output_dep.index
                    and same_device_size(x.layout.dtype, output.dtype)
                ):
                    # Input and output tensors are being accessed identically and elem size is the same.
                    # We can simply propagate the device_layout.
                    stl = SpyreTensorLayout(
                        x_stl.device_size,
                        x_stl.dim_map,
                        x_stl.stride_map,
                        get_device_dtype(output.dtype),
                    )
                else:
                    # TODO: Once we eliminate the dim_map from the STL,
                    #       we should be able to preserve the input stride_map
                    #       unless the operation is changing elems_per_stick.
                    #       Until then, use the default layout for a mostly row major dimension
                    #       ordering, adjusted to put the stick dimension last and move all
                    #       non-stick size one dimensions to the right to avoid tiling them.
                    in_device_coords = device_coordinates(x.layout, x.dep)
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
                    stl = SpyreTensorLayout(
                        output.size, output.stride, output.dtype, dim_order
                    )

                return FixedTiledLayout(
                    output.device, output.dtype, output.size, output.stride, stl
                )

    elif op == spyreop.layernormnorm.default:
        # Output layout is determined by layout of first argument only
        x = args[0]
        x_stl = x.layout.device_layout
        if x.layout.size != output.size or x.layout.stride != output.stride:
            raise Unsupported(
                f"views not supported for spyre.layernormnorm({x.layout.size})=>{output.size}) "
            )
        stl = SpyreTensorLayout(
            x_stl.device_size, x_stl.dim_map, x_stl.stride_map, x_stl.device_dtype
        )
        return FixedTiledLayout(
            output.device, output.dtype, output.size, output.stride, stl
        )
    else:
        in_coords = [host_coordinates(arg.layout, arg.dep) for arg in args]
        in_device_coords = [device_coordinates(arg.layout, arg.dep) for arg in args]
        out_coords = host_coordinates(output, output_dep)

        # Stick compatability check.
        # For all tensors whose stick dimension is being iterated over,
        # the indexing expression must be identical.
        stick_exprs = set()
        for idc in in_device_coords:
            if idc[-1] != 0:
                stick_exprs.add(idc[-1])
        stick_expr = next(iter(stick_exprs)) if stick_exprs else None

        if len(stick_exprs) > 1:
            # This is a legal PyTorch operation that we cannot execute without inserting restickify operations.
            logger.warning(
                f"Injecting restickify to resolve pointwise op with nonuniform stick indexing: {stick_exprs}."
            )

            # Arbitrary Choice 1: let arg[0] define the stick variable nd restick all others that have a conflict
            stick_expr = in_device_coords[0][-1]
            assert stick_expr != 0, "Expected arg 0 to have non-zero stick indexing expression"
            for arg_i, (ic, idc, arg) in enumerate(
                zip(in_coords[1:], in_device_coords[1:], args[1:]), start=1
            ):
                if idc[-1] != stick_expr:
                    schedule_restickify(n, arg, arg_i, stick_expr, ic, restick_needed)

        # If the indexing and device element size are identical
        # across all inputs and the output we can just propagate the device layout.
        can_use_same_layout = True
        for arg, arg_coors in zip(args, in_coords):
            if (
                arg_coors != out_coords
                or arg.dep.index != output_dep.index
                or not same_device_size(arg.layout.dtype, output.dtype)
            ):
                can_use_same_layout = False
                break

        if can_use_same_layout:
            template_stl = args[0].layout.device_layout
            stl = SpyreTensorLayout(
                template_stl.device_size,
                template_stl.dim_map,
                template_stl.stride_map,
                get_device_dtype(output.dtype),
            )
        else:
            # Use row major adjusted to put stick dimension last
            # and move all non-stick size one dimensions to the right to avoid tiling them.
            if len(stick_exprs) == 0:
                maybe_stick_dim = None
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
            stl = SpyreTensorLayout(output.size, output.stride, output.dtype, dim_order)

        result = FixedTiledLayout(
            output.device, output.dtype, output.size, output.stride, stl
        )

        if logger.isEnabledFor(logging.DEBUG):
            input_info = ", ".join(
                [f"in{i}:{list(arg.layout.size)}" for i, arg in enumerate(args)]
            )
            logger.debug(
                f"{op.__name__} layout: {input_info} -> out:{list(result.size)}, "
                f"device_size={list(result.device_layout.device_size)}"
            )

        return result


def reduction_layout(
    n: SchedulerNode, args: list[SchedNodeArg], restick_needed: dict
) -> FixedTiledLayout:
    red: Reduction = n.node.data
    output: FixedLayout = n.node.get_layout()
    output_dep = next(iter(n.read_writes.writes))
    if (
        red.reduction_type == MATMUL_REDUCTION_OP
        or red.reduction_type == BATCH_MATMUL_OP
    ):
        x = args[0]
        y = args[1]
        x_coords = host_coordinates(x.layout, x.dep)
        x_dev_coords = device_coordinates(x.layout, x.dep)
        y_coords = host_coordinates(y.layout, y.dep)
        y_dev_coords = device_coordinates(y.layout, y.dep)
        out_coords = host_coordinates(output, output_dep)
        x_stick_expr = x_dev_coords[-1]
        y_stick_expr = y_dev_coords[-1]
        x_stick_dim = matching_dim(x_coords, x_stick_expr)
        y_stick_dim = matching_dim(y_coords, y_stick_expr)
        if x_stick_dim is None or y_stick_dim is None:
            raise Unsupported(
                f"{red.reduction_type}: failed to map stick_dims to host coords"
            )

        # Hardware stick constraints (DF16):
        #   Input1 (x): stick on reduction_dim (the x coord that does NOT appear in output)
        #   Input2 (y): stick on generated_dim (the y coord that appears in output)
        #   Output:     stick on generated_dim
        # Restickify whichever input has its stick on the wrong dim.
        if matching_dim(out_coords, x_stick_expr) is not None:
            # x's stick is on a dim that appears in the output — move it to reduction_dim
            reduction_coord = next(
                c for c in x_coords if matching_dim(out_coords, c) is None
            )
            logger.warning(
                f"Injecting restickify on {red.reduction_type} x input to move stick to reduction_dim"
            )
            tl = schedule_restickify(n, x, 0, reduction_coord, x_coords, restick_needed)
            x_stick_expr = device_coordinates(tl, x.dep)[-1]
        # y's stick must be on the generated_dim, i.e. a dim that appears in the output.
        # If y_stick_expr doesn't appear in out_coords, y needs restickifying.
        if matching_dim(out_coords, y_stick_expr) is None:
            logger.warning(
                f"Injecting restickify on {red.reduction_type} y input to move stick to generated_dim"
            )
            # Target is the y coord that appears in the output (the generated_dim)
            generated_coord = next(
                c for c in y_coords if matching_dim(out_coords, c) is not None
            )
            tl = schedule_restickify(n, y, 1, generated_coord, y_coords, restick_needed)
            y_stick_expr = device_coordinates(tl, y.dep)[-1]

        out_stick_dim = matching_dim(out_coords, y_stick_expr)
        if out_stick_dim is None:
            raise Unsupported(
                f"{red.reduction_type}: failed to map output stick_dim to host coords {out_coords} {y_stick_expr}"
            )

        out_dims = len(output.size)
        out_dim_order = list(range(out_dims - 2))
        if out_stick_dim == out_dims - 1:
            out_dim_order = out_dim_order + [out_dims - 2, out_dims - 1]
        else:
            out_dim_order = out_dim_order + [out_dims - 1, out_dims - 2]
        stl = SpyreTensorLayout(output.size, output.stride, output.dtype, out_dim_order)
        return FixedTiledLayout(
            output.device, output.dtype, output.size, output.stride, stl
        )
    elif red.reduction_type == "exx2":
        x = args[0]
        x_coords = host_coordinates(x.layout, x.dep)
        x_dev_coords = device_coordinates(x.layout, x.dep)
        x_stick_expr = x_dev_coords[-1]
        x_stick_dim = matching_dim(x_coords, x_stick_expr)
        if x_stick_dim is None or x_stick_dim != len(x.layout.size) - 1:
            # TODO: Insert a restickify to enable the operation to be performed
            raise Unsupported(f"exx2: illegal device layout {x.layout}")

        dim_map = list(range(len(output.size))) + [-1]
        stl = SpyreTensorLayout(output.size, output.stride, output.dtype, dim_map)
        return FixedTiledLayout(
            output.device, output.dtype, output.size, output.stride, stl
        )
    else:
        x = args[0]
        x_coords = host_coordinates(x.layout, x.dep)
        x_dev_coords = device_coordinates(x.layout, x.dep)
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
        stl = SpyreTensorLayout(output.size, output.stride, output.dtype, out_dim_order)
        result = FixedTiledLayout(
            output.device, output.dtype, output.size, output.stride, stl
        )

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                f"{red.reduction_type} layout: in:{list(args[0].layout.size)} -> out:{list(result.size)}, "
                f"device_size={list(result.device_layout.device_size)}"
            )

        return result


def generic_layout(n: ExternKernelSchedulerNode) -> FixedTiledLayout:
    output: FixedLayout = n.node.get_layout()
    # Use the generic stick format
    stl = SpyreTensorLayout(output.size, output.dtype)
    return FixedTiledLayout(
        output.device, output.dtype, output.size, output.stride, stl
    )


def propagate_spyre_tensor_layouts(
    nodes: list[BaseSchedulerNode],
) -> tuple[list[BaseSchedulerNode], dict[BaseSchedulerNode, list[dict[str, Any]]]]:
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
                tb.data.data.layout = FixedTiledLayout(
                    ptl.device, ptl.dtype, ptl.size, ptl.stride, stl
                )

    # Nodes are in topological order (guarenteed by caller).
    # Visit them and use the inputs' FixedTiledLayouts and the operation being
    # performed by the node to convert its output FixedLayout to a FixedTiledLayout.
    restick_needed: dict[BaseSchedulerNode, list[dict[str, Any]]] = {}
    it = iter(nodes)
    for n in it:
        if isinstance(n, SchedulerNode) and isinstance(n.node, ComputedBuffer):
            n.node.decide_layout()
            if isinstance(n.node.data, Pointwise):
                output_layout = pointwise_layout(n, get_mem_deps(n), restick_needed)
                n.node.layout = output_layout
            elif isinstance(n.node.data, Reduction):
                output_layout = reduction_layout(n, get_mem_deps(n), restick_needed)
                n.node.layout = output_layout
            else:
                logger.warning(f"Warning: unhandled node type {type(n.node)}")
        elif isinstance(n, ExternKernelSchedulerNode):
            if isinstance(n.node, FallbackKernel):
                n = next(it, None)
                if not (
                    isinstance(n, ExternKernelSchedulerNode)
                    and isinstance(n.node, MultiOutput)
                ):
                    raise RuntimeError("FallbackKernel must be followed by MultiOutput")

                output_layout = generic_layout(n)
                n.node.layout = output_layout
            else:
                logger.warning(f"unhandled node type {type(n.node)}")
        elif isinstance(n, NopKernelSchedulerNode):
            output_layout = generic_layout(n)
            n.node.layout = output_layout
        else:
            logger.warning(f"unhandled scheduler node type {type(n)}")

    return nodes, restick_needed
