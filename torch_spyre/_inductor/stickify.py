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
import os

import sympy
import torch
from torch._inductor.dependencies import MemoryDep, ReadWrites
from torch.utils._ordered_set import OrderedSet
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


def is_sparse(stl: SpyreTensorLayout) -> bool:
    return stl.dim_map[-1] == -1


def device_layout_like(
    layout: FixedTiledLayout, dtype: torch.dtype
) -> SpyreTensorLayout:
    """
    Return a SpyreTensorLayout with the same tiling pattern as layout adjusted for the device_size of dtype.
    """
    if get_elem_in_stick(layout.dtype) == get_elem_in_stick(dtype):
        return SpyreTensorLayout(
            layout.device_layout.device_size,
            layout.device_layout.dim_map,
            layout.device_layout.stride_map,
            get_device_dtype(dtype),
        )
    else:
        adjusted_device_size = list(layout.device_layout.device_size)
        stick_dim_idx = -3 if len(adjusted_device_size) > 2 else -2
        old = get_elem_in_stick(layout.dtype)
        new = get_elem_in_stick(dtype)
        if old > new:
            scaling_factor = old / new
            adjusted_device_size[-1] = int(adjusted_device_size[-1] * scaling_factor)
            adjusted_device_size[stick_dim_idx] = int(
                (adjusted_device_size[stick_dim_idx] + scaling_factor - 1)
                / scaling_factor
            )
        else:
            scaling_factor = new / old
            adjusted_device_size[-1] = int(adjusted_device_size[-1] / scaling_factor)
            adjusted_device_size[stick_dim_idx] = int(
                adjusted_device_size[stick_dim_idx] * scaling_factor
            )
        return SpyreTensorLayout(
            adjusted_device_size,
            layout.device_layout.dim_map,
            layout.device_layout.stride_map,
            get_device_dtype(dtype),
        )


def pointwise_layout(n: SchedulerNode, args: list[SchedNodeArg]) -> FixedTiledLayout:
    pw: Pointwise = n.node.data
    output: FixedLayout = n.node.get_layout()
    output_dep = next(iter(n.read_writes.writes))
    origin_node = next(iter(pw.origins))
    op = origin_node.target

    print(f"MRA pointwise_layout: op={op} nargs={len(args)}")
    for arg in args:
        print("MRA: input arg: ", arg)

    if len(args) == 1:
        x = args[0]
        x_stl = x.layout.device_layout
        match op:
            case spyreop.slice.default:
                if not is_sparse(x_stl):
                    raise Unsupported("slice on non-sparse tensor")
                if len(x.layout.size) != 1:
                    raise Unsupported("slice on non 1-D tensor")
                stl = SpyreTensorLayout(output.size, output.dtype)
                return FixedTiledLayout(
                    output.device, output.dtype, output.size, output.stride, stl
                )

            case spyreop.swap.default:
                if not is_sparse(x_stl):
                    raise Unsupported("swap on non-sparse tensor")
                if len(x.layout.size) != 1:
                    raise Unsupported("swap on non 1-D tensor")
                stl = SpyreTensorLayout(output.size, output.dtype, [0, -1])
                return FixedTiledLayout(
                    output.device, output.dtype, output.size, output.stride, stl
                )

            case spyreop.restickify.default:
                # Produces correct answer!!!!!!
                # Either one of these passes the stick test!
                # 1,0,1 gets the right answer
                stl = SpyreTensorLayout([2, 256, 64], [1,0,1], [16384, 1, 256], get_device_dtype(output.dtype))
                
                # #  0,1,0 also passes stick test but gets wrong answer because backend uses dim-map to determine stick
                # stl = SpyreTensorLayout([2, 256, 64], [0,1,0], [16384, 1, 256], get_device_dtype(output.dtype))
                output_stride = [sympy.Integer(1), output.size[0]]

                return FixedTiledLayout(
                    output.device, output.dtype, output.size, output_stride, stl
                )

            case aten.clone.default:
                if is_sparse(x_stl):
                    # TODO: Determine whether we already support cloning a sparse tensor
                    #       or what functionality needs to be added to enable it.  Restickify?
                    raise Unsupported("clone on sparse tensor")

                # Clone is generated by an explicit `contiguous()`; on spyre that means use the default row major tiling.
                stl = SpyreTensorLayout(
                    output.size, output.dtype, list(range(len(output.size)))
                )
                return FixedTiledLayout(
                    output.device, output.dtype, output.size, output.stride, stl
                )

            case _:
                in_coords = host_coordinates(x.layout, x.dep)
                out_coords = host_coordinates(output, output_dep)
                if in_coords == out_coords and x.dep.index == output_dep.index:
                    # Input and output tensors are being accessed identically.
                    # We can simply propagate the device_layout.
                    stl = device_layout_like(x.layout, output.dtype)
                else:
                    # TODO: This needs further work
                    # Use row major adjusted to put stick dimension last and any
                    # non-stick size one dimensions in the output to the interior
                    # to avoid tiling them.
                    in_device_coords = device_coordinates(x.layout, x.dep)
                    if is_sparse(x_stl):
                        raise Unsupported("TODO: unary op with view on sparse tensor")
                    stick_expr = in_device_coords[-1]
                    maybe_stick_dim = matching_dim(out_coords, stick_expr)
                    out_stick_dim = -1 if maybe_stick_dim is None else maybe_stick_dim
                    dim_order = [
                        d
                        for d in range(len(output.size))
                        if d != out_stick_dim and out_coords[d] != 0
                    ]
                    dim_order += [
                        d for d in range(len(output.size)) if out_coords[d] == 0
                    ]
                    dim_order += [out_stick_dim]
                    stl = SpyreTensorLayout(output.size, output.dtype, dim_order)

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
        print ()
        print ("MRA: ---------------- Calling host_coordinates for inputs ---------------")
        in_coords = [host_coordinates(arg.layout, arg.dep) for arg in args]
        print()
        print ("MRA: ----------- Calling in_device_coords for inputs --------------- ")
        in_device_coords = [device_coordinates(arg.layout, arg.dep) for arg in args]
        print ()
        print ("MRA: ---------- Calling host_coordinates for output -----------")
        out_coords = host_coordinates(output, output_dep)


        print ("MRA: in_coords: ", in_coords)
        print ("MRA: in_device_coords: ", in_device_coords)
        print ("MRA: out_coords: ", out_coords)

        

        # Stick compatability check.
        # For all tensors whose stick dimension is being iterated over,
        # the indexing expression must be identical.
        stick_exprs = set()
        for idc in in_device_coords:
            if idc[-1] != 0:
                stick_exprs.add(idc[-1])

        print ("MRA: stick_exprs: ", stick_exprs)

        if len(stick_exprs) > 1:
            # TODO: This is a legal PyTorch operation that we cannot execute without inserting restickify operations.

            print ("ERROR: Spyre limitation: pointwise op with nonuniform stick indexing: {stick_exprs}")
            print_node(n)

            raise Unsupported(
                f"Spyre limitation: pointwise op with nonuniform stick indexing: {stick_exprs}"
            )
            # print ("ERROR: Spyre limitation: pointwise op with nonuniform stick indexing: ", stick_exprs)

        # See if the indexing across all inputs and the output is identical.
        can_use_same_layout = True
        for arg, arg_coors in zip(args, in_coords):
            if arg_coors != out_coords or arg.dep.index != output_dep.index:
                can_use_same_layout = False
                break

        if can_use_same_layout:
            # Identical indexing. Therefore no views or broadcasts. Just propagate layout
            stl = device_layout_like(args[0].layout, output.dtype)
            out_stride = output.stride
            input_stride = list(args[0].layout.stride)
            alloc_stride = input_stride if input_stride != list(out_stride) else None
            print(f"MRA can_use_same_layout: out_stride={out_stride} input_stride={input_stride} alloc_stride={alloc_stride}")
            result = FixedTiledLayout(
                output.device, output.dtype, output.size, out_stride, stl,
                alloc_stride=alloc_stride,
                alloc_device_layout=stl if alloc_stride is not None else None,
            )
        else:
            # Use row major adjusted to put stick dimension last
            # TODO: Should we also push size 1 dims to the interior here like in unary above??
            if len(stick_exprs) == 0:
                raise Unsupported(
                    "pointwise op with views/broadcasts without stick dim"
                )
            stick_expr = next(iter(stick_exprs))
            maybe_stick_dim = matching_dim(out_coords, stick_expr)
            out_stick_dim = -1 if maybe_stick_dim is None else maybe_stick_dim
            dim_order = [d for d in range(len(output.size)) if d != out_stick_dim]
            dim_order += [out_stick_dim]
            if os.environ.get("SPYRE_COLMAJOR_OUTPUT", "0") == "1":
                # Col-major output: stride_map=[16384,1,256], host stride [1,NROWS].
                # The loop recomputes decide_layout so store index is col-major too.
                s0 = output.size[0]
                s1 = output.size[1]
                stl = SpyreTensorLayout(
                    [s1 // 64, s0, 64],
                    [1, 0, 1],
                    [s0 * 64, sympy.Integer(1), s0],
                    get_device_dtype(output.dtype),
                )
                out_stride = [sympy.Integer(1), s0]
            else:
                stl = SpyreTensorLayout(output.size, output.dtype, dim_order)
                out_stride = output.stride
            result = FixedTiledLayout(
                output.device, output.dtype, output.size, out_stride, stl
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


def reduction_layout(n: SchedulerNode, args: list[SchedNodeArg]) -> FixedTiledLayout:
    red: Reduction = n.node.data
    output: FixedLayout = n.node.get_layout()
    output_dep = next(iter(n.read_writes.writes))
    if (
        red.reduction_type == MATMUL_REDUCTION_OP
        or red.reduction_type == BATCH_MATMUL_OP
    ):
        x = args[0]
        y = args[1]
        x_stl = x.layout.device_layout
        y_stl = y.layout.device_layout
        if is_sparse(x_stl) or is_sparse(y_stl):
            raise Unsupported(f"{red.reduction_type} on sparse tensor {x_stl} {y_stl}")

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

        if (
            x_stick_dim != len(x.layout.size) - 1
            or y_stick_dim != len(y.layout.size) - 1
        ):
            # TODO: This is a legal PyTorch operation that we cannot execute without inserting restickify operations.
            raise Unsupported(
                f"Spyre limitation: {red.reduction_type} requires restickify"
            )
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
        stl = SpyreTensorLayout(output.size, output.dtype, out_dim_order)
        return FixedTiledLayout(
            output.device, output.dtype, output.size, output.stride, stl
        )
    elif red.reduction_type == "exx2":
        x = args[0]
        x_stl = x.layout.device_layout
        if is_sparse(x_stl) or x_stl.host_stick_dim() != (len(x.layout.size) - 1):
            raise Unsupported(f"exx2 unsupported layout {x_stl}")
        dim_map = list(range(len(output.size))) + [-1]
        stl = SpyreTensorLayout(output.size, output.dtype, dim_map)
        return FixedTiledLayout(
            output.device, output.dtype, output.size, output.stride, stl
        )
    else:
        x = args[0]
        x_stl = x.layout.device_layout

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
        stl = SpyreTensorLayout(output.size, output.dtype, out_dim_order)
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


def print_node(n):
    print("=== SchedulerNode ===")

    if hasattr(n, "_kernel"):
        print("Has kernel:", n._kernel)
    else:
        print("Has Kernel:  NO")

    print("reads:", n.read_writes.reads)
    for dep in n.read_writes.reads:
        print("Read Dep Name", repr(dep.name))

    print("writes:", n.read_writes.writes)
    for dep in n.read_writes.writes:
        print("Write Dep Name", repr(dep.name))

    if hasattr(n, "min_order"):
        print("min_order:", n.min_order)
    else:
        print("No min order field")
    if hasattr(n, "max_order"):
        print("max_order:", n.max_order)
    else:
        print("No max order field")
    print("ancestors:", n.ancestors)
    print("unmet_dependencies:", n.unmet_dependencies)
    print()

    print("=== Buffer node: ===")
    buffer = n.node
    if hasattr(buffer, "operation_name"):
        print("Node operation name", buffer.get_operation_name())
    else:
        print("Node operation name is missing")
    print("node.node:", buffer)

    # print("--- CLOSURES ---")
    # fn = buffer.data.inner_fn
    # print("FN:", fn)
    # print("FREEVARS:", fn.__code__.co_freevars)
    # print("CLOSURE:", fn.__closure__)
    # for cell in fn.__closure__ or []:
    #     print("CELL:", cell.cell_contents, type(cell.cell_contents))

    print("---------------------------------")
    print()


def propagate_spyre_tensor_layouts(
    nodes: list[BaseSchedulerNode],
) -> list[BaseSchedulerNode]:
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

    it = iter(nodes)
    for n in it:
        if isinstance(n, SchedulerNode) and isinstance(n.node, ComputedBuffer):
            print ("ABOUT TO DECIDE LAYOUT")
            n.node.decide_layout()
            if isinstance(n.node.data, Pointwise):
                old_stride = list(n.node.layout.stride)
                output_layout = pointwise_layout(n, get_mem_deps(n))
                n.node.layout = output_layout
                if (os.environ.get("SPYRE_COLMAJOR_OUTPUT", "0") == "1"
                        and list(output_layout.stride) != old_stride):
                    # Re-run decide_layout with col-major stride so the kernel's
                    # store index and stride_map are consistent.
                    from torch._inductor.ir import FlexibleLayout
                    flex = FlexibleLayout(
                        output_layout.device, output_layout.dtype, output_layout.size
                    )
                    n.node.layout = flex
                    n.node.freeze_layout_with_stride_order(
                        sorted(range(len(output_layout.stride)),
                               key=lambda i: output_layout.stride[i])
                    )
                    ComputedBuffer.get_default_sizes_body.clear_cache(n.node)
                    n.recompute_size_and_body()
                    n.node.layout = output_layout
            elif isinstance(n.node.data, Reduction):
                output_layout = reduction_layout(n, get_mem_deps(n))
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
        print_node(n)

    return nodes
