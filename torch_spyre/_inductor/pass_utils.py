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

from typing import NamedTuple


import sympy
from torch._inductor.ir import FixedLayout, Pointwise, Reduction
from torch._inductor.scheduler import SchedulerNode
from torch._inductor.dependencies import MemoryDep
from torch._inductor.virtualized import V
from torch_spyre._inductor.errors import Unsupported

from .ir import FixedTiledLayout
from .views import compute_coordinates


class SchedNodeArg(NamedTuple):
    dep: MemoryDep
    layout: FixedTiledLayout


def get_mem_deps(n: SchedulerNode) -> list[SchedNodeArg]:
    res: list[SchedNodeArg] = []
    for arg in n.read_writes.reads:
        if isinstance(arg, MemoryDep):
            buf = V.graph.get_buffer(arg.name)
            layout = buf.get_layout()
            if not isinstance(layout, FixedTiledLayout):
                raise RuntimeError(f"{buf} does not have FixedTiledLayout")
            res.append(SchedNodeArg(arg, layout))
    return res


def host_coordinates(layout: FixedLayout, dep: MemoryDep) -> list[sympy.Expr]:
    return compute_coordinates(layout.size, layout.stride, dep.ranges, dep.index)


def device_coordinates(layout: FixedTiledLayout, dep: MemoryDep) -> list[sympy.Expr]:
    return compute_coordinates(
        layout.device_layout.device_size,
        layout.device_layout.stride_map,
        dep.ranges,
        dep.index,
    )


def iteration_space(n: SchedulerNode) -> dict[sympy.Symbol, sympy.Expr]:
    if isinstance(n.node.data, Pointwise):
        # The iteration space of a Pointwise is that of its output
        return next(iter(n.read_writes.writes)).ranges.copy()
    elif isinstance(n.node.data, Reduction):
        # The iteration space of a Reduction is that of its input
        return next(iter(n.read_writes.reads)).ranges.copy()
    else:
        raise Unsupported("Unexpected node type")
    

    
# Debugging methods while developing
def dump_node(n, print_closures: bool):
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

    if print_closures:
        print("--- CLOSURES ---")
        fn = buffer.data.inner_fn
        print("FN:", fn)
        print("FREEVARS:", fn.__code__.co_freevars)
        print("CLOSURE:", fn.__closure__)

        for cell in fn.__closure__ or []:
            print("CELL:", cell.cell_contents, type(cell.cell_contents))

    print("---------------------------------")
    print()


def dump_ir(nodes: list[SchedulerNode], label: str = "", print_closures: bool = False):
    print(f"=============== DUMPING FULL IR from {label} ==================")
    print()

    print("== Graph Inputs ==")
    print([x for x in V.graph.graph_inputs])

    print("== Registered Buffers ==")
    for buf in V.graph.buffers:
        print(buf.name)

    print("== IR NODES ==")

    for n in nodes:
        if isinstance(n, SchedulerNode):
            dump_node(n, print_closures)
