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
from torch._inductor.ir import ComputedBuffer, Pointwise, ops
from torch._inductor.scheduler import (
    BaseSchedulerNode,
    SchedulerNode,
)
from torch._inductor.virtualized import V

from torch.utils._ordered_set import OrderedSet
from torch.fx import Node

logger = get_inductor_logger("insert_nodes")

aten = torch.ops.aten

from torch._inductor.ops_handler import WrapperHandler

class NameSwapHandler(WrapperHandler):
    def __init__(self, inner, old_name: str, new_name: str):
        super().__init__(inner)
        self._old = old_name
        self._new = new_name

    def load(self, name, index):
        return super().load(self._new if name == self._old else name, index)

# Temporary debugging methods while developing
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

    print("--- CLOSURES ---")
    fn = buffer.data.inner_fn
    print("FN:", fn)
    print("FREEVARS:", fn.__code__.co_freevars)
    print("CLOSURE:", fn.__closure__)

    for cell in fn.__closure__ or []:
        print("CELL:", cell.cell_contents, type(cell.cell_contents))

    print("---------------------------------")
    print()


def dump_ir(nodes: list[BaseSchedulerNode]):
    print("=============== DUMPING FULL IR ==================")
    print()

    print("== Graph Inputs ==")
    print([x for x in V.graph.graph_inputs])

    print("== Registered Buffers ==")
    for buf in V.graph.buffers:
        print(buf.name)

    print("== IR NODES ==")

    for n in nodes:
        if isinstance(n, SchedulerNode):
            print_node(n)


def insert_permutes(
    nodes: list[BaseSchedulerNode], permute_needed: dict
) -> list[BaseSchedulerNode]:
    
    if not permute_needed:
        return nodes

    graph = V.graph
    scheduler = V.graph.scheduler
    for n in list(nodes):  # copy because loop updates scheduler.nodes

        if n in permute_needed:
            permute_info = permute_needed[n]

            # Get the arg info for the input we need to permute
            mem_dep = list(n.read_writes.reads)[permute_info["arg_index"]]
            arg_name = mem_dep.name
            arg_buff = V.graph.get_buffer(arg_name).data.data

            # Create node to do the permute
            def inner_fn(index, _ab=arg_buff):
                i0, i1 = index
                tmp0 = ops.load(_ab.name, i0 + 64 * i1)
                return tmp0

            pw_node = Pointwise(
                device=torch.device("spyre"),
                dtype=torch.float16,
                inner_fn=inner_fn,
                ranges=[64, 64],
            )

            new_origin_node = Node(
                graph=V.graph.graph,
                name="clone_default",
                op="call_function",
                target=aten.clone.default,
                args=(),
                kwargs={},
            )

            # No constructor for these and object is frozen
            object.__setattr__(pw_node, "origin_node", new_origin_node)
            object.__setattr__(pw_node, "origins", OrderedSet([new_origin_node]))
            if hasattr(n.node.data, "stack_traces"):
                object.__setattr__(pw_node, "stack_traces", n.node.data.stack_traces)

            # Create the output buffer
            new_buff = ComputedBuffer(
                name="buf0000_injected",  # Not actually used: renamed by register_buffer for some reason
                layout=permute_info["target_layout"],
                data=pw_node,
                _split_size=None,
                _original_inner_fn=None,
                _original_ranges=None,
                _original_reduction_ranges=None,
            )
            _ = graph.register_operation(new_buff)

            registered_buff = graph.register_buffer(new_buff)
            if registered_buff != new_buff.name:
                new_buff.name = registered_buff

            new_sn = scheduler.create_scheduler_node(new_buff)

            # ===================================================
            # Now create a wrapper that replaces reads of the modified arg with the new buffer
            orig_inner = n.node.data.inner_fn
            def new_inner_fn(index, _old=arg_buff.name, _new=new_buff.name, _orig=orig_inner):
                with V.set_ops_handler(NameSwapHandler(V.ops, _old, _new)):
                    return _orig(index)
            object.__setattr__(n.node.data, "inner_fn", new_inner_fn)

            # Must create new ComputedBuffer to update internal Scheduler metadata
            # (Or figure out what it needs and update it)
            new_computed_buffer = ComputedBuffer(
                name=n.node.name,
                layout=n.node.layout,
                data=n.node.data,
                _split_size=n.node._split_size,
                _original_inner_fn=n.node._original_inner_fn,
                _original_ranges=n.node._original_ranges,
                _original_reduction_ranges=n.node._original_reduction_ranges,
            )
            new_computed_buffer.operation_name = n.node.operation_name
            n.node = new_computed_buffer

            # Recomputes internal metadata, including read/write dependencies based on new inner_fn
            n._compute_attrs()

            new_buff_writes = list(new_sn.read_writes.writes)
            orig_reads = list(n.read_writes.reads)
            n.read_writes.reads = OrderedSet(
                [
                    new_buff_writes[0] if i == permute_info["arg_index"] else read
                    for i, read in enumerate(orig_reads)
                ]
            )

            for buf in new_sn.get_outputs():
                scheduler.name_to_buf[buf.get_name()] = buf
            scheduler.nodes.append(new_sn)


    scheduler.compute_dependencies()
    scheduler.name_to_fused_node = {n.get_name(): n for n in scheduler.nodes}
    # Can skip sorting if new_sn was inserted in the correct spot to
    # maintain topological order, but safer to just re-sort
    sorted_order = scheduler._topological_sort_nodes()
    scheduler.nodes = [node for group in sorted_order for node in group]
    scheduler.compute_ancestors()

    #dump_ir(scheduler.nodes)

    return scheduler.nodes
