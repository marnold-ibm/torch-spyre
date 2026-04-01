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
from torch._inductor.ir import ComputedBuffer, StorageBox, TensorBox
from torch._inductor.ops_handler import WrapperHandler
from torch._inductor.scheduler import (
    BaseSchedulerNode,
    SchedulerNode,
)
from torch._inductor.virtualized import V

from torch.utils._ordered_set import OrderedSet
from .pass_utils import dump_ir

logger = get_inductor_logger("insert_nodes")


class NameSwapHandler(WrapperHandler):
    def __init__(self, inner, name_map: dict[str, str]):
        super().__init__(inner)
        self._name_map = name_map

    def load(self, name, index):
        return super().load(self._name_map.get(name, name), index)


def _create_restickify_node(
    permute_info: dict, n: BaseSchedulerNode, scheduler
) -> tuple[str, object]:
    """
    Insert one restickify scheduler node for a given incompatible arg specified in permute_info.
    Restickify nodes will comply with the layout specified in permute_infos.
    
    Returns (old_buffer_name, new_scheduler_node).
    """
    from torch_spyre._inductor.stickify import propagate_spyre_tensor_layouts  # noqa: PLC0415

    mem_dep = list(n.read_writes.reads)[permute_info["arg_index"]]
    arg_name = mem_dep.name

    graph_lowering = V.graph
    fx_graph = graph_lowering.graph

    # Build env so run_node can find the TensorBox for arg_name
    env = {}
    for tbs in graph_lowering.name_to_users.values():
        for tb in tbs:
            tb_fx_node = list(tb.data.origins)[0]
            env[tb_fx_node] = tb
    graph_lowering.env.update(env)

    # Find the FX node that feeds arg_index and needs permute
    fx_arg_node = n.node.data.origin_node.args[permute_info["arg_index"]]
    fx_non_placeholder = next(n2 for n2 in fx_graph.nodes if n2.op != "placeholder")
    fx_graph.inserting_before(fx_non_placeholder)
    new_fx_node = fx_graph.create_node(
        "call_function", torch.ops.spyre.restickify, (fx_arg_node, [0, 1])
    )
    graph_lowering.orig_gm.recompile()

    # Lower via run_node — handles buffer registration automatically
    new_tb = graph_lowering.run_node(new_fx_node)
    new_buff = new_tb.data.data  # TensorBox -> StorageBox -> ComputedBuffer
    graph_lowering.env[new_fx_node] = new_tb

    new_sn = scheduler.create_scheduler_node(new_buff)
    new_sn.node.layout = permute_info["target_layout"]
    ComputedBuffer.get_default_sizes_body.clear_cache(new_sn.node)
    new_sn._compute_attrs()

    return arg_name, new_sn


def _apply_permutes_to_node(
    n: BaseSchedulerNode, permute_infos: list[dict], scheduler
) -> None:
    """Create a restickify node for each incompatible input arg of node n.  
    Use NameSwapHandler to patch n's inner_fn to use the new buffer names instead of 
    original input buffers.  
    """
    name_map = {}

    for permute_info in permute_infos:
        old_name, new_sn = _create_restickify_node(permute_info, n, scheduler)
        name_map[old_name] = new_sn.node.name

        for buf in new_sn.get_outputs():
            scheduler.name_to_buf[buf.get_name()] = buf
        scheduler.nodes.append(new_sn)

    # Patch inner_fn once with the full name_map covering all permuted args
    orig_inner = n.node.data.inner_fn
    def new_inner_fn(index, _map=name_map, _orig_inner=orig_inner):
        with V.set_ops_handler(NameSwapHandler(V.ops, _map)):
            return _orig_inner(index)

    object.__setattr__(n.node.data, "inner_fn", new_inner_fn)

    # Rebuild ComputedBuffer to minimise internal datastructures that need to be hacked
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

    # Recompute internal metadata including read/write dependencies based on new inner_fn
    ComputedBuffer.get_default_sizes_body.clear_cache(n.node)
    n._compute_attrs()


def insert_permutes(
    nodes: list[BaseSchedulerNode], permute_needed: dict
) -> list[BaseSchedulerNode]:
    """
    Insert restickify nodes for all nodes in permute_needed. 
    Returns the new list of nodes including the inserted restickify nodes.
    """
    if not permute_needed:
        return nodes

    scheduler = V.graph.scheduler
    for n in list(nodes):  # copy because loop updates scheduler.nodes
        if n in permute_needed:
            _apply_permutes_to_node(n, permute_needed[n], scheduler)

    scheduler.compute_dependencies()
    scheduler.name_to_fused_node = {n.get_name(): n for n in scheduler.nodes}

    # Can maybe skip sorting if new_sn was inserted in the correct spot to
    # maintain topological order, but safer to just re-sort
    sorted_order = scheduler._topological_sort_nodes()
    scheduler.nodes = [node for group in sorted_order for node in group]
    scheduler.compute_ancestors()

    dump_ir(scheduler.nodes, "insert_permutes")

    return scheduler.nodes