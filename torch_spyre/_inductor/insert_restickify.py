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
from torch._inductor.ir import ComputedBuffer, TensorBox
from torch._inductor.ops_handler import WrapperHandler
from torch._inductor.scheduler import (
    BaseSchedulerNode,
)
from torch._inductor.virtualized import V

from torch.utils._ordered_set import OrderedSet

logger = get_inductor_logger("insert_restickify")


class NameSwapHandler(WrapperHandler):
    def __init__(self, inner, name_map: dict[str, str]):
        super().__init__(inner)
        self._name_map = name_map

    def load(self, name, index):
        return super().load(self._name_map.get(name, name), index)


def _create_restickify_node(
    restick_arg_info: dict, n: BaseSchedulerNode, scheduler
) -> tuple[str, object]:
    """
    Insert one restickify scheduler node for a given incompatible arg specified in restick_arg_info.
    Restickify nodes will comply with the layout specified in restick_arg_infos.

    Returns (old_buffer_name, new_scheduler_node).
    """
    mem_dep = list(n.read_writes.reads)[restick_arg_info["arg_index"]]
    arg_name = mem_dep.name

    graph_lowering = V.graph
    fx_graph = graph_lowering.graph

    # View ops (e.g. permute) lower to ReinterpretView with no buffer name, so
    # they are absent from name_to_users and missing from env. Rebuild from
    # name_to_users so fetch_args_kwargs_from_env can resolve fx_arg_node.
    env = {}
    for tbs in graph_lowering.name_to_users.values():
        for tb in tbs:
            tb_fx_node = list(tb.data.origins)[0]
            env[tb_fx_node] = tb
    graph_lowering.env.update(env)

    # Find the FX node whose buffer name matches arg_name.
    # Using arg_index to index origin_node.args is fragile since FX args include
    # scalars/constants that don't appear in read_writes.reads.
    fx_arg_node = next(
        fx_node
        for fx_node, tb in graph_lowering.env.items()
        if isinstance(fx_node, torch.fx.Node)
        and isinstance(tb, TensorBox)
        and tb.get_name() == arg_name
    )
    # Insert before the first computation node so the restickify node
    # precedes all potential consumers in the graph node list.
    first_compute_node = next(n2 for n2 in fx_graph.nodes if n2.op != "placeholder")
    with fx_graph.inserting_before(first_compute_node):
        restick_fx_node = fx_graph.create_node(
            "call_function", torch.ops.spyre.restickify, (fx_arg_node,)
        )
    # Lower via run_node — handles buffer registration automatically
    restick_tb = graph_lowering.run_node(restick_fx_node)
    restick_buff = restick_tb.data.data  # TensorBox -> StorageBox -> ComputedBuffer
    assert isinstance(restick_buff, ComputedBuffer), (
        f"Expected ComputedBuffer, got {type(restick_buff).__name__}"
    )
    # restick_fx_node is synthetically created post-lowering and has no ATen metadata.
    # Restickify runs before any view op on the arg, so there is no ATen op to
    # attribute it to. Leave origins as just restick_fx_node — the comment will show [].
    restick_buff.origins = OrderedSet([restick_fx_node])
    graph_lowering.env[restick_fx_node] = restick_tb

    restick_sn = scheduler.create_scheduler_node(restick_buff)
    restick_sn.node.layout = restick_arg_info["target_layout"]
    ComputedBuffer.get_default_sizes_body.clear_cache(restick_sn.node)
    restick_sn._compute_attrs()

    return arg_name, restick_sn


def insert_restickify_on_node_inputs(
    n: BaseSchedulerNode, restick_infos: list[dict], scheduler
) -> None:
    """Create a restickify node for each incompatible input arg of node n.
    Use NameSwapHandler to patch n's inner_fn to use the new buffer names instead of
    original input buffers.
    """
    name_map = {}

    for restick_arg_info in restick_infos:
        old_name, restick_sn = _create_restickify_node(restick_arg_info, n, scheduler)
        name_map[old_name] = restick_sn.node.name

        for buf in restick_sn.get_outputs():
            scheduler.name_to_buf[buf.get_name()] = buf
        scheduler.nodes.append(restick_sn)

    # Patch inner_fn once with the full name_map covering all restickified args
    orig_inner = n.node.data.inner_fn

    def new_inner_fn(*args, _map=name_map, _orig_inner=orig_inner):
        with V.set_ops_handler(NameSwapHandler(V.ops, _map)):
            return _orig_inner(*args)

    object.__setattr__(n.node.data, "inner_fn", new_inner_fn)

    # Rebuilding ComputedBuffer around patched inner_fn to reduce
    # number of internal datastructures that need to be hacked
    new_consumer_buffer = ComputedBuffer(
        name=n.node.name,
        layout=n.node.layout,
        data=n.node.data,
        _split_size=n.node._split_size,
        _original_inner_fn=n.node._original_inner_fn,
        _original_ranges=n.node._original_ranges,
        _original_reduction_ranges=n.node._original_reduction_ranges,
    )
    new_consumer_buffer.operation_name = n.node.operation_name
    new_consumer_buffer.origins = n.node.origins
    n.node = new_consumer_buffer

    # Recompute internal metadata including read/write dependencies based on new inner_fn
    ComputedBuffer.get_default_sizes_body.clear_cache(n.node)
    n._compute_attrs()


def insert_restickify(
    nodes: list[BaseSchedulerNode], restick_needed: dict
) -> list[BaseSchedulerNode]:
    """
    Insert restickify(ies) before all nodes in restick_needed.
    """
    scheduler = V.graph.scheduler
    for n in list(nodes):  # copy because loop updates scheduler.nodes
        if n in restick_needed:
            insert_restickify_on_node_inputs(n, restick_needed[n], scheduler)

    scheduler.compute_dependencies()
    scheduler.name_to_fused_node = {n.get_name(): n for n in scheduler.nodes}

    # Can maybe skip sorting if new_sn was inserted in the correct spot to
    # maintain topological order, but safer to just re-sort
    sorted_order = scheduler._topological_sort_nodes()
    scheduler.nodes = [node for group in sorted_order for node in group]
    scheduler.compute_ancestors()

    return scheduler.nodes
