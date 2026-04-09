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
    NodeUser,
)
from torch._inductor.virtualized import V

from torch.utils._ordered_set import OrderedSet

logger = get_inductor_logger("insert_restickify")


class NameSwapHandler(WrapperHandler):
    """
    Wrapper to patch a node's inner_fn to use new buffer names after inserting
    nodes upstream that change the input buffers.
    """

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

    # View ops (e.g. permute) lower to ReinterpretView with no buffer name and
    # are absent from env. Patch env from name_to_users so the search below can
    # resolve them.
    env = {}
    for tbs in graph_lowering.name_to_users.values():
        for tb in tbs:
            tb_fx_node = list(tb.data.origins)[0]
            env[tb_fx_node] = tb
    graph_lowering.env.update(env)

    # Search by buffer name rather than arg_index: FX args include scalars and
    # constants that don't appear in read_writes.reads, making index-based lookup
    # unreliable.
    fx_arg_node = next(
        fx_node
        for fx_node, tb in graph_lowering.env.items()
        if isinstance(fx_node, torch.fx.Node)
        and isinstance(tb, TensorBox)
        and tb.get_name() == arg_name
    )
    # Insert at a valid position in the FX graph; scheduling order is determined
    # by the topological sort in insert_restickify, not by position here.
    first_compute_node = next(n2 for n2 in fx_graph.nodes if n2.op != "placeholder")
    with fx_graph.inserting_before(first_compute_node):
        restick_fx_node = fx_graph.create_node(
            "call_function", torch.ops.spyre.restickify, (fx_arg_node,)
        )
    # Lower the FX node; run_node lowers and registers the output buffer in graph.buffers.
    restick_tb = graph_lowering.run_node(restick_fx_node)
    restick_buff = restick_tb.data.data  # TensorBox -> StorageBox -> ComputedBuffer
    assert isinstance(restick_buff, ComputedBuffer), (
        f"Expected ComputedBuffer, got {type(restick_buff).__name__}"
    )
    # Synthetic node with no corresponding ATen op; set origins to the synthetic
    # FX node so code that expects non-empty origins doesn't crash.
    restick_buff.origins = OrderedSet([restick_fx_node])
    graph_lowering.env[restick_fx_node] = restick_tb

    restick_sn = scheduler.create_scheduler_node(restick_buff)
    restick_sn.node.layout = restick_arg_info["target_layout"]
    ComputedBuffer.get_default_sizes_body.clear_cache(restick_sn.node)
    restick_sn._compute_attrs()
    
    return arg_name, restick_sn


def insert_restickify_on_node_inputs(
    n: BaseSchedulerNode,
    resticks_needed: list[dict],
    scheduler,
    global_name_map: dict[str, str],
) -> None:
    """Create a restickify node for each incompatible input arg of node n."""
    name_map = {}

    for restick_arg_info in resticks_needed:
        old_name, restick_sn = _create_restickify_node(restick_arg_info, n, scheduler)
        name_map[old_name] = restick_sn.node.name

        for buf in restick_sn.get_outputs():
            scheduler.name_to_buf[buf.get_name()] = buf
        scheduler.nodes.append(restick_sn)
        global_name_map[old_name] = restick_sn.node.name

        # Keep buf.users consistent so downstream passes (fusion, memory planning)
        # see the correct graph structure without needing to re-run
        # compute_dependencies().
        #
        # Before restickify:  input_buf -> n
        # After restickify:   input_buf -> restick_sn -> n
        #
        # 1. input_buf: replace n as user with restick_sn
        if old_name in scheduler.name_to_buf:
            input_buf = scheduler.name_to_buf[old_name]
            input_buf.users = [
                u for u in input_buf.users if u.node is not n
            ]
            input_buf.users.append(NodeUser(restick_sn, can_inplace=False))
        # 2. restick output buf: n is its sole user
        for restick_out_buf in restick_sn.get_outputs():
            restick_out_buf.users = [NodeUser(n, can_inplace=False)]

    # Patch inner_fn once with the full name_map covering all restickified args
    orig_inner = n.node.data.inner_fn

    def new_inner_fn(*args, _map=name_map, _orig_inner=orig_inner):
        with V.set_ops_handler(NameSwapHandler(V.ops, _map)):
            return _orig_inner(*args)

    object.__setattr__(n.node.data, "inner_fn", new_inner_fn)

    # Reconstruct ComputedBuffer so internal caches see the patched inner_fn.
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


def insert_restickify(nodes: list[BaseSchedulerNode]) -> list[BaseSchedulerNode]:
    """
    Insert restickify(ies) before all nodes in restickify_plan
    """

    restickify_plan = getattr(V.graph, "restickify_plan", {})
    if not restickify_plan:
        return nodes

    scheduler = V.graph.scheduler

    # name_map accumulated across all restickify insertions: old_buf -> restick_buf
    global_name_map: dict[str, str] = {}
    for n in list(nodes):  # copy because loop updates scheduler.nodes
        if n in restickify_plan:
            insert_restickify_on_node_inputs(n, restickify_plan[n], scheduler, global_name_map)

    # Do NOT call scheduler.compute_dependencies() again. It already ran in
    # Scheduler.__init__ and left unmet_dependencies with mutation-renamed buffer
    # names (e.g. buf18 already renamed to buf20 happens for Granite). Re-running it 
    # applies those renames again, making nodes appear to read their own 
    # output → topo-sort cycle. Instead, update unmet_dependencies incrementally 
    # via prune_deps() (for new restickify nodes) and manual dep-swapping (for 
    # patched consumer nodes).
    for node in scheduler.nodes:
        node.prune_deps()

    # Patch unmet_dependencies on consumer nodes: swap old read deps for restickify deps.
    for n in list(nodes):
        if n in restickify_plan:
            from torch._inductor.dependencies import MemoryDep, StarDep
            new_unmet = OrderedSet()
            for dep in n.unmet_dependencies:
                if dep.name in global_name_map:
                    # Replace with a dep on the restickify buffer
                    new_name = global_name_map[dep.name]
                    if isinstance(dep, MemoryDep):
                        new_unmet.add(MemoryDep(new_name, dep.index, dep.var_names, dep.size, dep.mode))
                    else:
                        new_unmet.add(StarDep(new_name))
                else:
                    new_unmet.add(dep)
            n.unmet_dependencies = new_unmet

    # Update name_to_fused_node to include the newly inserted restickify nodes.
    # The scheduler set this during __init__ before our pass ran, so the new nodes
    # are absent and _get_unmet_dep_nodes will KeyError without this update.
    scheduler.name_to_fused_node.update(
        {n.get_name(): n for n in scheduler.nodes}
    )

    sorted_order = scheduler._topological_sort_nodes()
    scheduler.nodes = [node for group in sorted_order for node in group]
    scheduler.compute_ancestors()

    return scheduler.nodes
