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


"""Join-aware pre-pass for AllSameNode layout candidates.

greedy_local_min_cost/beam_global_min_cost each commit an op's output layout
using only that op's own local input costs. Two independent sibling ops that
both feed a shared multi-input AllSameNode join (e.g. torch.maximum(M,
block_max)) can each commit to a candidate that is individually cheapest for
themselves but not mutually stick-compatible with each other at the join --
and no per-op-local cost function can see far enough ahead to avoid this.

This module runs as a pre-pass, before optimize_restickify_locations, and
resolves such conflicts by jointly searching each cluster of siblings/joins
for the assignment that minimizes total cost, then narrowing the affected
buffers' own candidate lists so the later optimizer converges on that
assignment. See docs in the coarse_tile / restickify design notes for the
buf20/buf21/buf22 flash-attention case this was built to fix.
"""

import itertools
import math
from collections import defaultdict

from torch._inductor.dependencies import MemoryDep
from torch._inductor.graph import GraphLowering
from torch._inductor.virtualized import V

from .logging_utils import get_inductor_logger
from .optimize_restickify import AllSameNode

logger = get_inductor_logger("resolve_join_clusters")

INF = math.inf

# Skip clusters whose Cartesian product of candidate assignments exceeds this
# bound. optimize_restickify_locations still runs afterward and raises its
# usual, detailed _no_feasible_layout_error if the cluster turns out to be
# genuinely infeasible, so skipping is no worse than doing nothing here.
_MAX_COMBINATIONS = 4096


def _stl_key(stl):
    return (tuple(stl.device_size), tuple(stl.stride_map))


def _find_join_ops(operations: list) -> list:
    """Return every op whose restick_cost_fn is a multi-input AllSameNode."""
    joins = []
    for op in operations:
        cost_fn = getattr(op, "restick_cost_fn", None)
        if isinstance(cost_fn, AllSameNode) and len(cost_fn.edge_costs) > 1:
            joins.append(op)
    return joins


def _sibling_names(op) -> list[str]:
    return [
        dep.name for dep in op.get_read_writes().reads if isinstance(dep, MemoryDep)
    ]


def _build_clusters(join_ops: list) -> list[list]:
    """Union-find over join ops that share at least one input buffer.

    Two join ops belong in the same cluster if they share a sibling input --
    this keeps a buffer that feeds two different joins (e.g. a loop-carried
    value read by both the running-max update and a later correction term)
    in one cluster, so all of its constraints are considered together.
    """
    parent: dict[int, int] = {id(op): id(op) for op in join_ops}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    buf_to_join: dict[str, int] = {}
    for op in join_ops:
        for name in _sibling_names(op):
            if name in buf_to_join:
                union(id(op), buf_to_join[name])
            else:
                buf_to_join[name] = id(op)

    groups: dict[int, list] = defaultdict(list)
    for op in join_ops:
        groups[find(id(op))].append(op)
    return list(groups.values())


def _consumer_counts(operations: list) -> dict[str, int]:
    """Return {buf_name: number of ops that read it} over the whole graph.

    A buffer with more than one consumer can't have its candidate list
    truncated for just one consumer's benefit, so _walk_chain uses this to
    stop before it would affect an unrelated second reader.
    """
    counts: dict[str, int] = defaultdict(int)
    for op in operations:
        for dep in op.get_read_writes().reads:
            if isinstance(dep, MemoryDep):
                counts[dep.name] += 1
    return counts


def _walk_chain(buf, consumer_counts: dict[str, int]) -> list:
    """Walk upstream from `buf` through single-input AllSameNode links.

    Continues while the current buffer's restick_cost_fn is a single-edge
    AllSameNode (one real input) and that input has at most one consumer
    (safe to retarget without affecting a different reader). Returns
    [root, ..., buf] -- root is the furthest-upstream buffer reached.
    """
    chain = [buf]
    current = buf
    while True:
        cost_fn = getattr(current, "restick_cost_fn", None)
        if not isinstance(cost_fn, AllSameNode) or len(cost_fn.edge_costs) != 1:
            break
        reads = [
            dep for dep in current.get_read_writes().reads if isinstance(dep, MemoryDep)
        ]
        if len(reads) != 1:
            break
        in_name = reads[0].name
        if consumer_counts.get(in_name, 0) > 1:
            break
        in_buf = V.graph.get_buffer(in_name)
        if not hasattr(in_buf, "layouts"):
            break
        chain.append(in_buf)
        current = in_buf
    chain.reverse()
    return chain


def _derive_chain_value(chain: list, root_stl) -> tuple:
    """Propagate a root candidate down a single-input AllSameNode chain.

    Returns (final_stl, total_cost), or (None, INF) if the root candidate is
    incompatible with the root's own upstream inputs, or some later link has
    no compatible candidate for the value produced by the previous link.
    """
    root_cost = _own_upstream_cost(chain[0], root_stl)
    if root_cost == INF:
        return None, INF
    stl, cost = root_stl, root_cost
    for buf in chain[1:]:
        cost_fn = buf.restick_cost_fn
        ec = cost_fn.edge_costs[0]
        best_c, best_cost = None, INF
        for candidate in buf.layouts:
            c = ec.cost(stl, candidate)
            if c < best_cost:
                best_cost, best_c = c, candidate
        if best_c is None or best_cost == INF:
            return None, INF
        stl, cost = best_c, cost + best_cost
    return stl, cost


def _flexible_siblings(
    cluster: list, consumer_counts: dict[str, int]
) -> dict[str, list]:
    """Return {buf_name: chain} for every distinct input buffer in the
    cluster whose upstream chain root has more than one candidate layout.

    Each chain is [root, ..., direct_sibling] as returned by _walk_chain --
    a chain of length 1 means the sibling itself is the root.
    """
    siblings: dict[str, list] = {}
    for op in cluster:
        for name in _sibling_names(op):
            if name in siblings:
                continue
            buf = V.graph.get_buffer(name)
            if not hasattr(buf, "layouts"):
                continue
            chain = _walk_chain(buf, consumer_counts)
            if len(chain[0].layouts) > 1:
                siblings[name] = chain
    return siblings


def _own_upstream_cost(buf, candidate) -> float:
    """Cost of `buf` committing to `candidate`, against its own upstream
    inputs (0.0 for buffers with no restick_cost_fn, e.g. graph inputs)."""
    cost_fn = getattr(buf, "restick_cost_fn", None)
    if cost_fn is None:
        return 0.0
    in_layouts = []
    for dep in buf.get_read_writes().reads:
        if isinstance(dep, MemoryDep):
            in_buf = V.graph.get_buffer(dep.name)
            in_layouts.append(
                getattr(in_buf, "committed_stl", None) or in_buf.layouts[0]
            )
    return cost_fn.cost(in_layouts, candidate)


def _join_best_cost(join_op, assignment: dict) -> float:
    """Best achievable cost for `join_op`, given a candidate assignment for
    its sibling inputs, minimized over the join's own output candidates."""
    sibling_names = _sibling_names(join_op)
    in_layouts = []
    for name in sibling_names:
        if name in assignment:
            in_layouts.append(assignment[name])
        else:
            buf = V.graph.get_buffer(name)
            in_layouts.append(getattr(buf, "committed_stl", None) or buf.layouts[0])
    cost_fn = join_op.restick_cost_fn
    best = INF
    for candidate in join_op.layouts:
        best = min(best, cost_fn.cost(in_layouts, candidate))
    return best


def _resolve_cluster(cluster: list, consumer_counts: dict[str, int]) -> None:
    siblings = _flexible_siblings(cluster, consumer_counts)
    logger.debug(
        "resolve_join_clusters: cluster %s -> siblings %s",
        [op.get_name() for op in cluster],
        {n: [b.get_name() for b in chain] for n, chain in siblings.items()},
    )
    if not siblings:
        return

    names = list(siblings.keys())
    chains = [siblings[name] for name in names]
    root_candidate_lists = [chain[0].layouts for chain in chains]
    n_combinations = math.prod(len(c) for c in root_candidate_lists)
    if n_combinations > _MAX_COMBINATIONS:
        logger.info(
            "resolve_join_clusters: skipping cluster with %d combinations "
            "(> %d cap): %s",
            n_combinations,
            _MAX_COMBINATIONS,
            [op.get_name() for op in cluster],
        )
        return

    best_cost = INF
    best_assignment = None  # name -> (sibling_stl, chain_cost, root_stl)
    for combo in itertools.product(*root_candidate_lists):
        assignment = {}
        total = 0.0
        feasible = True
        for name, chain, root_stl in zip(names, chains, combo):
            sibling_stl, chain_cost = _derive_chain_value(chain, root_stl)
            if sibling_stl is None:
                feasible = False
                break
            assignment[name] = (sibling_stl, chain_cost, root_stl)
            total += chain_cost
        if not feasible:
            continue
        join_layouts = {name: stl for name, (stl, _, _) in assignment.items()}
        for join_op in cluster:
            total += _join_best_cost(join_op, join_layouts)
        join_layouts_log = {name: list(stl.stride_map) for name, (stl, _, _) in assignment.items()}
        logger.debug(
            "resolve_join_clusters: combo cost=%.1f assignment=%s",
            total, join_layouts_log,
        )
        if total < best_cost:
            best_cost = total
            best_assignment = assignment

    if best_assignment is not None and best_cost < INF:
        logger.debug(
            "resolve_join_clusters: best assignment for cluster %s cost=%.1f: %s",
            [op.get_name() for op in cluster],
            best_cost,
            {n: list(stl.stride_map) for n, (stl, _, _) in best_assignment.items()},
        )

    if best_assignment is None or best_cost == INF:
        # No jointly feasible assignment found -- leave candidates untouched
        # so optimize_restickify_locations's own diagnostic fires normally.
        return

    for name, (sibling_stl, _, root_stl) in best_assignment.items():
        chain = siblings[name]
        root = chain[0]
        own_best_root = min(root.layouts, key=lambda c: _own_upstream_cost(root, c))
        if _stl_key(own_best_root) == _stl_key(root_stl):
            continue  # chain's own optimum already matches the joint winner

        stl = root_stl
        root.layouts[:] = [stl]
        for buf in chain[1:]:
            cost_fn = buf.restick_cost_fn
            ec = cost_fn.edge_costs[0]
            best_c, best_c_cost = None, INF
            for candidate in buf.layouts:
                c = ec.cost(stl, candidate)
                if c < best_c_cost:
                    best_c_cost, best_c = c, candidate
            buf.layouts[:] = [best_c]
            stl = best_c
        logger.debug(
            "resolve_join_clusters: pinned chain %s to joint-optimal candidate "
            "device_size=%s (own-optimal candidate was not jointly feasible)",
            [buf.get_name() for buf in chain],
            list(sibling_stl.device_size),
        )


def resolve_join_clusters(graph: GraphLowering) -> None:
    """Pin sibling candidates that feed a shared multi-input join to the
    combination with globally-minimal cost, when that differs from each
    sibling's own independently-cheapest choice."""
    join_ops = _find_join_ops(graph.operations)
    if not join_ops:
        return
    consumer_counts = _consumer_counts(graph.operations)
    # Process clusters in reverse topological order: downstream joins first.
    # This ensures upstream join ops (e.g. buf20) still have multiple candidates
    # when a downstream join (e.g. buf22) evaluates them jointly.
    for cluster in reversed(_build_clusters(join_ops)):
        _resolve_cluster(cluster, consumer_counts)
