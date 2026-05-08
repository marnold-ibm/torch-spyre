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

"""Split oversized pointwise ops into memory-safe chunks.

Runs after ``propagate_spyre_tensor_layouts`` / ``insert_restickify`` and
before ``span_reduction``.  Each chunk becomes a normal
``ComputedBuffer`` that work-division handles without special-casing.
"""

import math

import torch
from torch._inductor import lowering as ind_lowering
from torch._inductor.ir import (
    ComputedBuffer,
    MutationLayoutSHOULDREMOVE,
    Operation,
    Pointwise,
    Scatter,
)
from torch._inductor.virtualized import V

from . import config
from .work_division import MAX_SPAN_BYTES
from .ir import FixedTiledLayout
from .logging_utils import get_inductor_logger

logger = get_inductor_logger("chunk_large_tensors")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _needs_chunking(
    op: ComputedBuffer, max_cores: int
) -> tuple[int, FixedTiledLayout] | None:
    """Return ``(total_device_bytes, layout)`` when the op exceeds capacity."""
    layout = op.layout
    device_size = layout.device_layout.device_size
    total_bytes = math.prod(int(s) for s in device_size) * layout.dtype.itemsize
    if total_bytes > MAX_SPAN_BYTES * max_cores:
        return total_bytes, layout
    return None


def _find_split_dim(layout: FixedTiledLayout) -> int:
    """Pick the host dimension to split along.

    Walks device dims outermost-first (skipping the within-stick dim) and
    returns the first host dim whose size > 1.  Falls back to the largest
    host dim.
    """
    stl = layout.device_layout
    host_size = [int(s) for s in layout.size]
    host_stride = [int(s) for s in layout.stride]

    for device_dim in range(len(stl.device_size) - 1):
        sm = int(stl.stride_map[device_dim])
        if sm <= 0:
            continue
        host_dim = next(
            (d for d, s in enumerate(host_stride) if s == sm),
            None,
        )
        if host_dim is not None and host_size[host_dim] > 1:
            return host_dim

    return max(range(len(host_size)), key=lambda d: host_size[d])


def _make_chunk_layout(
    original_ftl: FixedTiledLayout,
    split_dim_idx: int,
    chunk_size: int,
) -> FixedTiledLayout:
    """Build a ``FixedTiledLayout`` for a single chunk."""
    from torch_spyre._C import SpyreTensorLayout

    host_size = [int(s) for s in original_ftl.size]
    host_size[split_dim_idx] = chunk_size

    host_stride = [1] * len(host_size)
    for d in range(len(host_size) - 2, -1, -1):
        host_stride[d] = host_stride[d + 1] * host_size[d + 1]

    stl = SpyreTensorLayout(host_size, original_ftl.dtype)
    return FixedTiledLayout(
        original_ftl.device,
        original_ftl.dtype,
        host_size,
        host_stride,
        stl,
    )


def _make_chunk_fn(orig_fn, dim: int, offset: int):
    """Return an ``inner_fn`` that reads from *orig_fn* with *offset* on *dim*."""

    def inner_fn(index):
        idx = list(index)
        idx[dim] = idx[dim] + offset
        return orig_fn(idx)

    return inner_fn


def _make_overwrite_fn(overwrite_op, loader, offset: int, split_dim: int):
    """Return ``(inner_fn, output_indexer)`` for an overwrite-scatter."""

    def inner_fn(index):
        return overwrite_op(loader(index))

    def output_indexer(index):
        out = list(index)
        out[split_dim] = out[split_dim] + offset
        return out

    return inner_fn, output_indexer


def _register_and_insert(
    buf: ComputedBuffer,
    op: ComputedBuffer,
    operations: list[Operation],
    insert_pos: int,
) -> int:
    """Register *buf* in the graph and insert it at *insert_pos*.

    ``V.graph.register_operation`` appends to the same ``operations`` list,
    so the duplicate is removed before the positioned insert.

    Returns the next insert position.
    """
    buf.name = V.graph.register_buffer(buf)
    V.graph.register_operation(buf)
    buf.origins = op.origins
    if buf in operations:
        operations.remove(buf)
    operations.insert(insert_pos, buf)
    return insert_pos + 1


# ---------------------------------------------------------------------------
# Core chunking logic
# ---------------------------------------------------------------------------


def _chunk_op(
    op: ComputedBuffer,
    max_cores: int,
    operations: list[Operation],
    op_index: int,
    total_bytes: int,
    original_ftl: FixedTiledLayout,
) -> None:
    original_ranges = list(op.data.ranges)
    original_inner_fn = op.data.inner_fn

    split_dim_idx = _find_split_dim(original_ftl)
    full_size = int(original_ranges[split_dim_idx])
    num_chunks = math.ceil(total_bytes / (MAX_SPAN_BYTES * max_cores))
    chunk_size = math.ceil(full_size / num_chunks)
    num_chunks = math.ceil(full_size / chunk_size)

    logger.info(
        "Chunking %s: dim=%d, full_size=%d, num_chunks=%d, "
        "chunk_size=%d, total_bytes=%.2fGB",
        op.get_name(),
        split_dim_idx,
        full_size,
        num_chunks,
        chunk_size,
        total_bytes / (1024**3),
    )

    overwrite_fn = ind_lowering.ops_wrapper(torch.ops.spyre.overwrite.__name__)

    # --- chunk 0: shrink the original op in-place -------------------------
    # Layout stays full-size (buf0 is allocated at full size); only the
    # iteration ranges are reduced so chunk 0 computes rows 0..chunk_size-1.
    chunk0_size = min(chunk_size, full_size)
    chunk0_ranges = list(original_ranges)
    chunk0_ranges[split_dim_idx] = chunk0_size
    object.__setattr__(op.data, "ranges", chunk0_ranges)

    # --- chunks 1..N-1: new compute + overwrite-scatter buffers -----------
    insert_pos = op_index + 1
    mutation_target = op

    for c in range(1, num_chunks):
        offset = c * chunk_size
        this_chunk_size = min(chunk_size, full_size - offset)
        chunk_ranges = list(original_ranges)
        chunk_ranges[split_dim_idx] = this_chunk_size

        # -- compute buffer --
        chunk_pw = Pointwise(
            device=op.data.device,
            dtype=op.data.dtype,
            inner_fn=_make_chunk_fn(original_inner_fn, split_dim_idx, offset),
            ranges=chunk_ranges,
        )
        object.__setattr__(chunk_pw, "origins", op.data.origins)
        object.__setattr__(chunk_pw, "traceback", op.data.traceback)

        chunk_layout = _make_chunk_layout(original_ftl, split_dim_idx, this_chunk_size)
        chunk_buf = ComputedBuffer(name=None, layout=chunk_layout, data=chunk_pw)
        chunk_buf.origin_node = op.origin_node
        insert_pos = _register_and_insert(chunk_buf, op, operations, insert_pos)

        # -- overwrite-scatter buffer --
        overwrite_inner, overwrite_indexer = _make_overwrite_fn(
            overwrite_fn, chunk_buf.make_loader(), offset, split_dim_idx
        )
        overwrite_data = Scatter(
            device=op.data.device,
            dtype=op.data.dtype,
            inner_fn=overwrite_inner,
            ranges=chunk_ranges,
            output_indexer=overwrite_indexer,
        )
        overwrite_buf = ComputedBuffer(
            name=None,
            layout=MutationLayoutSHOULDREMOVE(mutation_target),
            data=overwrite_data,
        )
        insert_pos = _register_and_insert(overwrite_buf, op, operations, insert_pos)
        mutation_target = overwrite_buf


def chunk_large_tensors(operations: list[Operation]) -> None:
    """Split pointwise ops whose device footprint exceeds the hardware limit.

    Must run **after** ``propagate_spyre_tensor_layouts`` /
    ``insert_restickify`` and **before** ``core_division_planning``.
    """
    max_cores = config.sencores
    i = 0
    while i < len(operations):
        op = operations[i]
        if (
            isinstance(op, ComputedBuffer)
            and isinstance(op.data, Pointwise)
            and isinstance(op.layout, FixedTiledLayout)
            and len(op.data.ranges) == 3
        ):
            result = _needs_chunking(op, max_cores)
            if result is not None:
                total_bytes, layout = result
                _chunk_op(op, max_cores, operations, i, total_bytes, layout)
        i += 1
