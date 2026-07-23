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
"""Unit tests for coarse_tile.py's _coarse_tile_advance_expr construction."""

import sympy

from torch_spyre._inductor.coarse_tile import _build_advance_expr


class _FakeLoopInfo:
    def __init__(self, loop_tiled_dims, loop_count):
        self.loop_tiled_dims = loop_tiled_dims
        self.loop_count = loop_count


class _FakeData:
    def __init__(self, ranges):
        self.ranges = ranges


class _FakeOp:
    def __init__(self, ranges):
        self.data = _FakeData(ranges)


def test_single_level_single_dim_matches_original_formula():
    """One level, one host dim -- degenerate case, must equal ranges[d]."""
    loop_info = _FakeLoopInfo(loop_tiled_dims=[[0]], loop_count=[4])
    op = _FakeOp(ranges=[64])
    expr = _build_advance_expr(loop_info, op)
    lvl0 = sympy.Symbol("_ct_lvl0")
    assert expr == lvl0 * 64


def test_two_levels_distinct_host_dims():
    """2-D case: Lq (outer, dim 0) and D (inner, dim 1) -- independent dims."""
    loop_info = _FakeLoopInfo(loop_tiled_dims=[[0], [1]], loop_count=[2, 2])
    op = _FakeOp(ranges=[256, 128])
    expr = _build_advance_expr(loop_info, op)
    lvl0 = sympy.Symbol("_ct_lvl0")
    lvl1 = sympy.Symbol("_ct_lvl1")
    assert expr == lvl0 * 256 + lvl1 * 128


def test_two_levels_shared_host_dim_flattened_1d():
    """Flattened 1-D case: both levels tile host dim 0, one term each,
    inner level's coefficient scaled by outer levels sweeping the full
    inner tile first -- must NOT collapse to one term."""
    loop_info = _FakeLoopInfo(loop_tiled_dims=[[0], [0]], loop_count=[2, 2])
    op = _FakeOp(ranges=[8192, 8192])
    expr = _build_advance_expr(loop_info, op)
    lvl0 = sympy.Symbol("_ct_lvl0")
    lvl1 = sympy.Symbol("_ct_lvl1")
    # level 1 (inner) advances by ranges[0]=8192 per its own iteration.
    # level 0 (outer) advances by ranges[0] * loop_count[1] = 8192*2=16384,
    # since one full sweep of the inner level must complete first.
    assert expr == lvl0 * 16384 + lvl1 * 8192


def test_no_tiled_dims_returns_none():
    loop_info = _FakeLoopInfo(loop_tiled_dims=[[], []], loop_count=[1, 1])
    op = _FakeOp(ranges=[1, 1])
    expr = _build_advance_expr(loop_info, op)
    assert expr is None
