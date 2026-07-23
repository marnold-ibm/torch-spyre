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
"""Unit tests for compute_ops.py's tile_advance_expr coefficient extraction."""

import sympy

from torch_spyre._inductor.codegen.compute_ops import _level_stride_from_expr


def test_single_level_symbol_extracts_coefficient():
    expr = sympy.Symbol("_ct_lvl0") * 16384
    stride = _level_stride_from_expr(expr, level_idx=0)
    assert stride == 16384


def test_multi_level_expr_extracts_each_level_independently():
    expr = sympy.Symbol("_ct_lvl0") * 16384 + sympy.Symbol("_ct_lvl1") * 8192
    assert _level_stride_from_expr(expr, level_idx=0) == 16384
    assert _level_stride_from_expr(expr, level_idx=1) == 8192


def test_level_not_present_returns_none():
    expr = sympy.Symbol("_ct_lvl0") * 16384
    assert _level_stride_from_expr(expr, level_idx=1) is None


def test_none_expr_returns_none():
    assert _level_stride_from_expr(None, level_idx=0) is None
