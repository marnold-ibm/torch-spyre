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
"""Unit test for TensorArg.tile_advance_expr's default and shape."""

import sympy

from torch_spyre._inductor.op_spec import OpSpec, TensorArg


def _minimal_tensor_arg(**kwargs):
    return TensorArg(
        is_input=True,
        arg_index=0,
        device_dtype=None,
        device_size=[],
        device_coordinates=[],
        allocation=None,
        **kwargs,
    )


def test_tile_advance_expr_defaults_to_none():
    arg = _minimal_tensor_arg()
    assert arg.tile_advance_expr is None


def test_tile_advance_expr_accepts_sympy_expr():
    expr = sympy.Symbol("_ct_lvl0") * 128
    arg = _minimal_tensor_arg(tile_advance_expr=expr)
    assert arg.tile_advance_expr == expr


def test_dim_advance_overrides_field_removed():
    arg = _minimal_tensor_arg()
    assert not hasattr(arg, "dim_advance_overrides")
    assert not hasattr(OpSpec, "dim_advance_overrides")
