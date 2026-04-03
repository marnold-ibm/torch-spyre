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

# Tests for restickify insertion in pointwise operations.
#
# Restickify is triggered when a transposed (non-contiguous) tensor is used
# in a pointwise op alongside a contiguous tensor, and the layouts are
# stick-incompatible. The compiler inserts a restickify kernel to convert
# the layout before the pointwise op proceeds.
#
# Shapes use multiples of 64 (stick size = 64 fp16 elements) to ensure
# stick-aligned inputs that exercise the restickify path rather than fallback.

import pytest
import torch

from utils_inductor import _compile_and_run, compare_with_cpu

DEVICE = torch.device("spyre")


def _compare(fn, *args):
    spyre_result = _compile_and_run(fn, args, DEVICE)
    compare_with_cpu(fn, *args, target=spyre_result, run_eager=False)
    cpu_result = fn(*args)
    assert cpu_result.stride() == spyre_result.stride(), (
        f"Stride mismatch: CPU {cpu_result.stride()} vs Spyre {spyre_result.stride()}"
    )


SIZE_PAIRS_2D = [
    (256, 128),
    (64, 128),
]


@pytest.fixture(params=SIZE_PAIRS_2D, ids=lambda p: f"{p[0]}x{p[1]}")
def tensors(request):
    s1, s2 = request.param
    # A, B: shape [s1, s2]
    A = torch.randn((s1, s2), dtype=torch.float16)
    B = torch.randn((s1, s2), dtype=torch.float16)
    X = torch.randn((s2, s1), dtype=torch.float16)
    Y = torch.randn((s2, s1), dtype=torch.float16)
    return A, B, X, Y


# -------- Pointwise tests ----------

# 2-arg tests
def test_2arg_at_plus_x(tensors):
    A, _, X, _ = tensors
    _compare(lambda a, x: a.t() + x, A, X)

def test_2arg_x_plus_at(tensors):
    A, _, X, _ = tensors
    _compare(lambda a, x: x + a.t(), A, X)

def test_2arg_xt_plus_a(tensors):
    A, _, X, _ = tensors
    _compare(lambda a, x: x.t() + a, A, X)

def test_2arg_a_plus_xt(tensors):
    A, _, X, _ = tensors
    _compare(lambda a, x: a + x.t(), A, X)

# 3-arg tests
def test_3arg_at_bt_x(tensors):
    A, B, X, _ = tensors
    _compare(lambda a, b, x: a.t() + b.t() + x, A, B, X)

def test_3arg_at_x_bt(tensors):
    A, B, X, _ = tensors
    _compare(lambda a, b, x: a.t() + x + b.t(), A, B, X)

def test_3arg_x_at_bt(tensors):
    A, B, X, _ = tensors
    _compare(lambda a, b, x: x + a.t() + b.t(), A, B, X)

def test_3arg_at_x_y(tensors):
    A, _, X, Y = tensors
    _compare(lambda a, x, y: a.t() + x + y, A, X, Y)

# 4-arg tests
def test_4arg_at_bt_x_y(tensors):
    A, B, X, Y = tensors
    _compare(lambda a, b, x, y: a.t() + b.t() + x + y, A, B, X, Y)

def test_4arg_at_x_bt_y(tensors):
    A, B, X, Y = tensors
    _compare(lambda a, b, x, y: a.t() + x + b.t() + y, A, B, X, Y)

def test_4arg_x_at_y_bt(tensors):
    A, B, X, Y = tensors
    _compare(lambda a, b, x, y: x + a.t() + y + b.t(), A, B, X, Y)

def test_4arg_at_x_y_bt(tensors):
    A, B, X, Y = tensors
    _compare(lambda a, b, x, y: a.t() + x + y + b.t(), A, B, X, Y)

# 3D tests
# a: [s0, s1, s2], x: [s0, s2, s1] — transpose dims 1 and 2
@pytest.fixture(params=[(2, 256, 128), (4, 128, 64)], ids=lambda p: f"{p[0]}x{p[1]}x{p[2]}")
def tensors_3d(request):
    s0, s1, s2 = request.param
    a = torch.randn((s0, s1, s2), dtype=torch.float16)
    x = torch.randn((s0, s2, s1), dtype=torch.float16)
    return a, x

def test_3d_transpose12_plus_x(tensors_3d):
    a, x = tensors_3d
    _compare(lambda a, x: a.transpose(1, 2) + x, a, x)

def test_3d_x_plus_transpose12(tensors_3d):
    a, x = tensors_3d
    _compare(lambda a, x: x + a.transpose(1, 2), a, x)

# 4D tests
# a: [s0, s1, s2, s3], x: [s0, s3, s2, s1] — transpose dims 1 and 3
@pytest.fixture(params=[(2, 256, 3, 128), (2, 128, 4, 64)], ids=lambda p: f"{p[0]}x{p[1]}x{p[2]}x{p[3]}")
def tensors_4d(request):
    s0, s1, s2, s3 = request.param
    a = torch.randn((s0, s1, s2, s3), dtype=torch.float16)
    x = torch.randn((s0, s3, s2, s1), dtype=torch.float16)
    return a, x

def test_4d_transpose13_plus_x(tensors_4d):
    a, x = tensors_4d
    _compare(lambda a, x: a.transpose(1, 3) + x, a, x)

def test_4d_x_plus_transpose13(tensors_4d):
    a, x = tensors_4d
    _compare(lambda a, x: x + a.transpose(1, 3), a, x)


# ------- Matmul Tests ---------

MATMUL_SIZE_PAIRS = [(128, 256), (64, 128)]

@pytest.fixture(params=MATMUL_SIZE_PAIRS, ids=[f"{a}x{b}" for a, b in MATMUL_SIZE_PAIRS])
def matmul_tensors_ab(request):
    a, b = request.param
    x = torch.randn((a, b), dtype=torch.float16) * 0.1
    y = torch.randn((a, b), dtype=torch.float16) * 0.1
    return x, y

@pytest.fixture(params=MATMUL_SIZE_PAIRS, ids=[f"{a}x{b}" for a, b in MATMUL_SIZE_PAIRS])
def matmul_tensors_ab_ba(request):
    a, b = request.param
    x = torch.randn((a, b), dtype=torch.float16) * 0.1
    y = torch.randn((b, a), dtype=torch.float16) * 0.1
    return x, y

def test_matmul_xt_y(matmul_tensors_ab):
    x, y = matmul_tensors_ab
    _compare(lambda x, y: torch.matmul(x.t(), y), x, y)

def test_matmul_x_yt(matmul_tensors_ab):
    x, y = matmul_tensors_ab
    _compare(lambda x, y: torch.matmul(x, y.t()), x, y)

def test_matmul_xt_yt(matmul_tensors_ab_ba):
    x, y = matmul_tensors_ab_ba
    _compare(lambda x, y: torch.matmul(x.t(), y.t()), x, y)
