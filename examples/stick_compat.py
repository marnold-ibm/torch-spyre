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

"""
Demonstrates why y.t().restickify() + x cannot produce a col-major output in a single pass,
and why the intuition that restickify fixes the stick incompatibility is wrong.
"""

import sympy
import os
import torch
os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")
from torch_spyre._C import SpyreTensorLayout
from torch_spyre._inductor.views import compute_coordinates

# ── Problem statement ─────────────────────────────────────────────────────────

print("TL;DR")
print("  Restickify cannot fix stick incompatibility by changing only the device layout and leaving the")
print("  host stride unchanged.  That intuition is wrong.")
print()
print("  The host stride is inextricably linked to the load index expression — they are the same thing,")
print("  expressed two different ways.  Forcing a [1,256] host stride and col-major layout changes both the")
print("  dim order and the index expression — they cancel out, leaving the stick variable unchanged, and stick")
print("  incompatibility remains.")
print()
print("Running example: y.t().restickify() + x")
print()
print("  y:          shape [128, 256], host stride [256, 1]  (row-major)")
print("  y.t():      shape [256, 128], host stride [1, 256]  (col-major, transposed view)")
print("  buf0:       shape [256, 128], host stride [1, 256]  (restickify output, same host stride as y.t())")
print("  x (arg1_1): shape [256, 128], host stride [128, 1]  (row-major)")
print()


# ── Running example: compute stick variables ──────────────────────────────────

c0, c1 = sympy.symbols("c0 c1")
var_ranges = {c0: 256, c1: 128}

# y.t(): col-major [1,256], dim_map=[1,0,1]
stl_yt = SpyreTensorLayout([256, 128], torch.float16, [0, 1])
flat_yt = c0 + c1 * 256   # host index: i0*stride[0] + i1*stride[1] = i0*1 + i1*256
device_yt = compute_coordinates(stl_yt.device_size, stl_yt.stride_map, var_ranges, flat_yt)
yt_stick = device_yt[-1]

print("y.t() (col-major, host stride [1,256], dim_map=[1,0,1]):")
print(f"  STL:              {stl_yt}")
print(f"  device coords:    {device_yt}")
print(f"  stick expression: {yt_stick}  =>  loop variable c0")
print()

# buf0: col-major [1,256], dim_map=[0,1,0] (after restickify)
stl_buf0 = SpyreTensorLayout([256, 128], torch.float16, [1, 0])
flat_buf0 = c0 + c1 * 256  # same host index — restickify does not change the host stride
device_buf0 = compute_coordinates(stl_buf0.device_size, stl_buf0.stride_map, var_ranges, flat_buf0)
buf0_stick = device_buf0[-1]

print("buf0 (col-major, host stride [1,256], dim_map=[0,1,0], after restickify):")
print(f"  STL:              {stl_buf0}")
print(f"  device coords:    {device_buf0}")
print(f"  stick expression: {buf0_stick}  =>  loop variable c0  (unchanged from y.t())")
print()

# x: row-major [128,1], dim_map=[1,0,1]
stl_x = SpyreTensorLayout([256, 128], torch.float16, [0, 1])
flat_x = c0 * 128 + c1    # host index: i0*stride[0] + i1*stride[1] = i0*128 + i1*1
device_x = compute_coordinates(stl_x.device_size, stl_x.stride_map, var_ranges, flat_x)
x_stick = device_x[-1]

print("x (row-major, host stride [128,1], dim_map=[1,0,1]):")
print(f"  STL:              {stl_x}")
print(f"  device coords:    {device_x}")
print(f"  stick expression: {x_stick}  =>  loop variable c1")
print()

compatible_before = sympy.simplify(yt_stick - x_stick) == 0
compatible_after  = sympy.simplify(buf0_stick - x_stick) == 0
print(f"Stick compatibility (y.t() vs x):  {'COMPATIBLE' if compatible_before else 'INCOMPATIBLE'}")
print(f"Stick compatibility (buf0  vs x):  {'COMPATIBLE' if compatible_after  else 'INCOMPATIBLE'}")
print()

# ── Explanation ───────────────────────────────────────────────────────────────

print("The stick variable of buf0 is c0 regardless of dim_map, as the add kernel shows directly:")
print()
print("  def inner_fn(index):       # (from y.t().restickify() + x)")
print("      i0, i1 = index")
print("      tmp0 = ops.load(buf0,    i0 + 256 * i1)  # stride [1,256]: i0 has stride 1 => stick = c0")
print("      tmp1 = ops.load(arg1_1,  i1 + 128 * i0)  # stride [128,1]: i1 has stride 1 => stick = c1")
print("      tmp2 = tmp0 + tmp1")
print("      return tmp2")
print()
print("The load index is the host stride written as code; restickify cannot change either.")
print("If col-major stride [1,256] is required, a separate restickify kernel is needed after the add.")