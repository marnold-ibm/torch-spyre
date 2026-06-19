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

"""Minimal scatter+exp: y.scatter_(0, index, src.exp()) to isolate aten.scatter_ + unary fusion."""

import torch
import torch_spyre._inductor.propagate_named_dims as pnd

declare_tensor_dim = pnd.declare_tensor_dim
name_tensor_dims = pnd.name_tensor_dims

torch.manual_seed(3)

M = 128
N = 256
P = 3

y = torch.zeros(M, N, dtype=torch.float16)
src = torch.rand(P, N, dtype=torch.float16)
index = torch.randint(0, M, (P, N), dtype=torch.int32)


# CPU reference
def kernel(y, src, index):
    return y.scatter_(0, index, src.exp())


ref = kernel(y.clone(), src, index)

# Device run
y_dev = y.to("spyre")
src_dev = src.to("spyre")
index_dev = index.to("spyre")

declare_tensor_dim("M", M)
declare_tensor_dim("N", N)
declare_tensor_dim("P", P)

name_tensor_dims(y_dev, ["M", "N"])
name_tensor_dims(src_dev, ["P", "N"])
name_tensor_dims(index_dev, ["P", "N"])

result = torch.compile(kernel)(y_dev, src_dev, index_dev).cpu()

diff = torch.abs(ref - result)
print(f"max abs diff: {diff.amax().item()}")

torch.testing.assert_close(
    result,
    ref,
    equal_nan=True,
    atol=0.01,
    rtol=0.01,
    msg=lambda msg: f"compiled spyre <-> cpu mismatch\n\n{msg}\n",
)
print("PASSED")
