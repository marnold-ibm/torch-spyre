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

"""Minimal scatter+exp: y[i] = src.exp() to isolate indirect store + unary fusion."""

import torch
import torch_spyre._inductor.propagate_named_dims as pnd

declare_tensor_dim = pnd.declare_tensor_dim
name_tensor_dims = pnd.name_tensor_dims

torch.manual_seed(3)

M = 128
N = 256
P = 3
Q = 192

y = torch.zeros(M, N, dtype=torch.float16)
src = torch.rand(P, N, dtype=torch.float16)
i = torch.randint(0, M, (P,), dtype=torch.int32)


# CPU reference
def kernel(y, src, i):
    y[i] = src.exp()
    return y


ref = kernel(y.clone(), src, i)

# Device run
y_dev = y.to("spyre")
src_dev = src.to("spyre")
i_dev = i.to("spyre")

declare_tensor_dim("M", M)
declare_tensor_dim("N", N)
declare_tensor_dim("P", P)
declare_tensor_dim("Q", Q)

name_tensor_dims(y_dev, ["M", "N"])
name_tensor_dims(src_dev, ["P", "N"])
name_tensor_dims(i_dev, ["P"])

result = torch.compile(kernel)(y_dev, src_dev, i_dev).cpu()

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
