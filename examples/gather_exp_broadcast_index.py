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

"""gather+exp with broadcast on index: i is expanded from (P, 1) to (P, Q)."""

import torch
import torch_spyre._inductor.propagate_named_dims as pnd

declare_tensor_dim = pnd.declare_tensor_dim
name_tensor_dims = pnd.name_tensor_dims

torch.manual_seed(3)

M = 128
N = 256
P = 3
Q = 192

x = torch.rand(M, N, dtype=torch.float16)
i_base = torch.randint(0, M, (P, 1), dtype=torch.int32)
i = i_base.expand(P, Q)


def kernel(x, i):
    return x[i].exp()


ref = kernel(x, i)

x_dev = x.to("spyre")
i_dev = i_base.to("spyre").expand(P, Q)

declare_tensor_dim("M", M)
declare_tensor_dim("N", N)
declare_tensor_dim("P", P)
declare_tensor_dim("Q", Q)

name_tensor_dims(x_dev, ["M", "N"])
name_tensor_dims(i_dev, ["P", "Q"])

result = torch.compile(kernel)(x_dev, i_dev).cpu()

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
