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

"""Minimal gather: x[i] with 1d index tensor to isolate index-select."""

import torch
from torch_spyre._inductor import config

torch.manual_seed(3)

M = 128
N = 256
P = 192

x = torch.rand(M, N, dtype=torch.float16)
i = torch.randint(0, 128, (P,), dtype=torch.int32)

# CPU reference
def kernel(x, i):
    return x[i]

ref = kernel(x, i)

# Device run
x_dev = x.to("spyre")
i_dev = i.to("spyre")

result = torch.compile(kernel)(x_dev, i_dev).cpu()

diff = torch.abs(ref - result)
print(f"max abs diff: {diff.amax().item()}")

torch.testing.assert_close(
    result, ref, equal_nan=True, atol=0.01, rtol=0.01,
    msg=lambda msg: f"compiled spyre <-> cpu mismatch\n\n{msg}\n",
)
print("PASSED")
