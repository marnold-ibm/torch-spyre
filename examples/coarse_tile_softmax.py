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

import torch
from torch_spyre._inductor import spyre_hint, config
import torch_spyre._inductor.propagate_named_dims as prd

declare_tensor_dim = prd.declare_tensor_dim
name_tensor_dims = prd.name_tensor_dims

torch.manual_seed(0xAFFE)

B, D = 256, 128  # batch rows, row length
x = torch.randn(B, D, dtype=torch.float16) * 0.01


def f(x):
    with spyre_hint(slices={"B": 4}):
        max_val = x.amax(dim=-1, keepdim=True)
        x_shifted = x - max_val
        exp_x = x_shifted.exp()
        sum_exp = exp_x.sum(dim=-1, keepdim=True)
        return exp_x / sum_exp


r = f(x)

x_dev = x.to("spyre")

declare_tensor_dim("B", B)
declare_tensor_dim("D", D)

name_tensor_dims(x_dev, ["B", "D"])

config.coarse_tiling = True

result = torch.compile(f)(x_dev).cpu()

print(r)
print(result)
print(torch.abs(r - result).amax())