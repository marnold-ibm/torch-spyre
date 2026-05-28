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

A, B = 256, 128
x = torch.randn(A, B, dtype=torch.float16) * 0.01
y = torch.randn(A, B, dtype=torch.float16) * 0.01


def f(x, y):
    with spyre_hint(slices={"A": 4}):
        return x + y


r = f(x, y)

x_dev = x.to("spyre")
y_dev = y.to("spyre")

declare_tensor_dim("A", A)
declare_tensor_dim("B", B)

name_tensor_dims(x_dev, ["A", "B"])
name_tensor_dims(y_dev, ["A", "B"])

config.coarse_tiling = True

result = torch.compile(f)(x_dev, y_dev).cpu()

print(r)
print(result)
print(torch.abs(r - result).amax())