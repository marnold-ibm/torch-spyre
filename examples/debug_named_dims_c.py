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

"""Case C: permute followed by matmul — closer to test_paged.
queries [B, Lq, H, D] permuted to [B, H, Lq, D], multiplied by scale.
keys    [B, Lk, H, D] permuted to [B, H, D, Lk] for matmul.
scores = matmul(queries_perm, keys_perm) -> [B, H, Lq, Lk]

This is the core of the attention score computation in test_paged without
the indirect index access, so we can see whether named dim propagation
through permute+matmul works correctly.

Expected with current code: _lone_sym assertion fires on buf6 (queries clone).
Expected with fix: clean propagation B->d0, H->d1, Lq->d2, D/Lk->d3.
"""

import math
import torch
from torch_spyre._inductor import spyre_hint
from torch_spyre._inductor.propagate_named_dims import (
    declare_tensor_dim,
    get_last_output_named_dims,
    name_tensor_dims,
)

B, H, Lq, Lk, D = 12, 32, 256, 256, 128


def fn(queries: torch.Tensor, keys: torch.Tensor, scale: float):
    # queries: [B, Lq, H, D] -> permute -> [B, H, Lq, D]
    # keys:    [B, Lk, H, D] -> permute -> [B, H, D, Lk]
    q = queries.permute(0, 2, 1, 3) * scale        # [B, H, Lq, D]
    k = keys.permute(0, 2, 3, 1) * scale            # [B, H, D, Lk]
    return torch.matmul(q, k)                        # [B, H, Lq, Lk]


if __name__ == "__main__":
    import os
    os.environ["SPYRE_INDUCTOR_LOG"] = "1"
    os.environ["SPYRE_INDUCTOR_LOG_LEVEL"] = "INFO"

    queries = torch.randn(B, Lq, H, D, dtype=torch.float16, device="spyre")
    keys    = torch.randn(B, Lk, H, D, dtype=torch.float16, device="spyre")

    declare_tensor_dim("B", B)
    declare_tensor_dim("H", H)
    declare_tensor_dim("Lq", Lq)
    declare_tensor_dim("Lk", Lk)
    declare_tensor_dim("D", D)

    name_tensor_dims(queries, ["B", "Lq", "H", "D"])
    name_tensor_dims(keys,    ["B", "Lk", "H", "D"])

    with spyre_hint(tiles={"B": 6}):
        with spyre_hint(tiles={"H": 8}):
            with spyre_hint(tiles={"Lq": 4}):
                c_fn = torch.compile(fn)
                out = c_fn(queries, keys, 1.0 / math.sqrt(D))

    assert get_last_output_named_dims() == ["B", "H", "Lq", "Lk"], f"Case C: got {get_last_output_named_dims()}"
    print("Case C done, shape:", out.shape, "ASSERTIONS PASSED")
