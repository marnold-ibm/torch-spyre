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

"""Case B: permute only — the problematic pattern.
queries is [B, Lq, H, D] (Lq and H swapped vs. case A).
The function does queries.permute(0,2,1,3) to get [B, H, Lq, D],
then multiplies by scale and clones — matching the buf5->buf6 path in test_paged.

Expected with current code: _lone_sym assertion fires (or wrong named dims / non-determinism).
Expected with fix: d0->B, d1->H, d2->Lq, d3->D, no warnings.
"""

import math
import torch
from torch_spyre._inductor import spyre_hint
from torch_spyre._inductor.propagate_named_dims import (
    declare_tensor_dim,
    get_last_output_named_dims,
    name_tensor_dims,
)

B, H, Lq, D = 12, 32, 256, 128


def fn(queries: torch.Tensor, scale: float):
    # queries: [B, Lq, H, D] — permute to [B, H, Lq, D] then clone
    return (queries.permute(0, 2, 1, 3) * scale).contiguous()


if __name__ == "__main__":
    import os
    os.environ["SPYRE_INDUCTOR_LOG"] = "1"
    os.environ["SPYRE_INDUCTOR_LOG_LEVEL"] = "INFO"

    # Input is [B, Lq, H, D] — note Lq and H order
    queries = torch.randn(B, Lq, H, D, dtype=torch.float16, device="spyre")

    declare_tensor_dim("B", B)
    declare_tensor_dim("H", H)
    declare_tensor_dim("Lq", Lq)
    declare_tensor_dim("D", D)

    name_tensor_dims(queries, ["B", "Lq", "H", "D"])

    with spyre_hint(tiles={"B": 6}):
        with spyre_hint(tiles={"H": 8}):
            with spyre_hint(tiles={"Lq": 4}):
                c_fn = torch.compile(fn)
                out = c_fn(queries, 1.0 / math.sqrt(D))

    assert get_last_output_named_dims() == ["B", "H", "Lq", "D"], f"Case B: got {get_last_output_named_dims()}"
    print("Case B done, shape:", out.shape, "ASSERTIONS PASSED")
