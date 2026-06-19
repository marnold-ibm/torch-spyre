"""Case D: view/reshape — named dims describe a logical structure that differs
from the physical layout.  w and x are stored as [1, 2, 384] but annotated
['A','B','D','E'] (4 logical dims).  The function views them to [1,2,3,2,64]
before use.  This exercises the len(layout.size) != len(buf_named_dims) path.
"""

import os
os.environ["SPYRE_INDUCTOR_LOG"] = "1"
os.environ["SPYRE_INDUCTOR_LOG_LEVEL"] = "INFO"

import torch
import torch_spyre._inductor.passes as passes

import torch_spyre._inductor.propagate_named_dims as prd

passes.propagate_real_dims = prd.propagate_named_dims
declare_tensor_dim = prd.declare_tensor_dim
get_last_output_named_dims = prd.get_last_output_named_dims
name_tensor_dims = prd.name_tensor_dims

DEVICE = torch.device("spyre")
torch.manual_seed(0xAFFE)

a, b, c, d, e = 2, 3, 4, 2, 64
w = torch.randn((1, a, b*d*e), dtype=torch.float16) * 0.1
x = torch.randn((1, a, b*d* e), dtype=torch.float16) * 0.1
y = torch.randn((1, a, c, d, e), dtype=torch.float16) * 0.1
z = torch.randn((1, a, c, 1, 1, 1), dtype=torch.float16) * 0.1

w_device = w.to(DEVICE)
x_device = x.to(DEVICE)
y_device = y.to(DEVICE)
z_device = z.to(DEVICE)

declare_tensor_dim("A", a)
declare_tensor_dim("B", b)
declare_tensor_dim("C", c)
declare_tensor_dim("D", d)
declare_tensor_dim("E", e)

name_tensor_dims(w_device, ["A", "B", "D", "E"])
name_tensor_dims(x_device, ["A", "B", "D", "E"])
name_tensor_dims(y_device, ["A", "C", "D", "E"])
name_tensor_dims(z_device, ["A", "C"])

# ----------------------------------------

def func(w, x, y, z):
    t = w + x
    t = t.view(1, a, b, d, e)
    t = t.unsqueeze(2) + y.unsqueeze(3)
    return t + z


cpu_result = func(w, x, y, z)
print("CPU Result shape", cpu_result.shape, "strides", cpu_result.stride())

compiled_sm = torch.compile(func)
compiled_result = compiled_sm(w_device, x_device, y_device, z_device).cpu()

# print(f"AIU result\n{compiled_result}")
print("AIU shape :", compiled_result.shape, "strides:", compiled_result.stride())

cpu_delta = torch.abs(compiled_result - cpu_result).max()
print(f"Max delta Compiled Spyre vs. CPU: {cpu_delta}")

try:
    torch.testing.assert_close(
        compiled_result,
        cpu_result,
        equal_nan=True,
        atol=0.1,
        rtol=0.1,
        msg=lambda msg: f"compiled spyre <-> cpu mismatch\n\n{msg}\n",
    )
    print("ANSWER CORRECT!")

except AssertionError as e:
    print(e)


if cpu_result.stride() == compiled_result.stride():
    print("STRIDES CORRECT")
else:
    print(
        f"ERROR: Stride mismatch: CPU {cpu_result.stride()} vs. Compiled Spyre {compiled_result.stride()}"
    )

assert cpu_result.shape == compiled_result.shape, (
    f"Shape mismatch: CPU {cpu_result.shape} vs. Compiled Spyre {compiled_result.shape}"
)

assert get_last_output_named_dims() == ["A", "C", "B", "D", "E"], f"Case D: got {get_last_output_named_dims()}"
print("Case D NAMED DIM ASSERTIONS PASSED")
