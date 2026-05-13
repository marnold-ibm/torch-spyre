import torch
DEVICE = torch.device("spyre")
torch.manual_seed(0xAFFE)

s = 128
s1 = 10
s2 = 20
a = torch.randn((s1, s2), dtype=torch.float16) * 0.1
b = torch.randn((s2, s1), dtype=torch.float16) * 0.1
c = torch.randn((s,s), dtype=torch.float16) * 0.1
d = torch.randn((s,s), dtype=torch.float16) * 0.1
# buf0 = a.T + b: conflict, planner picks a stick in buf0's kernel namespace
# buf1 = buf0.T + c: buf0 accessed transposed — d0/d1 swap meaning between kernels
func = lambda a,b,c,d: a + b.t()


# =============================


# Compute on the cpu
cpu_result = func(a,b,c,d)
print ("CPU Result shape", cpu_result.shape, "strides", cpu_result.stride())

print(f"CPU result\n{cpu_result}", "\nShape:", cpu_result.shape)


a_device = a.to(DEVICE)
b_device = b.to(DEVICE)
c_device = c.to(DEVICE)
d_device = d.to(DEVICE)


compiled_sm = torch.compile(func)
compiled_result = compiled_sm(a_device, b_device, c_device, d_device).cpu()

print(f"AIU result\n{compiled_result}")
print ("AIU shape :", compiled_result.shape, "strides:", compiled_result.stride())

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
    print ("ANSWER CORRECT!")

except AssertionError as e:
    print(e)


if cpu_result.stride() == compiled_result.stride():
    print ("STRIDES CORRECT")
else:
    print (f"ERROR: Stride mismatch: CPU {cpu_result.stride()} vs. Compiled Spyre {compiled_result.stride()}")

    
assert (cpu_result.shape == compiled_result.shape), f"Shape mismatch: CPU {cpu_result.shape} vs. Compiled Spyre {compiled_result.shape}"

