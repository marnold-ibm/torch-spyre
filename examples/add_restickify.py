import torch
DEVICE = torch.device("spyre")
torch.manual_seed(0xAFFE)

size1 = 256
size2 = 128

x = torch.randn((size1, size2), dtype=torch.float16)
y = torch.randn((size2, size1), dtype=torch.float16)
z = torch.randn((size1, size2), dtype=torch.float16)
print ("Size and stride of x:", x.shape, x.stride())
print ("Size and stride of y:", y.shape, y.stride())
print ("Size and stride of y.t():", y.t().shape, y.t().stride())

# THIS WORKS
# cpu_func = lambda x,y,z: z+ y.t() + x 
# spyre_func = lambda x, y, z: z + y.t().restickify() + x


# Correct result and strides produced with
#  DCI_HACK=1 SPYRE_COLMAJOR_OUTPUT=1 SENCORES=1 TORCH_SPYRE_DEBUG=1 TORCH_COMPILE_DEBUG=0  python -u add_restickify.py 


cpu_func = lambda x,y,z: y.t() + x

# spyre_func = lambda x, y: y.t() + x
# spyre_func = lambda x, y: y.t().contiguous() + x
# spyre_func = lambda x, y, z: (y.t().restickify([0,1]) + x).restickify([1,0])

# Works with DCI change
# spyre_func = lambda x, y, z: y.t().restickify([0,1]) + x
# spyre_func = lambda x, y, z: y.t().contiguous() + x
spyre_func = lambda x, y, z: y.t().restickify([0,1]) + x

# =============================


# Compute on the cpu
cpu_result = cpu_func(x,y,z)
print ("CPU Result shape", cpu_result.shape, "strides", cpu_result.stride())

print(f"CPU result\n{cpu_result}", "\nShape:", cpu_result.shape)


x_device = x.to(DEVICE)
y_device = y.to(DEVICE)
z_device = z.to(DEVICE)

compiled_sm = torch.compile(spyre_func)
compiled_result = compiled_sm(x_device, y_device, z_device).cpu()

print(f"AIU result\n{compiled_result}")
print ("AIU shape :", compiled_result.shape, "strides:", compiled_result.stride())
print ("CPU shape", cpu_result.shape, "strides", cpu_result.stride())

cpu_delta = torch.abs(compiled_result - cpu_result).max()
print(f"Max delta Compiled Spyre vs. CPU: {cpu_delta}")


# # ── Mismatch analysis ─────────────────────────────────────────────────────────
# tol = 0.1
# diff = torch.abs(compiled_result - cpu_result)
# match = diff <= tol

# print(f"\nMismatch analysis (tol={tol}):")
# print(f"  total cells:     {match.numel()}")
# print(f"  matching:        {match.sum().item()}")
# print(f"  mismatching:     {(~match).sum().item()}")
# print()

# # Print a grid showing which cells match (.) and which don't (X)
# print("  Match grid (. = match, X = mismatch)  rows=p0, cols=p1:")
# for r in range(cpu_result.shape[0]):
#     row_str = "  "
#     for c in range(cpu_result.shape[1]):
#         row_str += "." if match[r, c] else "X"
#     print(row_str)

# # Show the boundary row/col where mismatches start
# mismatch_rows = (~match).any(dim=1).nonzero(as_tuple=True)[0]
# mismatch_cols = (~match).any(dim=0).nonzero(as_tuple=True)[0]
# if len(mismatch_rows):
#     print(f"\n  First mismatch row: {mismatch_rows[0].item()},  last: {mismatch_rows[-1].item()}")
#     print(f"  First mismatch col: {mismatch_cols[0].item()},  last: {mismatch_cols[-1].item()}")

try:
    torch.testing.assert_close(
        compiled_result,
        cpu_result,
        equal_nan=True,
        atol=0.1,
        rtol=0.01,
        msg=lambda msg: f"compiled spyre <-> cpu mismatch\n\n{msg}\n",
    )
    print ("ANSWER CORRECT!")

except AssertionError as e:
    print(e)

assert (cpu_result.shape == compiled_result.shape), f"Shape mismatch: CPU {cpu_result.shape} vs. Compiled Spyre {compiled_result.shape}"

if cpu_result.stride() == compiled_result.stride():
    print ("STRIDES CORRECT")
else:
    print (f"ERROR: Stride mismatch: CPU {cpu_result.stride()} vs. Compiled Spyre {compiled_result.stride()}")