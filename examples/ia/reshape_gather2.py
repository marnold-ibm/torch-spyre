import torch

t = torch.arange(4096, dtype=torch.float16)  # 16 pages with 256 tokens
idx = torch.tensor((7, 3), dtype=torch.int32)  # list of pages


# extract pages
def f(t, idx):
    return t.reshape(16, 256)[idx]


t_dev = t.to("spyre")
idx_dev = idx.to("spyre")

ref = f(t, idx)
print(ref)

result = torch.compile(f)(t_dev, idx_dev).cpu()

diff = torch.abs(ref - result)
print(f"max abs diff: {diff.amax().item()}")

torch.testing.assert_close(
    result, ref, equal_nan=True, atol=0.01, rtol=0.01,
    msg=lambda msg: f"compiled spyre <-> cpu mismatch\n\n{msg}\n",
)
print("PASSED")