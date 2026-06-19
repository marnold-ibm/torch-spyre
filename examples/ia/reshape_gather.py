import torch

x = torch.rand(3, 1024, dtype=torch.float16)
i = torch.tensor((1, 2), dtype=torch.int32)


def kernel(x, i):
    return x.reshape(12, 256)[i]


ref = kernel(x, i)

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