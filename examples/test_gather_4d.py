import torch
from torch_spyre._C import SpyreTensorLayout

torch.manual_seed(0)

# Shape: [A, B, C, D]
A, B, C, D = 5, 8, 16, 512  # D=512 is stick-aligned (64 elems/stick * N)
dtype = torch.float16


def test_gather_dim0():
    Q = 12  # number of gathers; Q < A so we pick real rows
    i = torch.randint(0, A, (Q,), dtype=torch.int32)
    x = torch.rand(A, B, C, D, dtype=dtype)

    stl = SpyreTensorLayout(list(x.size()), list(x.stride()), dtype, [1, 0, 2, 3])

    def fn(x, i):
        return x[i]

    ref = fn(x, i)

    x_dev = x.to("spyre", device_layout=stl)
    i_dev = i.to("spyre")
    compiled = torch.compile(fn)
    out = compiled(x_dev, i_dev).cpu()

    torch.testing.assert_close(ref, out, atol=1e-2, rtol=1e-2)
    print(
        f"[PASS] gather dim 0: x{list(x.shape)}[i{list(i.shape)}] -> out{list(out.shape)}"
    )


def test_gather_dim1():
    Q = 20
    i = torch.randint(0, B, (Q,), dtype=torch.int32)
    x = torch.rand(A, B, C, D, dtype=dtype)

    stl = SpyreTensorLayout(list(x.size()), list(x.stride()), dtype, [0, 1, 2, 3])

    def fn(x, i):
        return x[:, i, :]  # out shape: [A, Q, C, D]

    ref = fn(x, i)

    x_dev = x.to("spyre", device_layout=stl)
    i_dev = i.to("spyre")
    compiled = torch.compile(fn)
    out = compiled(x_dev, i_dev).cpu()

    torch.testing.assert_close(ref, out, atol=1e-2, rtol=1e-2)
    print(
        f"[PASS] gather dim 1: x{list(x.shape)}[:, i{list(i.shape)}, :] -> out{list(out.shape)}"
    )


def test_gather_dim2():
    Q = 32  # number of gathers; Q > C
    i = torch.randint(0, C, (Q,), dtype=torch.int32)
    x = torch.rand(A, B, C, D, dtype=dtype)

    stl = SpyreTensorLayout(list(x.size()), list(x.stride()), dtype, [0, 2, 1, 3])

    def fn(x, i):
        return x[:, :, i, :]  # out shape: [A, B, Q, D]

    ref = fn(x, i)

    x_dev = x.to("spyre", device_layout=stl)
    i_dev = i.to("spyre")
    compiled = torch.compile(fn)
    out = compiled(x_dev, i_dev).cpu()

    torch.testing.assert_close(ref, out, atol=1e-2, rtol=1e-2)
    print(
        f"[PASS] gather dim 2: x{list(x.shape)}[:, :, i{list(i.shape)}, :] -> out{list(out.shape)}"
    )


if __name__ == "__main__":
    test_gather_dim0()
    test_gather_dim1()
    test_gather_dim2()
