import torch

# Three use cases of indirect access (gather along the leading dim of a table).
# Each example isolates only the indexing op and the shapes involved.


def paged_kv(keys: torch.Tensor, slot_idxs: torch.Tensor) -> torch.Tensor:
    # Paged attention: gather KV-cache slots for each sequence position.
    #   keys      [cache, H, D]   physical KV cache
    #   slot_idxs [B, Lk]         slot index per (batch, key position)
    #   out       [B, Lk, H, D]
    return keys[slot_idxs]


def check(fn, *args):
    # Run fn on CPU and on Spyre (via torch.compile) and compare values.
    cpu_out = fn(*args)
    spyre_args = [a.to(device="spyre") for a in args]
    spyre_out = torch.compile(fn)(*spyre_args)
    torch.testing.assert_close(cpu_out, spyre_out.cpu())
    print(f"{fn.__name__}: {list(cpu_out.shape)} OK")


if __name__ == "__main__":
    B = 2
    cache, H, Dh, Lk = 32768, 8, 128, 256
    keys = torch.randn(cache, H, Dh, dtype=torch.float16)
    slot_idxs = torch.randint(0, cache, (B, Lk), dtype=torch.int64)
    check(paged_kv, keys, slot_idxs)
