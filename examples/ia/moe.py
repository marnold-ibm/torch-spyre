import torch


def moe(expert_w: torch.Tensor, expert_ids: torch.Tensor) -> torch.Tensor:
    # Mixture of experts: select each token's expert weight matrix.
    #   expert_w   [E, D, F]   per-expert weight matrices
    #   expert_ids [B, S]      chosen expert per token
    #   out        [B, S, D, F]
    return expert_w[expert_ids]


def check(fn, *args):
    # Run fn on CPU and on Spyre (via torch.compile) and compare values.
    cpu_out = fn(*args)
    spyre_args = [a.to(device="spyre") for a in args]
    spyre_out = torch.compile(fn)(*spyre_args)
    torch.testing.assert_close(cpu_out, spyre_out.cpu())
    print(f"{fn.__name__}: {list(cpu_out.shape)} OK")


if __name__ == "__main__":
    B, S = 2, 128
    E, Dm, F = 8, 512, 2048
    expert_w = torch.randn(E, Dm, F, dtype=torch.float16)
    expert_ids = torch.randint(0, E, (B, S), dtype=torch.int64)
    check(moe, expert_w, expert_ids)