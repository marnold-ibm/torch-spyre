import torch

# Three use cases of indirect access (gather along the leading dim of a table).
# Each example isolates only the indexing op and the shapes involved.


def embedding(table: torch.Tensor, token_ids: torch.Tensor) -> torch.Tensor:
    # LLM embedding lookup: select a row per token.
    #   table     [V, D]      vocab x embed_dim
    #   token_ids [B, S]      int token ids
    #   out       [B, S, D]
    return table[token_ids]



def check(fn, *args):
    # Run fn on CPU and on Spyre (via torch.compile) and compare values.
    cpu_out = fn(*args)
    spyre_args = [a.to(device="spyre") for a in args]
    spyre_out = torch.compile(fn)(*spyre_args)
    torch.testing.assert_close(cpu_out, spyre_out.cpu())
    print(f"{fn.__name__}: {list(cpu_out.shape)} OK")


if __name__ == "__main__":
    V, D, B, S = 32000, 512, 2, 128
    table = torch.randn(V, D, dtype=torch.float16)
    token_ids = torch.randint(0, V, (B, S), dtype=torch.int64)
    check(embedding, table, token_ids)
