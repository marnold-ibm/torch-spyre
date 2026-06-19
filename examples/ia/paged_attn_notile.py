import math

import torch
from torch_spyre._inductor import config, spyre_hint
from torch_spyre._inductor.propagate_named_dims import (
    declare_tensor_dim,
    name_tensor_dims,
)

B, H, L, D = 12, 32, 256, 128
Lq = L
Lk = L
kv_block_size = 64
q_block_size = 64
Tk = Lk // kv_block_size
cache_size = 32768


def paged_cpu(
    queries: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    block_table: torch.Tensor,
    q_block_size,
    kv_block_size,
):
    scale = 1.0 / math.sqrt(math.sqrt(D))

    output = torch.zeros_like(queries).transpose(1, 2)
    real_max = torch.full(
        (B, H, Lq), float("-inf"), device=queries.device, dtype=torch.float16
    )
    denominator = torch.zeros((B, H, Lq), device=queries.device, dtype=torch.float16)

    for b_start in range(0, B, 2):
        b_end = b_start + 2
        for lq_start in range(0, Lq, q_block_size):
            lq_end = lq_start + q_block_size
            for h_start in range(0, H, 4):
                h_end = h_start + 4

                queries_tile = queries[b_start:b_end, lq_start:lq_end, h_start:h_end].permute((0, 2, 1, 3))
                real_max_tile = real_max[b_start:b_end, h_start:h_end, lq_start:lq_end]
                denominator_tile = denominator[b_start:b_end, h_start:h_end, lq_start:lq_end]
                output_tile = output[b_start:b_end, h_start:h_end, lq_start:lq_end]

                for block_slice in range(0, Tk):
                    block_ids = block_table[b_start:b_end, block_slice]
                    slot_idxs_start = block_ids * kv_block_size
                    slot_idxs = slot_idxs_start.unsqueeze(1) + torch.arange(0, kv_block_size, dtype=torch.int64, device=slot_idxs_start.device).unsqueeze(0)

                    keys_tile = keys[slot_idxs, h_start:h_end]
                    values_tile = values[slot_idxs, h_start:h_end]
                    keys_tile_T = keys_tile.permute(0, 2, 3, 1)

                    scores = torch.matmul(queries_tile * scale, keys_tile_T * scale)  # tile_b, tile_h, tile_lq, tile_lk
                    scores = scores.transpose(-1, -2).contiguous()  # avoid stick reduction
                    block_max = torch.amax(scores, dim=-2) # tile_b, tile_h, tile_lq
                    running_max = torch.maximum(real_max_tile, block_max) # tile_b, tile_h, tile_lq

                    exp_scores = torch.exp(scores - running_max.unsqueeze(-2))  # tile_b, tile_h, tile_lk, tile_lq
                    correction = torch.exp(real_max_tile - running_max) # tile_b, tile_h, tile_lq

                    denominator_tile.copy_(denominator_tile * correction + exp_scores.sum(dim=-2)) # tile_b, tile_h, tile_lq
                    output_tile.copy_(output_tile * correction.unsqueeze(-1) + torch.matmul(exp_scores.transpose(-1, -2), values_tile.permute(0, 2, 1, 3))) # tile_b, tile_h, tile_lq, D

                    real_max_tile.copy_(running_max)

    return output / denominator.unsqueeze(-1)


def paged_spyre(
    queries: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    slot_idxs: torch.Tensor,
    q_block_size,
    kv_block_size,
):
    scale = 1.0 / math.sqrt(math.sqrt(D))

    output = torch.zeros_like(queries).transpose(1, 2)
    real_max = torch.full(
        (B, H, Lq), float("-inf"), device=queries.device, dtype=torch.float16
    )
    denominator = torch.zeros((B, H, Lq), device=queries.device, dtype=torch.float16)

    ### INDIRECT ACCESS BELOW ###
    keys_tile = keys[slot_idxs] # tile_b, tile_lk, tile_h
    values_tile = values[slot_idxs] # tile_b, tile_lk, tile_h
    ### END INDIRECT ACCESS ###
    keys_T = keys_tile.permute(0, 2, 3, 1).contiguous()

    scores = torch.matmul(queries.permute((0, 2, 1, 3)) * scale, keys_T * scale)  # tile_b, tile_h, tile_lq, tile_lk
    scores = scores.transpose(-1, -2).contiguous()  # avoid stick reduction
    block_max = torch.amax(scores, dim=-2) # tile_b, tile_h, tile_lq
    running_max = torch.maximum(real_max, block_max) # tile_b, tile_h, tile_lq

    exp_scores = torch.exp(scores - running_max.unsqueeze(-2))  # tile_b, tile_h, tile_lk, tile_lq
    correction = torch.exp(real_max - running_max) # tile_b, tile_h, tile_lq

    denominator.copy_(denominator * correction + exp_scores.sum(dim=-2)) # tile_b, tile_h, tile_lq
    output.copy_(output * correction.unsqueeze(-1) + torch.matmul(exp_scores.transpose(-1, -2), values_tile.permute((0, 2, 1, 3)))) # tile_b, tile_h, tile_lq, D

    real_max.copy_(running_max)

    return output / denominator.unsqueeze(-1)


if __name__ == "__main__":
    queries_t = torch.randn(B, Lq, H, D, dtype=torch.float16)
    keys_t = torch.randn(cache_size, H, D, dtype=torch.float16)
    values_t = torch.randn(cache_size, H, D, dtype=torch.float16)

    block_table_t = torch.randint(0, 32768 // kv_block_size, (B, Tk), dtype=torch.int64)
    offsets = torch.arange(0, kv_block_size, dtype=torch.int64, device=block_table_t.device)
    slot_idxs_t = torch.zeros((B, Lk), dtype=torch.int64, device=block_table_t.device)
    for i in range(Tk):
        slot_start = i*kv_block_size
        slot_end = (i+1)*kv_block_size
        slot_idxs_t[:, slot_start:slot_end] = block_table_t[:, i].unsqueeze(1) * kv_block_size + offsets

    attn_t = paged_cpu(queries_t, keys_t, values_t, block_table_t, q_block_size, kv_block_size)

    declare_tensor_dim("B", B)
    declare_tensor_dim("H", H)
    declare_tensor_dim("Lq", Lq)
    declare_tensor_dim("Lk", Lk)
    declare_tensor_dim("cache", cache_size)
    declare_tensor_dim("Tk", Tk)
    declare_tensor_dim("D", D)

    queries_t_spyre = queries_t.to(device="spyre")
    keys_t_spyre = keys_t.to(device="spyre")
    values_t_spyre = values_t.to(device="spyre")
    slot_idxs_t_spyre = block_table_t.to(device="spyre")

    name_tensor_dims(queries_t_spyre, ["B", "Lq", "H", "D"])
    name_tensor_dims(keys_t_spyre, ["cache", "H", "D"])
    name_tensor_dims(values_t_spyre, ["cache", "H", "D"])
    name_tensor_dims(slot_idxs_t_spyre, ["B", "Lk"])

    c_flash_spyre = torch.compile(paged_spyre)
    attn_t_spyre = c_flash_spyre(queries_t_spyre, keys_t_spyre, values_t_spyre, slot_idxs_t_spyre, q_block_size, kv_block_size)
    torch.testing.assert_close(attn_t, attn_t_spyre.cpu(), atol=0.1, rtol=0.1)
