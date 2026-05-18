# Loop-Level IR Pass Refactoring: Recommendations

## Background

There are three optimization passes that manipulate the loop-level IR
directly — inserting new nodes, rewriting inner functions, and updating
dependencies:

1. **Insert Restickify** (`torch_spyre/_inductor/insert_restickify.py`) —
   inserts layout-conversion nodes before consumers whose inputs have
   incompatible stick layouts
2. **Chunk Large Tensors** (`torch_spyre/_inductor/chunk_large_tensors.py`) —
   splits oversized pointwise ops into memory-safe chunks
3. **Matmul Padding** (`torch_spyre/_inductor/padding.py` +
   `torch_spyre/_inductor/pass_utils.py`) — pads matmul inputs to
   stick-aligned sizes before computation

A fourth pass, **Dedup Constants**
(`torch_spyre/_inductor/dedup_constants.py`), also manipulates the IR and
is relevant to one of the findings below.

---

## What Should Be Refactored

### 1. Extract `patch_inner_fn` into `pass_utils.py`

**Affects:** Insert Restickify, Dedup Constants

Both passes need to redirect which upstream buffer a node reads from —
Insert Restickify because it inserts a new restickified copy of a buffer
and needs the consumer to read from it instead; Dedup Constants because it
eliminates a duplicate constant and needs all consumers to read the
canonical one instead.

Both independently implement the same 3-step protocol:

1. Wrap `inner_fn` with a `NameSwapHandler` that intercepts
   `load("old_name", index)` and redirects to `load("new_name", index)`
2. Patch the frozen dataclass field via
   `object.__setattr__(op.data, "inner_fn", new_inner_fn)`
3. Clear the `get_default_sizes_body` cache so stale results are not reused

Dedup Constants already extracted this as a local private function. Insert
Restickify has it inlined. The `NameSwapHandler` class itself lives in
`insert_restickify.py`, which means Dedup Constants imports a utility from
a pass — a backwards dependency.

**Recommendation:** Move `NameSwapHandler` and the 3-step function into
`pass_utils.py` as `patch_inner_fn(op, name_map)`. Both callers import
from there.

**Why it matters:** Step 3 (cache clear) is easy to forget and its
omission causes silent stale-sizing bugs. One canonical function enforces
the full protocol. The backwards import dependency is also eliminated.

---

### 2. Use `replace_computed_buffer_body` in Insert Restickify

**Affects:** Insert Restickify

`pass_utils.py` already has `replace_computed_buffer_body()` — it
constructs a fresh `ComputedBuffer` with an updated body, copies all
metadata fields (`operation_name`, `origins`, `origin_node`, all
`_original_*` fields), clears the cache, and swaps it into the operations
list. It exists because Matmul Padding needed it and extracted it there.

Insert Restickify does the exact same thing inline (~15 lines), with one
extra step: updating `V.graph.name_to_buffer`. It was never updated to use
the shared helper.

**Recommendation:** Replace the inline duplicate in
`insert_restickify.py` with a call to `replace_computed_buffer_body`,
followed by the one extra `name_to_buffer` update line.

**Why it matters:** If a new `_original_*` metadata field is ever added to
`ComputedBuffer`, only `replace_computed_buffer_body` would need updating
— currently there is a silent second copy that would drift out of sync.

---

### 3. Extract `insert_ops_before` into `pass_utils.py`

**Affects:** Insert Restickify, Matmul Padding

After `run_node()` appends new ops to the tail of the operations list,
both passes must relocate them immediately before the consumer op to
preserve topological order. Both implement this manually with slightly
different code.

**Recommendation:** Add `insert_ops_before(new_ops, consumer, operations)`
to `pass_utils.py`. It returns the new index of the consumer after
insertion. Both callers replace their manual remove-then-insert loops with
a single call.

**Why it matters:** The topological-order invariant is easy to break
subtly. A single well-tested implementation is better than two slightly
different ones.

---

## What Should Be Left Alone

**Chunk Large Tensors** has no overlap with any of the above. It creates
buffers directly (no `run_node()`), so new ops never end up appended at
the tail needing relocation. It modifies inner functions by adjusting index
offsets rather than redirecting buffer names. It already has its own
well-contained `_register_and_insert` helper locally. Nothing in this pass
benefits from the proposed helpers.

The `object.__setattr__()` calls for mutating frozen dataclass fields
appear throughout all passes but should not be abstracted. This is a
standard PyTorch idiom that every reader already understands, and wrapping
it would add indirection with no clarity benefit.

---

## Summary

| Refactoring | Passes affected | Key benefit |
|---|---|---|
| Extract `patch_inner_fn` + move `NameSwapHandler` to `pass_utils` | Insert Restickify, Dedup Constants | Enforces 3-step protocol; fixes backwards import |
| Use `replace_computed_buffer_body` in Insert Restickify | Insert Restickify | Eliminates metadata field drift trap |
| Extract `insert_ops_before` to `pass_utils` | Insert Restickify, Matmul Padding | Single implementation of topological-order invariant |
| Chunk Large Tensors | — | No changes needed |
