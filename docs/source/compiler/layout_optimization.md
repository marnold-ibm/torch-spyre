# Layout Propagation and Restickify Optimization

This document describes the three compiler passes that assign `SpyreTensorLayout`
(STL) values to every op in the Inductor graph and insert the restickify
operations needed to satisfy hardware stick-compatibility constraints.

The three passes are:

| Pass | File | Purpose |
|------|------|---------|
| `propagate_spyre_tensor_layouts` | `propagate_layouts.py` | Forward propagation: assign *sets* of candidate STLs to each op's output |
| `optimize_restickify_locations` | `optimize_restickify.py` | Layout selection: reduce each candidate set to one committed STL, minimizing total restickify cost |
| `finalize_layouts` + `insert_restickify` | `insert_restickify.py` | Materialize: convert committed STLs to `FixedTiledLayout`, build the restickify insertion plan, and splice restickify ops into the graph |

---

## Background: Sticks and Restickify

The Spyre hardware memory model requires all tensor operands within a single
kernel to share the same *stick variable*. A stick is a 128-byte aligned
chunk of contiguous elements in device memory — 64 elements at `fp16`, 32 at
`fp32`. The *stick dimension* is the innermost device dimension; the *stick
variable* is the loop variable whose expression modulo the stick depth appears
in the innermost device coordinate.

Each operation imposes specific stick constraints on its inputs and outputs.
If those constraints are not met, the operation will not execute correctly or
will produce incorrect output. A *restickify* op must be inserted to produce a
new buffer whose stick satisfies the operation's requirements.

A `SpyreTensorLayout` (STL) fully describes how a tensor is stored in device
memory:

- `device_size` — shape of the device buffer (may differ from the host logical
  shape after stick-dimension transposition or padding)
- `stride_map` — how each device dimension corresponds to host strides (set to
  `-1` for size-1 or sparse axes)
- `device_dtype` — the on-device data format (e.g. `SEN169_FP16`)
- `element_arrangement` (`ElementArrangement`) — standard, staggered
  (`DL16_TO_FP32` / `FP32_TO_DL16` for RMSNorm upcasts), `QFP8CH`, `QFP8WT`
- The **stick is always the last device dimension**.

---

## Pass 1 — Layout Propagation (`propagate_spyre_tensor_layouts`)

### What it does

Layout propagation is a forward data-flow pass over the op graph. At each node
it takes the candidate STL sets already attached to the input buffers and
derives the set of candidate STLs for that node's output. Passing sets
forward — rather than committing to one layout immediately — is what gives the
downstream optimizer room to make a globally-better choice.

The current code does not propagate all possible layouts. It intentionally excludes layouts that are unlikely to be helpful, 
to keep the solution space manageable for the downstream optimizer. The downstream optimizer has no option to select 
new stick configurations not added as candidates by propagate_layouts. As a result, excluding these layouts has the potential 
to prevent us from finding the globally optimal solution. `propagate_layouts` should be modified to add additional 
candidate STLs if we find evidence that they are needed, or if the optimizer improves to the point that the larger state space 
becomes inconsequential to runtime.

### Candidate generation

Graph inputs are seeded from the actual on-device tensor layout of the graph
input tensor. Each op is then assigned a `restick_cost_fn` — one of three cost
nodes that encodes its stick-compatibility rules. The optimizer works on these
cost nodes rather than re-examining individual PyTorch ops.

### Cost node types

Three node types encode the different stick-compatibility rules:

**`AllSameNode`** — used for pointwise ops and most reductions. All inputs
and the output must share a stick. The cost is the sum of per-input
restickify costs to reach the chosen output STL. Each input targets the same
output STL.

**`FixedInOutNode`** — used for matmul, conv, exx2, and layernormnorm. The
output STL is fixed; any other choice costs `INF`. Each input has its own
required target STL (e.g. matmul's two inputs require different stick
variables).

**`AnyInNode`** — used for clone, no-ops, constants, and similar. Accepts
any input at zero cost and contributes nothing to the backward DP.

#### Single-arg ops

The default strategy is to preserve the input stick as the output stick —
threading the same loop variable through the op costs nothing. If the input
stick expression maps cleanly to an output dimension and produces an
offset-free output stick, that single candidate is returned. If it doesn't
(e.g. the dimension disappears, or the mapped output stick acquires a constant
offset from slicing), the pass scans all output dimensions as alternatives.

**Reductions** have one important exception: some reduction ops
(`REDUCTIONS_NON_STICK_DIM_ONLY`) cannot reduce along the stick dimension.
When the input stick carries the reduction variable, the input layout is not
preserved — only surviving (non-reduced) dimensions are offered as output
stick candidates.

**Type conversions** that change `elems_per_stick` (e.g. fp16↔fp32,
fp8→fp16) need special handling because the stick depth changes. Staggered
element arrangements (DL16 ↔ FP32, used for RMSNorm upcasts) propagate by
rescaling the device layout's stick depth in place, preserving any padding in
the input layout. Plain conversions rebuild a fresh dense layout from the
output host size, which avoids propagating degenerate device sizes that can
appear after QFP8 quantization rescaling.

#### Multi-arg pointwise

This is the main join point in the graph: multiple inputs must share a stick
variable to execute in the same kernel. The output stick is chosen to be
reachable from all inputs without requiring a restickify, if any such choice
exists. The pass collects the offset-free stick expressions from all input
candidates, tries each as the output stick, and keeps only the ones that
project back cleanly onto every input without introducing an offset. If none
of the input sticks survive this filter, all output dimensions are tried as
alternatives.

**Same-layout pass-through.** When all inputs are accessed identically to the
output — same size, stride, index expression, and element size — the output
device layout is copied from the input exactly as-is, including the
`ElementArrangement` and all non-stick dimension ordering. This matters beyond
performance: it guarantees that when the existing layout already satisfies all
constraints, the pass makes no changes to it. Reconstruction from host
size/stride would produce a valid layout but not necessarily the same one,
potentially introducing a spurious restickify or breaking in-place allocation.

**In-place promotion.** When any input has the exact same size, stride, and
index access pattern as the output, its layout is moved to the front of the
candidate list. This is a deliberate tie-breaking signal for the optimizer: on
a cost tie, the front candidate wins, which lets the memory allocator write
the result in-place into that input's buffer — an important optimization for
LX scratchpad planning. Note the distinction from same-layout pass-through:
that applies when *all* inputs match and copies the full device layout
verbatim; in-place promotion applies when *any* input matches and only
reorders the candidate list.

**ElementArrangement propagation.** Most ops use the `STANDARD` arrangement.
When a staggered EA (e.g. `DL16_TO_FP32`) is present on any input, it
propagates to the output and `STANDARD` candidates are suppressed. A
`STANDARD` input can participate alongside a staggered one only if its device
stick maps to a size-1 (broadcast) axis — otherwise the mixed arrangement
cannot be expressed.

#### Fixed-requirement ops (matmul, conv, exx2, layernormnorm)

These ops have hardware-mandated stick requirements that leave no room for the
optimizer to choose:

- **Matmul**: input1 must stick on the reduction variable (K); input2 and the
  output must stick on the generated variable (N).
- **Conv2d**: same structure — activation sticks on the input-channel
  contraction variable, weight and output stick on the out-channel variable.
- **exx2 / layernormnorm**: input must stick on the last (reduction) dimension.

Because the required sticks are fixed, the pass finds a compatible input
layout for each operand — using the existing candidate if it already has the
right stick, or computing a restickify target if not — and installs a
`FixedInOutNode` that returns `INF` cost for any other input/output
combination. The optimizer has no freedom here; its only job is to pay for any
restickify needed to reach the required layout.

#### `aten.clone`

Clone is unusual: it always materializes a new buffer in row-major layout
regardless of the input stick, which means it *is* a restickify whenever the
sticks differ. The pass therefore attaches `AnyInNode` — zero cost for any
input layout — because no additional op ever needs to be inserted before a
clone.

The exception is when the input stick has a constant offset (e.g. from slicing
into the stick dimension). In that case the offset must be resolved before the
clone can absorb the transpose, so a restickify *is* needed upstream, and
`FixedInOutNode` is used to enforce the required pre-clone layout.

#### Slice views

Views that slice into a base tensor (e.g. `x[1:]`) are handled specially: the
layout is rewritten to carry the base tensor's size/stride with the view offset
attached, and the pass verifies that the storage offset is device-stick-aligned
before proceeding.

<!-- TODO: expand this section -->

---

## Pass 2 — Restickify Optimization (`optimize_restickify_locations`)

### What it does

Given the candidate STL sets and cost nodes from propagation, the beam
optimizer commits one STL per op (written to `op.committed_stl`) to minimize
total restickify cost.

### STL candidates as stick-dimension proxies

An important subtlety: the candidate STLs produced by propagation vary only
in **which host dimension is mapped to the stick**. The non-stick dimension
ordering in a candidate STL is chosen reasonably but is not something the
optimizer reasons about or optimizes over. In effect, each candidate is a
proxy for the set of all valid layouts that place the stick on a particular
loop variable — the non-stick dimensions are along for the ride.

The optimizer works correctly under this interpretation because the cost
function it calls — `compute_restickify_needed` / `stick_compatible` in
`pass_utils.py` — only checks **stick compatibility**, not full STL equality.
`stick_compatible()` returns true when all tensors' stick expressions share at
most one loop variable and that variable doesn't appear in any non-stick
coordinate. Two STLs with different non-stick orderings but the same stick
variable are therefore judged compatible and assigned cost 0, even though they
are not identical objects.

The practical consequence is that the optimizer's cost landscape is essentially
binary per edge: 0 (stick-compatible, no restickify) or element-count
(restickify needed). There is no gradient over non-stick layout choices today.

This will need to change in the future. When the backend gains the ability to
express preferences over non-stick dimension orderings — for example, to
exploit cache locality or memory bandwidth differences between layouts — the
cost function would need to return non-zero costs for stick-compatible but
suboptimally-ordered STL pairs. At that point the propagation pass would also
need to emit multiple candidates per stick location (varying non-stick
orderings), and the cost model would need to distinguish "compatible but
suboptimal" from "needs restickify."

### Cost model

Each input edge carries an `EdgeCostMap` that lazily computes the cost of
pairing a producer's candidate STL with a consumer's candidate output STL.
The result is one of three values: 0 if the sticks are already compatible,
the total device element count of the input buffer if a restickify is needed
(since restickify runtime is broadly linear in size), or `INF` if no
restickify can make them compatible. Results are cached so the optimizer can
query the same pair repeatedly without recomputation.

`EdgeCostMap` also records the *target layout* for each feasible restickify —
the STL the restickify op should produce. This is consumed later by
`finalize_layouts` when it builds the insertion plan.

The optimizer uses the same three cost node types (`AllSameNode`,
`FixedInOutNode`, `AnyInNode`) defined in Pass 1. See
[Cost node types](#cost-node-types) above.

---

### Global beam optimizer

#### What it does

The optimizer must assign one STL to every op in the graph such that the
total restickify cost is minimized. The choices interact: picking a particular
stick at one op affects what sticks are compatible at downstream join points.
Exact search over all combinations is exponential. The beam optimizer solves
this by maintaining a population of up to `BEAM_WIDTH` hypotheses — each a partial
assignment of STLs covering all ops seen so far — and advancing
them through the graph in topological order. At each op, every hypothesis
branches into one new hypothesis per candidate STL. At the end, the hypothesis
with the lowest total cost wins and its assignments are committed.

#### The beam width tradeoff

The beam is the fundamental accuracy/tractability tradeoff. A wider beam is
more likely to find the global optimum; a narrower beam finishes faster but
may discard the optimal path early. Any time the optimal assignment gets
pruned from the beam — because it looked worse than `BEAM_WIDTH` other
hypotheses at some intermediate step — the optimizer will not find it. The
quality of the result therefore depends on how well the beam can be guided to
keep promising hypotheses and eliminate hopeless ones as early as possible.

#### Pruning optimizations

**Liveness merge.** After expanding at each op, two hypotheses that differ
only in the STLs of buffers that no future op will ever read have identical
futures. The one with the higher accumulated cost is strictly dominated — it
cannot produce a better final answer than the cheaper one regardless of future
choices — and is discarded immediately. This is not an approximation:
dominated hypotheses can never win. In graphs with many short-lived
intermediates, liveness merge can eliminate a large fraction of hypotheses
before the beam trim even fires, keeping beam capacity free for decisions that
still matter.

**Backward DP lower bound.** Before the forward pass runs, a backward pass
walks the graph in reverse topological order and computes, for every
`(op, candidate_stl)` pair, the minimum remaining restickify cost achievable
from that choice onward. This becomes the "future" component of each
hypothesis's lower bound: `lower_bound = cost_so_far + future_min_cost`. The
beam is sorted and trimmed by lower bound rather than actual cost, so a
hypothesis whose current choices lead toward an expensive or infeasible join
downstream is de-prioritized even if its cost-so-far looks cheap.

The bound is admissible — it never overestimates — because each downstream
consumer's minimum cost is computed independently, ignoring cross-consumer
constraints. That conservatism is deliberate: an overestimating bound could
prune the optimal path from the beam.

---

## Pass 3 — Finalization and Restickify Insertion

### `finalize_layouts`

Finalization is a two-phase commit. First, every op's `committed_stl` is
frozen into a `FixedTiledLayout` and assigned as the op's permanent layout —
optimizer-only attributes are cleaned up. Then, now that all layouts are
frozen, each input edge is inspected: if the producer's committed STL is
incompatible with what the consumer requires, a restickify is recorded in
`graph.restickify_plan`. The ordering matters — you need all committed layouts finalized before you
can determine which edges require a restickify.

Two special cases during the commit phase: tiled-reduction accumulator buffers
receive the same layout as their reduction op so that all sub-ops (fill,
combine, copy) agree on the device coordinate system; and mutation ops
targeting a `SpyreEmptyFallback` accumulator have the accumulator's layout
overwritten to match the committed STL.

### `insert_restickify`

Consumes `graph.restickify_plan` and splices the required restickify ops into
`graph.operations` before their consumers. Each restickify is lowered as a
real `spyre.restickify` IR node. The consumer op's `inner_fn` is then patched
via `NameSwapHandler` — a `WrapperHandler` subclass that intercepts buffer
loads and redirects them to the new restickified buffer, without touching any
index expressions. The consumer `ComputedBuffer` is reconstructed as a fresh
object to invalidate any cached size/body derived from the old inputs.

### `insert_post_mutation_restickify`

Handles a narrower special case: a slice-mutation into a graph input whose
storage offset falls mid-stick, requiring the input to be in an alternative
layout before the mutation can proceed. Because a restickify cannot write
in-place into the original buffer, a temporary buffer is used. The pass
inserts a restickify before the mutation (original → temp), retargets the
mutation to write into the temp buffer's slice, then inserts a copy-back after
(temp → original storage via `MutationLayoutSHOULDREMOVE`). The copy-back has
identical input and output STLs and reduces to an identity copy in codegen.

---

## Error reporting

If no feasible output layout exists for an op (all candidates have `INF` cost),
`_no_feasible_layout_error` builds a detailed `NotImplementedError` containing:

- Each input buffer's host size/stride/coordinates and all its candidate STLs
  with their device coordinates
- The output buffer's layout and candidate STLs
- Per-candidate triage: which input edge is blocking and a human-readable
  reason (e.g. "No mechanism to gather elements from multiple sticks into
  single stick", "No mechanism to scatter elements from one stick to multiple
  sticks")

---

## Full pass sequence

```
propagate_spyre_tensor_layouts(graph)
    → op.layouts + op.restick_cost_fn for every op

optimize_restickify_locations(graph)
    → op.committed_stl for every op

finalize_layouts(graph)
    → op.layout = FixedTiledLayout(committed_stl) for every op
    → graph.restickify_plan = {op_name: [{arg_name, target_layout}, ...], ...}

insert_post_mutation_restickify(graph)
    → inserts pre/post ops for offset-mutation edge cases

insert_restickify(graph)
    → splices restickify ComputedBuffers before affected consumer ops
    → patches consumer inner_fn via NameSwapHandler
```

---

## Configuration

| Symbol | Default | Effect |
|--------|---------|--------|
| `BEAM_WIDTH` | `200` | Maximum beam states retained after each op expansion |
| `MAX_BEAM_STATES_LOGGED` | `10` | Number of states logged at DEBUG level per step |
