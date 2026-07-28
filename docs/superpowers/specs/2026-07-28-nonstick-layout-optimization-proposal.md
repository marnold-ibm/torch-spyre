# Non-Stick Dimension Layout Optimization

## Overview

The Spyre layout optimizer currently selects an output `SpyreTensorLayout` for
each op that minimizes the total restickify cost needed to satisfy stick
constraints across the graph. This proposal extends the optimizer
to also reason about non-stick dimension ordering, which affects both
correctness (gather/scatter) and performance (matmul). The extension preserves
the existing optimizer abstraction and pass structure, adding a backward
preference pre-pass and a parallel cost node hierarchy for non-stick
constraints.

This is a straightforward extension of the existing design, not a fundamental change. Non-stick dim ordering is another dimension of the same layout assignment problem the optimizer already solves. While considering all possible non-stick dimension orderings would introduce an impractical expansion of the candidate space, most of these choices can be resolved cheaply at the local producer. The backward preference pre-pass exploits this property to keep the candidate set bounded while preserving the optimizer's ability to select effective layouts.

This work is not intended to replace future global optimization efforts. Rather, it addresses short- and 
medium-term performance opportunities while also serving as a practical exploration of optimization techniques and 
abstractions. The implementation and lessons learned here are expected to inform and, where appropriate, be reusable in the design of a future global optimizer.

## Motivation

A `SpyreTensorLayout` encodes both a stick dimension and an ordering of the
remaining (non-stick) device dimensions. Today the optimizer only varies the
stick dimension — non-stick ordering is left at whatever `propagate_layouts`
happens to produce. Three known cases where non-stick ordering matters:

- **Matmul**: the non-stick dim ordering affects memory access performance.
  Suboptimal ordering reduces throughput on memory-bound problem sizes.
  (Exact rules TBD pending discussion with team.)

- **Gather**: the indirect-access dimension must be the outermost device
  dimension on the input. If it is not, a restickify is required before the
  gather op, at the same cost as any other restickify.

- **Scatter**: the indirect-access dimension must be outermost on the *output*.
  This is a producer-side hard constraint.

Currently there is no mechanism to communicate these preferences upstream or to
price them in the optimizer's cost model.

## Key Property

Most ops can freely choose the non-stick dimension ordering of their outputs. The current exceptions are scatter, which has a hard output constraint, and matmul, which has a performance preference. Reordering non-stick dimensions may incur locality costs, but these are assumed to be small relative to costs such as copies or suboptimal matmul performance. This assumption motivates propagating preferences only one hop in the initial implementation: we expect this to be sufficient for the common case while keeping the implementation simple and the candidate set bounded.  If the assumption proves incorrect and deeper propagation is needed, extending the propagation depth is straightforward and requires no algorithmic changes. The resulting increase in candidate sets may, however, increase state space pressure, necessitating empirical evaluation of whether the current pruning heuristics remain sufficient.

## Design

### Pass Pipeline

The existing pipeline:

```
propagate_layouts (forward)
→ optimize_restickify (backward A* + forward beam)
→ finalize_layouts + insert_restickify
```

Becomes:

```
preference_pre_pass (backward, new)
→ propagate_layouts (forward, augmented)
→ optimize_restickify (backward A* + forward beam, unchanged)
→ finalize_layouts + insert_restickify (unchanged)
```

### Backward Preference Pre-Pass (new)

A new backward pass walks the graph in reverse and propagates non-stick dim
preferences from consumers to producers. Each op is annotated with the set of
non-stick dim preferences expressed by its downstream consumer(s), which
`propagate_layouts` then uses to generate the appropriate candidate STLs.

Preferences are expressed via device coordinates and loop variable symbols
extracted from `MemoryDep.index` — the same mechanism used throughout
`propagate_layouts` and `pass_utils.py`.

Propagation rules:
- Preferences flow backward. Because nearly every op can freely reorder its non-stick output dimensions, a preference can typically be resolved in a single hop. If the immediate upstream op cannot satisfy the preference due to a hard output constraint, propagation terminates there. We assume that further propagation provides limited additional benefit; if results show otherwise, that provides useful data for revisiting this assumption.
- Propagation terminates at ops with hard output non-stick constraints (scatter). The consumer of scatter may therefore incur a restickify cost.
- At join points where multiple consumers express different preferences, the implementation generates the Cartesian product of stick × non-stick choices, allowing the optimizer to select the best combination. A configurable maximum candidate count per op acts as a safety valve: if the limit is reached, generation stops, a warning is emitted, and additional candidates are not generated. Whether this limit is reached in practice will be measured and monitored.


### Augmented propagate_layouts (forward)

`propagate_layouts` is augmented to consume preference annotations. When
generating candidate STLs for an op, it uses the annotations to guide non-stick
dim ordering:

- **Single preference**: rewrite the non-stick ordering of existing candidates
  in-place (no blowup).
- **Multiple preferences**: generate the cartesian product of stick choices ×
  non-stick preferences, up to the configured candidate count cap.

The existing stick-only behavior is fully preserved. Preference annotations
cannot remove a stick-dimension candidate; they only guide or augment non-stick
ordering.

If a preferred candidate STL cannot be constructed for any reason (e.g. the
preference is incompatible with the tensor's shape or layout constraints), a
warning is emitted and the preference is skipped. The op's existing candidates
are left unchanged. The downstream op will incur whatever copy or restickify
cost it would have without the optimization. This is a performance issue, not a
correctness issue — the graph will still produce correct results.

### Candidate Representation

Each candidate STL in `op.layouts` is fully concrete: both stick dim and
non-stick dim ordering are specified. The optimizer sees concrete candidates
with concrete costs — no special cases needed.

The candidate set is the cartesian product of stick choices × non-stick
choices, but only preference-driven combinations are generated, not all N!.

### Cost Node Architecture

The existing `RestickNodeCost` hierarchy (`AllSameNode`, `FixedInOutNode`,
`AnyInNode`) handles stick constraints only and is unchanged.

A parallel `NonStickNodeCost` hierarchy is introduced:

- `AnyInNonStick` — no non-stick constraint (default for all existing ops)
- `PreferredInNonStick` — non-stick ordering preference with heuristic cost penalty (matmul)
- `RequiredInNonStick` — non-stick ordering required, else restickify cost (gather)

A factory function `make_cost_node(stick, nonstick=AnyInNonStick())` wires the
two together into a single object the beam calls as today. The combined `cost()`
returns a single float with no double-counting: if both stick and non-stick
dims are wrong, a single restickify fixes both, so cost is not doubled.

`compute_restickify_needed` in `pass_utils.py` is not changed — it remains
stick-only. Non-stick cost is computed separately in the non-stick node and
combined in `make_cost_node`.

Call sites in `propagate_layouts`:

```python
# existing ops — behavior unchanged
op.restick_cost_fn = make_cost_node(
    stick=AllSameNode.from_args(args, results, output_dep, op),
)

# matmul — adds soft non-stick preference
op.restick_cost_fn = make_cost_node(
    stick=FixedInOutNode.from_args(args, out_stl, [x_req, y_req], op),
    nonstick=PreferredInNonStick.from_args(...),
)

# gather — adds hard non-stick requirement
op.restick_cost_fn = make_cost_node(
    stick=AllSameNode.from_args(args, results, output_dep, op),
    nonstick=RequiredInNonStick.from_args(...),
)
```

`FixedInOutNode` is currently used by `_clone_layout`, `_exx2_layout`,
`_layernormnorm_layout`, and `_matmul_layouts`. Only matmul gets a non-default
non-stick node; the others default to `AnyInNonStick`.

### Interaction with Existing Optimizer Passes

**Beam / liveness merge**: unchanged. The optimizer abstraction holds: it
starts with a set of output STLs per op and reduces them to singletons using
cost functions. Non-stick dim ordering is simply part of what distinguishes two
candidate STLs. The liveness merge operates on full assignment tuples and is
correct regardless of candidate set size.

**Backward A\* pass**: unchanged structurally. With the extended cost model,
`min_input_cost` returns better lower bounds for candidates that satisfy
downstream non-stick preferences — states with poor non-stick ordering get
higher future cost estimates and are pruned earlier. The heuristic improves;
no algorithmic change.

**Finalize layouts + insert_restickify**: The call `edge.layout(in_stl,
target_stl)` (currently on `EdgeCostMap`, returns restickify target STL or
`None`) becomes `edge.fix_needed(in_stl, target_stl)`, which determines
whether a restickify is needed and returns the fully concrete target STL (stick
and non-stick dims both specified), or `None` if already compatible. A
restickify fixes both stick and non-stick dims in a single pass — no new op
type is needed. `finalize_layouts` is structurally unchanged.

`required_input_stls` must return fully concrete STLs (stick and non-stick dims
both specified) so `fix_needed` has enough information to compute the correct
restickify target.


## Cost Model

The current optimizer uses tensor element count as a proxy for restickify cost.
This is sufficient when only stick constraints exist — all restickifies have the
same relative cost per element — but breaks down once non-stick constraints are
introduced, since a restickify and a matmul performance penalty are not
comparable in element-count terms.

The cost model must be changed to express all costs in **estimated time**
(e.g. microseconds). This puts restickify costs and non-stick preference
penalties on a common scale so the optimizer can make meaningful tradeoffs
between them.

The first priority is to establish the **API contract**: each cost function
takes tensor size (and any other relevant properties) as input and returns a
time estimate. The internal formula can start simple and evolve as hardware
measurements become available. One cost function is defined per operation type:

| Operation | Cost function | Notes |
|---|---|---|
| Restickify | `restickify_cost(size)` | stick dimension wrong |
| Non-stick copy | `copy_cost(size)` | non-stick dims wrong, stick correct |
| Matmul suboptimal layout | `matmul_suboptimal_cost(size)` | soft penalty for wrong non-stick order |

When both stick and non-stick dims are wrong, a single restickify fixes both —
cost is `restickify_cost(size)`, not a sum of the two.

## Extensibility

The three cases covered here (matmul, gather, scatter) are not an exhaustive
list — there will likely be others as the hardware and compiler mature. The
framework is designed so that new non-stick constraints can be added without
architectural changes: each new case is a new `NonStickNodeCost` subclass, a
termination rule in the preference pre-pass if needed, and a cost function in
the cost model.

As one example of a future case: if a transformation requires a host-side
layout conversion for some ops (temporarily or otherwise), its cost can be
modeled as a cost function and plugged into the same framework, letting the
optimizer weigh it against restickify and matmul costs on the same time scale.
This is intended to be illustrative, not a complete list of future extensions.
