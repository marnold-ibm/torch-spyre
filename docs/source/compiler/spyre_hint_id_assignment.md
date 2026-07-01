# `spyre_hint` id assignment and the recompile trap

This note explains why `spyre_hint()` ids are assigned **after** tracing, and
documents several natural-looking approaches that do **not** work, so nobody
has to rediscover them.

## Background

`spyre_hint(**kwargs)` is a context manager users wrap around code to attach
coarse-tiling / work-division hints:

```python
with spyre_hint(tiles={"H": 4}):
    with spyre_hint(tiles={"Lk": 2}):
        ...
```

Each scope must get a distinct **hint id**. Downstream consumers
(`get_op_hints`, `coarse_tile`, `_hint_key`) require the ids to be:

- **unique** per distinct scope within a graph,
- **ordered outermost-first** — an outer scope's id must sort *before* the
  scopes it encloses (`get_op_hints`/`assign_dim_hints` do `sorted(...)`),
- **`< _SPAN_OVERFLOW_HINT_ID` (10000)** — that range is reserved for
  auto-generated span-overflow hints,
- **stable across recompiles of the same graph** — see below.

## The recompile trap

`spyre_hint()` runs **while Dynamo is tracing the user function**. Anything its
body reads from mutable state becomes part of Dynamo's world:

- Reading a **module global** (e.g. a `_hint_counter`) installs a *guard* on
  that global's value. The original implementation did `_hint_counter += 1`
  during tracing, so Dynamo guarded `_hint_counter == 0`. But tracing itself
  advanced the counter, so the guard could never pass again — the function
  **recompiled on every call**, with the counter drifting `0 → N → 2N → …`.
  (Symptom: `TORCH_LOGS=recompiles` shows
  `Recompiling ... _hint_counter == 0`, and each call re-runs the full
  Spyre/dxp backend compile.)

- Reading tracer state (`torch.fx.traceback.current_meta`) or inspecting
  `kwargs` with real logic (dict iteration, regex) makes Dynamo **trace into
  `spyre_hint` as a separate frame**, which then recompiles per call and
  fragments the graph (surfaces later as `no mechanism to resolve stick
  incompatibility`).

The original `f"_hint_{_hint_counter}"` avoided the *second* problem only
because it was simple enough for Dynamo to constant-fold; it still hit the
*first*.

**Key requirement that makes this subtle:** the same graph, compiled twice with
identical inputs, must produce the **same ids** (so the guards keep passing and
the cached graph is reused), while different graphs should not be forced to
share ids in a way that corrupts per-graph grouping.

## What does not work (verified dead ends)

1. **`@torch._dynamo.disable` on the id helper.** Always inserts a graph break —
   even for a pure, int-returning helper. Rejected: `spyre_hint` must not break
   the graph.

2. **Reset a global counter to 0 at the outermost scope.** Fixes the recompile
   and passes `test_flash.py`, but resetting fires again on every re-entry
   (AOT re-trace, sibling scopes), collapsing a graph's nested scopes onto the
   same id. Regressed `test_coarse_tile_e2e.py` (`DtException: out_reuse_dim`)
   when another graph had compiled first in the same process.

3. **Compute a content-addressed id inside `spyre_hint` during tracing**
   (hash of nesting + kwargs, or a memoized counter). Passes with
   `backend="eager"` but **fails on the real Inductor path**: Dynamo traces into
   the helper (reading `current_meta`, sorting `kwargs`), producing per-frame
   recompiles and stick-incompatibility. The eager backend masks this — always
   validate hint changes on the real device path.

## What works: assign ids post-trace

Split the responsibility:

- **In-trace (`spyre_hint`)** annotates with a **pure-content key** derived only
  from its own `kwargs`:

  ```python
  _SPYRE_HINT_PREFIX + repr(sorted(kwargs.items(), key=repr))
  ```

  This is constant-foldable — no global, no tracer state — so Dynamo installs no
  drifting guard and does not trace into it. Identical kwargs yield an identical
  key (stable across recompiles); distinct nested scopes yield distinct keys, so
  their annotations **accumulate** in `current_meta["custom"]` (a fixed key would
  make the inner scope clobber the outer).

- **Post-trace (`number_spyre_hints`, called from `collect_spyre_hints`)**
  rewrites those content keys into the ordered `_hint_N` form the rest of the
  compiler expects. There is no Dynamo guard/tracing concern here. Ordering
  comes from the fact that `annotate` inserts keys **outermost-first** in each
  node's `custom` dict; we take a stable topological order of the
  "appears-before on some node" relation, so an outer scope always gets a
  smaller id. Numbering is per-graph and deterministic.

Downstream code is unchanged — it still sees `_hint_1, _hint_2, …`.

## Open items for whoever finishes this

- **`< 10000` guarantee.** Post-trace numbering starts at 1 per graph and, in
  practice, stays far below `_SPAN_OVERFLOW_HINT_ID`. There is no explicit cap;
  if a pathological graph could exceed it, add an assertion or fold the span
  reservation into the numbering.

- **Duplicate-kwargs sibling scopes.** Two scopes with byte-identical `kwargs`
  produce the same content key. Within a single nested chain the kwargs differ
  in practice, but sibling/repeated scopes with identical kwargs would merge.
  If that becomes a real pattern, fold the enclosing chain into the key.

- **Coverage of the id contract.** There is no direct unit test asserting the
  full id contract (uniqueness, `< 10000`, outermost-first, stability across
  recompiles) in isolation; today it is covered indirectly by
  `test_coarse_tile_e2e.py` and the flash tests. A focused test would harden it.

## How it was validated

On real hardware:

- A 2-call probe (same graph, identical inputs): first call compiles
  (`sdsc` fires), second call is a cache hit — no `Recompiling` line, no `sdsc`.
- `test-spyre-scripts/test_flash.py` prints `SUCCESS` (numerics match CPU).
- `test_coarse_tile_e2e.py`, `test_propagate_named_dims.py`,
  `test_work_division_hint.py` all pass, including the run order that
  previously regressed (a softmax-shaped graph compiled before flash).
