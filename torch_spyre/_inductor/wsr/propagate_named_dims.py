# Copyright 2025 The Torch-Spyre Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import dataclasses
import logging
import sympy
import torch
from ..logging_utils import get_inductor_logger
from torch._inductor.ir import (
    ComputedBuffer,
    FixedLayout,
    InputBuffer,
    MutationLayoutSHOULDREMOVE,
    Operation,
    Pointwise,
    Reduction,
    StorageBox,
    TensorBox,
)
from torch._inductor.dependencies import MemoryDep, is_indirect
from torch._inductor.graph import GraphLowering
from torch._inductor.virtualized import V
from ..errors import Unsupported
from ..pass_utils import (
    host_coordinates,
    device_coordinates,
    indirect_sizes_from_op,
    op_out_coords,
    find_reduction_var,
)
from ..ir import SpyreConstantFallback
from ..propagate_hints import DimHint, get_op_hints
from torch_spyre._C import SpyreTensorLayout
from torch.utils.weak import WeakTensorKeyDictionary

logger = get_inductor_logger("propagate_named_dims")
hints_logger = get_inductor_logger("assign_dim_hints")


# Used for propagation of named dims if this pass runs.
# This pass does not run unless the driver program called name_tensor_dims.
_named_dims: dict[str, int] = {}
_named_tensor_dims = WeakTensorKeyDictionary()
_enabled = False


def reset():
    global _enabled
    _named_dims.clear()
    _named_tensor_dims.clear()
    _enabled = False



def name_tensor_dims(tensor: torch.Tensor, named_dims: list[str]) -> torch.Tensor:
    """Annotate a tensor with its named dimensions: [name, ...]"""
    global _enabled
    _enabled = True
    _named_tensor_dims[tensor] = named_dims
    return tensor


def _get_buffer(dep):
    return V.graph.get_buffer(dep.name)


def _get_layout(dep) -> "FixedLayout | None":
    buf = _get_buffer(dep)
    if buf is not None and hasattr(buf, "get_layout"):
        return buf.get_layout()
    tb = V.graph.graph_inputs.get(dep.name)
    if (
        isinstance(tb, TensorBox)
        and isinstance(tb.data, StorageBox)
        and isinstance(tb.data.data, InputBuffer)
    ):
        return tb.data.data.layout
    return None


def _get_dim_prop_info(dep):
    buf = _get_buffer(dep)
    if buf is not None:
        dpi = getattr(buf, "_dim_prop_info", None)
        if dpi is not None:
            return dpi
    tb = V.graph.graph_inputs.get(dep.name)
    return getattr(tb, "_dim_prop_info", None) if tb is not None else None


def _lone_sym(coord: sympy.Expr) -> sympy.Symbol | None:
    syms = coord.free_symbols
    return next(iter(syms)) if len(syms) == 1 else None


def _untracked_name(context: str, sym, size: int) -> str:
    name = f"_untracked_{size}"
    _named_dims.setdefault(name, size)
    logger.warning(
        f"{context}: loop var {sym} has no named dim mapping -- using {name}"
    )
    return name


def _assign_name(name: str, size: int) -> None:
    """Register name->size in _named_dims (first write wins)."""
    _named_dims.setdefault(name, size)


def compute_input_named_dims(
    dep: MemoryDep,
    op=None,
    ind_sizes=None,
    view_split: dict[str, list[str]] | None = None,
) -> dict:
    """Map loop vars to named dim names for a single input dep."""
    dpi = _get_dim_prop_info(dep)
    buf_named_dims = dpi.named_dims if dpi is not None else None
    if not buf_named_dims:
        if not dep.index.free_symbols:
            return {}
        context = f"{op.get_name()}/{dep.name}" if op is not None else dep.name
        return {
            sym: [_untracked_name(context, sym, int(size))]
            for sym, size in dep.ranges.items()
        }
    layout = _get_layout(dep)
    if layout is None:
        return {}
    coords = host_coordinates(layout, dep, ind_sizes)
    remaining = list(buf_named_dims)
    result: dict[sympy.Symbol, list[str]] = {}
    for i, coord in enumerate(coords):
        if not remaining:
            break
        dim_size = int(layout.size[i])
        if dim_size == 1:
            # Skip: size-1 dims are not annotated.  Broadcast dims (e.g. a [1,N]
            # buffer annotated ["M","N"]) silently become _untracked_ — we cannot
            # raise here without breaking legitimate unannotated size-1 dims.
            # See test_broadcast_expand_*
            continue
        name = remaining.pop(0)
        _assign_name(name, dim_size)
        # Loop vars for this layout dim: symbols in the coord that are also
        # loop variables of this dep (dep.ranges.keys()), sorted by coefficient
        # descending so the outermost (largest-stride) var comes first.
        loop_vars = sorted(
            coord.free_symbols & dep.ranges.keys(),
            key=lambda s: int(abs(coord.coeff(s))),
            reverse=True,
        )
        if len(loop_vars) == 1:
            result.setdefault(loop_vars[0], []).append(name)
        elif len(loop_vars) == 0:
            # This layout dim is index-selected by a gather/scatter index
            # symbol (e.g. `tmp0`).  Raise for anything else — a constant
            # or unexpected coord should not be silently skipped.
            sym = _lone_sym(coord)
            if sym is not None and is_indirect(sym.name):
                continue
            raise Unsupported(
                f"{dep.name}: layout dim {i} (size {dim_size}) has no loop vars "
                f"and no indirect index symbol in coord {coord!r} for name {name}"
            )
        elif len(loop_vars) > 1:
            # More loop vars than one name: a reshape split this layout dim.
            if name.startswith("_untracked_"):
                # No meaningful name to preserve — assign fresh untracked names
                # per loop var (e.g. a k/v projection output reshaped into heads).
                for loop_var in loop_vars:
                    size = int(dep.ranges[loop_var])
                    result.setdefault(loop_var, []).append(
                        _untracked_name(dep.name, loop_var, size)
                    )
                continue
            split_names = (view_split or {}).get(name)
            if split_names is None:
                raise Unsupported(
                    f"{dep.name}: layout dim {i} has {len(loop_vars)} loop vars for "
                    f"name '{name}' -- use spyre_hint(view_split={{'{name}': [...]}})"
                )
            if len(split_names) != len(loop_vars):
                raise Unsupported(
                    f"{dep.name}: view_split '{name}' -> {split_names} has "
                    f"{len(split_names)} names but {len(loop_vars)} loop vars"
                )
            for loop_var, split_name in zip(loop_vars, split_names):
                size = int(dep.ranges[loop_var])
                _assign_name(split_name, size)
                result.setdefault(loop_var, []).append(split_name)
    return result


def named_dims_for_sym(op: ComputedBuffer, sym: sympy.Symbol) -> list[tuple[str, int]]:
    """Return [(name, size), ...] for the named dims covered by a loop variable."""
    dp = getattr(op, "_dim_prop_info", None)
    names = dp.loop_var_dims.get(sym, []) if dp is not None else []
    return [(n, _named_dims[n]) for n in names if n in _named_dims]


def named_dims_for_coord(
    op: ComputedBuffer, coord: sympy.Expr
) -> list[tuple[str, int]] | None:
    """Return [(name, size), ...] for the named dims covered by a host coord expression."""
    sym = _lone_sym(coord)
    if sym is None:
        return None
    return named_dims_for_sym(op, sym)


def get_input_named_dims(inputs: list, op=None) -> dict:
    """
    Merge named dim mappings from all inputs into a single loop-var → names dict.
    Real names win over _untracked_ placeholders when both inputs cover the same sym.
    """
    loop_var_dims: dict[sympy.Symbol, list[str]] = {}
    ind_sizes = indirect_sizes_from_op(op)
    view_split = getattr(op, "_view_split_hint", {}) if op is not None else {}
    for inp in inputs:
        new = compute_input_named_dims(inp, op, ind_sizes=ind_sizes, view_split=view_split)
        for sym, names in new.items():
            if sym not in loop_var_dims or all(
                n.startswith("_untracked_") for n in loop_var_dims[sym]
            ):
                loop_var_dims[sym] = names
    return loop_var_dims


@dataclasses.dataclass
class _DimPropInfo:
    named_dims: list = dataclasses.field(default_factory=list)
    reduction_named_dims: list | None = None
    loop_var_dims: dict = dataclasses.field(default_factory=dict)


def _set_no_named_dims(op):
    op._dim_prop_info = _DimPropInfo()  # type: ignore[attr-defined]


def _compute_named_dims(op, inputs):
    loop_var_dims = get_input_named_dims(inputs, op)
    output_dep = next(iter(op.get_read_writes().writes))

    for sym in output_dep.ranges:
        if sym not in loop_var_dims:
            size = int(output_dep.ranges[sym])
            loop_var_dims[sym] = [_untracked_name(op.get_name(), sym, size)]
    out_coords = op_out_coords(op)

    named_dims = []
    for coord in out_coords:
        sym = _lone_sym(coord)
        if sym is None:
            continue
        names = loop_var_dims.get(sym, [])
        # Skip size-1 loop vars that carry only untracked placeholder names.
        # These arise when B=1: i0 (range 1) is absent from every input's
        # loop_var_dims (compute_input_named_dims skips size-1 layout dims),
        # so the fallback above assigns _untracked_1. Including it in
        # named_dims would prepend an extra entry that confuses _consume_names
        # in downstream ops (it would consume "B" into the wrong pair).
        # Omitting it keeps named_dims aligned with the non-unit layout dims,
        # matching the contract of compute_input_named_dims (size-1 → skip).
        if int(output_dep.ranges.get(sym, 0)) == 1 and all(
            n.startswith("_untracked_") for n in names
        ):
            continue
        named_dims.extend(names)
    reduction_named_dims = None
    if isinstance(op.data, Reduction):
        reduction_sym = find_reduction_var(inputs[0], output_dep)
        if reduction_sym not in loop_var_dims:
            size = int(inputs[0].ranges[reduction_sym])
            loop_var_dims[reduction_sym] = [
                _untracked_name(op.get_name(), reduction_sym, size)
            ]
        reduction_named_dims = loop_var_dims[reduction_sym]
    op._dim_prop_info = _DimPropInfo(  # type: ignore[attr-defined]
        named_dims=named_dims,
        loop_var_dims=loop_var_dims,
        reduction_named_dims=reduction_named_dims,
    )


def _log_dep_debug(label: str, dep: MemoryDep, ind_sizes=None) -> None:
    buf = _get_buffer(dep)
    layout = (
        buf.get_layout() if buf is not None and hasattr(buf, "get_layout") else None
    )
    dpi = _get_dim_prop_info(dep)
    named_dims = dpi.named_dims if dpi is not None else []
    logger.debug(f"  {label} {dep.name}: named_dims={named_dims}")
    if layout is not None:
        logger.debug(
            f"    host_size={list(layout.size)}  host_stride={list(layout.stride)}"
        )
        logger.debug(f"    host_coordinates={host_coordinates(layout, dep, ind_sizes)}")
    stl = getattr(buf, "layout", None) if buf is not None else None
    if isinstance(stl, SpyreTensorLayout):
        logger.debug(f"    device_size={stl.device_size}  stride_map={stl.stride_map}")
        logger.debug(
            f"    device_coordinates={device_coordinates(stl, dep, ind_sizes)}"
        )
    logger.debug(f"    index={dep.index}  ranges={dict(dep.ranges)}")


def _log_op_inputs(op: ComputedBuffer) -> None:
    for dep in op.get_read_writes().reads:
        if isinstance(dep, MemoryDep):
            dpi = _get_dim_prop_info(dep)
            named_dims = dpi.named_dims if dpi is not None else "?"
            buf = _get_buffer(dep)
            host_size = (
                list(buf.get_layout().size)
                if buf is not None and hasattr(buf, "get_layout")
                else "?"
            )
            logger.info(
                f"    input {dep.name}: named_dims={named_dims}  host_size={host_size}"
                f"  index={dep.index}  ranges={dict(dep.ranges)}"
            )


def _log_op(op: Operation) -> None:
    origins: set = getattr(getattr(op, "data", op), "origins", set())
    aten_ops = [str(n.target) for n in origins if hasattr(n, "target")]
    dp = getattr(op, "_dim_prop_info", None)
    if dp is None or not dp.loop_var_dims:
        logger.info(
            f"  {op.get_operation_name()}: skipped"
            f" ({type(op).__name__} / {type(getattr(op, 'data', op)).__name__})"
            f"  aten={aten_ops}"
        )
        if isinstance(op, ComputedBuffer):
            _log_op_inputs(op)
            logger.info(
                f"    output: ({op.get_name()}) named_dims={dp.named_dims if dp else []}"
            )
        return
    is_reduction = isinstance(op.data, Reduction)
    reduction_type = getattr(op.data, "reduction_type", None)
    logger.info(
        f"  {op.get_operation_name()}"
        f" ({'reduction' if is_reduction else 'pointwise'})"
        f"  aten={aten_ops}  reduction_type={reduction_type}"
    )
    _log_op_inputs(op)
    logger.info("    loop vars:")
    rw = op.get_read_writes()
    ranges = {}
    for dep in list(rw.reads) + list(rw.writes):
        if isinstance(dep, MemoryDep):
            ranges.update({str(s): int(v) for s, v in dep.ranges.items()})
    for sym, names in dp.loop_var_dims.items():
        sym_range: int | str = ranges.get(str(sym), "?")
        declared = [f"{n}={_named_dims[n] if n in _named_dims else '?'}" for n in names]
        logger.info(
            f"      {sym}: range={sym_range}  named_dim(s)={names}  declared={declared}"
        )
    if is_reduction:
        logger.info(f"    reduction over: {dp.reduction_named_dims}")
    logger.info(f"    output: ({op.get_name()}) named_dims={dp.named_dims}")
    logger.info("")


def _propagate_named_dims_impl(graph: GraphLowering) -> None:
    operations = graph.operations
    if graph.graph_input_names:
        for name, real_input in zip(graph.graph_input_names, V.get_real_inputs()):
            if isinstance(real_input, torch.Tensor):
                tb = graph.graph_inputs[name]
                if (
                    not isinstance(tb, TensorBox)
                    or not isinstance(tb.data, StorageBox)
                    or not isinstance(tb.data.data, InputBuffer)
                ):
                    raise Unsupported(
                        f"graph input {name} is not a TensorBox(StorageBox(InputBuffer))"
                    )
                layout = tb.data.data.layout
                if not isinstance(layout, FixedLayout):
                    raise Unsupported(f"graph input {name} does not have a FixedLayout")
                tb._dim_prop_info = _DimPropInfo(  # type: ignore[attr-defined]
                    named_dims=_named_tensor_dims.get(real_input) or []
                )

    for op in operations:
        if op.is_no_op():
            _set_no_named_dims(op)
        elif isinstance(op, ComputedBuffer):
            if isinstance(op.layout, MutationLayoutSHOULDREMOVE):
                continue
            hint = False
            view_split_hint: dict[str, list[str]] = {}
            for hint_dict in get_op_hints(op).values():
                if "named_dims" in hint_dict:
                    hint = True
                    named_dims = hint_dict["named_dims"]
                if "view_split" in hint_dict:
                    view_split_hint.update(hint_dict["view_split"])
            op._view_split_hint = view_split_hint  # type: ignore[attr-defined]
            if hint:
                coords = op_out_coords(op)
                layout_size = op.get_layout().size
                # zip() below truncates to the shorter of named_dims/layout_size,
                # so a name-count mismatch would silently drop names (leaving them
                # unregistered) rather than fail loudly like the input path.  Warn
                # so a bad in-graph annotation is visible instead of a no-op.
                if len(named_dims) != len(layout_size):
                    logger.warning(
                        f"{op.get_operation_name()}: named_dims hint has "
                        f"{len(named_dims)} name(s) {named_dims} but output layout "
                        f"has {len(layout_size)} dim(s) {list(layout_size)}; "
                        f"extra entries are ignored"
                    )
                loop_var_dims: dict[sympy.Symbol, list[str]] = {}
                for i, (coord, dim_name) in enumerate(zip(coords, named_dims)):
                    # Register the size for every name (including size-1 dims) so
                    # downstream consumers resolve it: named_dims_for_sym filters
                    # on `name in _named_dims`.  setdefault means the first op
                    # to see a name wins.
                    _named_dims.setdefault(dim_name, int(layout_size[i]))
                    # A size-1 dim yields coord == 0 (sym is None): the name stays
                    # in named_dims for positional alignment and is declared
                    # above, but it has no loop var to tile (it is optimized
                    # away), so it is absent from loop_var_dims.
                    sym = _lone_sym(coord)
                    if sym is not None:
                        loop_var_dims[sym] = [dim_name]
                op._dim_prop_info = _DimPropInfo(  # type: ignore[attr-defined]
                    named_dims=named_dims,
                    loop_var_dims=loop_var_dims,
                )
                continue
            origins: set = getattr(op.data, "origins", set())
            aten_ops = [str(n.target) for n in origins if hasattr(n, "target")]
            reduction_type = getattr(op.data, "reduction_type", None)
            logger.debug(
                f"\n--- {op.get_operation_name()} ({type(op.data).__name__})"
                f" aten={aten_ops} reduction_type={reduction_type}"
            )
            rw = op.get_read_writes()
            inputs = [d for d in rw.reads if isinstance(d, MemoryDep)]
            if logger.isEnabledFor(logging.DEBUG):
                ind_sizes = indirect_sizes_from_op(op)
                for dep in inputs:
                    _log_dep_debug("input", dep, ind_sizes)
                for dep in rw.writes:
                    if isinstance(dep, MemoryDep):
                        _log_dep_debug("output", dep, ind_sizes)
            if isinstance(op.data, (Pointwise, Reduction)):
                _compute_named_dims(op, inputs)
            else:
                logger.warning(f"unhandled node type {type(op.data)}")
                _set_no_named_dims(op)
        elif isinstance(op, SpyreConstantFallback):
            _set_no_named_dims(op)
        else:
            logger.warning(f"unhandled operation type {type(op)}")
            _set_no_named_dims(op)

    if logger.isEnabledFor(logging.INFO):
        logger.info("DECLARED DIMS")
        for name, size in _named_dims.items():
            logger.info(f"  {name} = {size}")

        logger.info("INPUT TENSORS")
        for name in graph.graph_input_names:
            tb = graph.graph_inputs[name]
            if isinstance(tb, TensorBox):
                dp = getattr(tb, "_dim_prop_info", None)
                logger.info(f"  {name}: named_dims={dp.named_dims if dp else []}")

        logger.info("OPS")
        for op in operations:
            _log_op(op)


def _graph_has_named_dims_hint(graph: GraphLowering) -> bool:
    """True if any op carries a spyre_hint(named_dims=[...]) annotation.

    This lets a decomposition name its own intermediate dims entirely in-graph
    (e.g. the flash SDPA decomposition), without a driver-side name_tensor_dims()
    call.  Such a hint is the only in-graph way to name a matmul output, whose
    loop vars carry no names from inputs.

    This scans graph.operations, which _propagate_named_dims_impl then scans
    again; the extra pass is a cheap gate on a per-compilation path.  If it ever
    shows up as a hotspot, fold the check into _propagate_named_dims_impl (run
    unconditionally, short-circuit inside) so operations is walked once.
    """
    for op in graph.operations:
        if isinstance(op, ComputedBuffer):
            for hint_dict in get_op_hints(op).values():
                if "named_dims" in hint_dict:
                    return True
    return False


def propagate_named_dims(
    graph: GraphLowering,
) -> None:
    """Propagate named dims from annotated inputs through the op graph."""
    global _enabled
    # Run when a driver called name_tensor_dims() (_enabled) OR when the graph
    # itself carries an in-graph named_dims hint. Do not set _enabled here — the
    # finally block still resets it, and _graph_has_named_dims_hint re-derives
    # the in-graph case each run.
    if not (_enabled or _graph_has_named_dims_hint(graph)):
        return
    try:
        _propagate_named_dims_impl(graph)
    except BaseException:
        # On the normal path _named_dims MUST survive: assign_dim_hints runs
        # next and reads it (named_dims_for_sym filters on `name in
        # _named_dims`), then clears it in its own reset(). Only when
        # _propagate_named_dims_impl raises does the pipeline abort before
        # assign_dim_hints runs -- then the sizes self-registered by the
        # in-graph path (and any driver-declared dims) would leak into the next
        # compilation, so clear them here on the error path only.
        _named_dims.clear()
        raise
    finally:
        _named_tensor_dims.clear()
        _enabled = False


def _assign_dim_hints_impl(operations: list[Operation]) -> None:
    for op in operations:
        if not isinstance(op, ComputedBuffer):
            continue
        # Reconstructed buffers can copy optional metadata; recompute it here.
        if hasattr(op, "work_div_loop_info"):
            del op.work_div_loop_info  # type: ignore[attr-defined]
        dp = getattr(op, "_dim_prop_info", None)
        op_hints = get_op_hints(op) if dp and dp.loop_var_dims else {}
        if not op_hints:
            op.dim_hints = []  # type: ignore[attr-defined]
            if dp is not None:
                del op._dim_prop_info  # type: ignore[attr-defined]
            continue

        assert dp is not None  # guaranteed by op_hints check above
        if any(hint_dict.get("work_div") for hint_dict in op_hints.values()):
            op.work_div_loop_info = {  # type: ignore[attr-defined]
                sym: list(names) for sym, names in dp.loop_var_dims.items()
            }

        reduction_dims = set(dp.reduction_named_dims or [])

        coord_for_name: dict[str, sympy.Symbol] = {}
        for coord in op_out_coords(op):
            sym = _lone_sym(coord)
            if sym is None:
                continue
            for name, _ in named_dims_for_sym(op, sym):
                coord_for_name[name] = sym
        # Also map reduction dim names to their loop variable.  Reduction dims
        # don't appear in output coordinates, so they would never be found by
        # the output-coord loop above.  dp.loop_var_dims covers all loop vars
        # (including the reduction dim), so we invert it for reduction names.
        for sym, names in dp.loop_var_dims.items():
            for name in names:
                if name in reduction_dims:
                    coord_for_name[name] = sym

        dim_hints = []
        for hint_id, hint_dict in sorted(op_hints.items()):
            # A hint scope uses exactly one of tiles/slices/num_tiles_per_dim.
            dims: dict[str, int] = next(
                (
                    v
                    for k in ("tiles", "slices", "num_tiles_per_dim")
                    if (v := hint_dict.get(k))
                ),
                {},
            )
            # TODO: support multiple dimensions per spyre_hint() call.
            # hint_id_to_ranges_pos in plan_coarse_tile_groups would need to
            # become dict[int, list[int]] and _hints_levels would need to
            # deduplicate by hint_id.
            if len(dims) > 1:
                raise NotImplementedError(
                    f"spyre_hint() argument {list(hint_dict.items())} specifies "
                    f"{len(dims)} dimensions; only one is currently allowed per "
                    f"spyre_hint() call (not yet implemented)"
                )
            for name, count in dims.items():
                sym = coord_for_name.get(name)
                dim_hints.append(
                    DimHint(
                        dim_names=[name],
                        split_count=count,
                        loop_var=sym,
                        is_reduction=name in reduction_dims,
                        hint_id=hint_id,
                    )
                )
        op.dim_hints = dim_hints  # type: ignore[attr-defined]

        # Clean up temp intermediates — only dim_hints persists.
        del op._dim_prop_info  # type: ignore[attr-defined]

    if hints_logger.isEnabledFor(logging.INFO):
        ops = [
            op
            for op in operations
            if isinstance(op, ComputedBuffer) and getattr(op, "dim_hints", None)
        ]
        if ops:
            hints_logger.info("=== assign_dim_hints ===")
            for op in ops:
                rw = op.get_read_writes()
                all_ranges = {
                    s: int(v)
                    for dep in [*rw.reads, *rw.writes]
                    for s, v in dep.ranges.items()
                }
                hints_logger.info(f"{op.get_operation_name()}:")
                for h in op.dim_hints:
                    r = all_ranges.get(h.loop_var, 0) if h.loop_var else 0
                    per_tile = r // h.split_count if r else "?"
                    reduction_tag = "  [reduction]" if h.is_reduction else ""
                    hints_logger.info(
                        f"  {h.dim_names}  range={r}"
                        f"  split_count={h.split_count}  -> {per_tile} per tile"
                        f"  loop_var={h.loop_var}{reduction_tag}"
                    )


def assign_dim_hints(graph: GraphLowering) -> None:
    """Combine spyre_hint scope annotations with propagated named dimensions.

    Reads the hint scopes (from spyre_hint() context managers in user code,
    attached to FX nodes via meta["custom"]) and matches hinted dimension names
    against the named loop variables on each op.  The named loop variables come
    from propagate_named_dims(), which propagates name_tensor_dims() annotations
    through the op graph — that pass must run before this one.

    Produces op.dim_hints: a flat list of DimHint, one per hinted dimension,
    ordered outermost hint scope first.  Consumed by hints_to_coarse_tile_groups
    to form coarse tiling groups.

    Deletes op._dim_prop_info when done — those fields are only needed here.
    """
    try:
        _assign_dim_hints_impl(graph.operations)
    finally:
        reset()
