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

# Tests for the plan_restickify analysis pass.
#
# Run just these tests with:
#   pytest tests/inductor/test_plan_restickify.py -m restickify_plan

from math import prod

import pytest
import torch

import torch_spyre  # noqa: F401 — registers the spyre backend
import torch_spyre._inductor.plan_restickify as _pr
import torch_spyre._inductor.stickify as _st
from utils_inductor import _compile_and_run

DEVICE = torch.device("spyre")
S = 128  # must be a multiple of 64


@pytest.fixture(autouse=True)
def enable_plan_capture(monkeypatch):
    monkeypatch.setenv("SPYRE_CAPTURE_RESTICKIFY_PLAN", "1")
    import torch._inductor.config as inductor_config
    monkeypatch.setattr(inductor_config, "force_disable_caches", True)
    yield
    torch._dynamo.reset_code_caches()


def _run(fn, *args):
    _compile_and_run(fn, args, DEVICE)


def _verify(expected_cost):
    """Assert planned cost and actual restickify elements both equal expected_cost."""
    assert _pr.last_frontier, "plan_restickify did not record a frontier"
    planned = _pr.last_frontier[0][2]
    actual = sum(
        prod(int(s) for s in entry["target_layout"].size)
        for entries in _st.last_restickify_plan.values()
        for entry in entries
    )
    assert planned == expected_cost, f"planned cost: expected {expected_cost}, got {planned}"
    assert actual == expected_cost, f"actual elements: expected {expected_cost}, got {actual}"


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

# -- one restickify: single transpose among otherwise-uniform args -----------


def test_plan_at_b_c_d():
    """a.t() + b + c + d — one restickify of a.t()."""
    a, b, c, d = [torch.randn((S, S), dtype=torch.float16) for _ in range(4)]
    _run(lambda a, b, c, d: a.t() + b + c + d, a, b, c, d)
    _verify(S * S)


def test_plan_a_bt_c_d():
    """a + b.t() + c + d — one restickify of b.t()."""
    a, b, c, d = [torch.randn((S, S), dtype=torch.float16) for _ in range(4)]
    _run(lambda a, b, c, d: a + b.t() + c + d, a, b, c, d)
    _verify(S * S)


def test_plan_a_b_ct_d():
    """a + b + c.t() + d — one restickify of c.t()."""
    a, b, c, d = [torch.randn((S, S), dtype=torch.float16) for _ in range(4)]
    _run(lambda a, b, c, d: a + b + c.t() + d, a, b, c, d)
    _verify(S * S)


def test_plan_a_b_c_dt():
    """a + b + c + d.t() — one restickify of d.t()."""
    a, b, c, d = [torch.randn((S, S), dtype=torch.float16) for _ in range(4)]
    _run(lambda a, b, c, d: a + b + c + d.t(), a, b, c, d)
    _verify(S * S)


# -- one restickify: single non-transpose among otherwise-transposed args ----


def test_plan_a_bt_ct_dt():
    """a + b.t() + c.t() + d.t() — one restickify of a."""
    a, b, c, d = [torch.randn((S, S), dtype=torch.float16) for _ in range(4)]
    _run(lambda a, b, c, d: a + b.t() + c.t() + d.t(), a, b, c, d)
    _verify(S * S)


def test_plan_at_b_ct_dt():
    """a.t() + b + c.t() + d.t() — one restickify of b."""
    a, b, c, d = [torch.randn((S, S), dtype=torch.float16) for _ in range(4)]
    _run(lambda a, b, c, d: a.t() + b + c.t() + d.t(), a, b, c, d)
    _verify(S * S)


def test_plan_at_bt_c_dt():
    """a.t() + b.t() + c + d.t() — one restickify of c."""
    a, b, c, d = [torch.randn((S, S), dtype=torch.float16) for _ in range(4)]
    _run(lambda a, b, c, d: a.t() + b.t() + c + d.t(), a, b, c, d)
    _verify(S * S)


def test_plan_at_bt_ct_d():
    """a.t() + b.t() + c.t() + d — one restickify of d."""
    a, b, c, d = [torch.randn((S, S), dtype=torch.float16) for _ in range(4)]
    _run(lambda a, b, c, d: a.t() + b.t() + c.t() + d, a, b, c, d)
    _verify(S * S)


# -- one restickify: conflict in one subexpression, rest uniform -------------


def test_plan_parens_one_conflict():
    """((a + b) + (c.t() + d)) + (e + f) — conflict only in inner group."""
    a, b, c, d, e, f = [torch.randn((S, S), dtype=torch.float16) for _ in range(6)]
    _run(lambda a, b, c, d, e, f: ((a + b) + (c.t() + d)) + (e + f), a, b, c, d, e, f)
    _verify(S * S)


# -- matmul forced constraints -----------------------------------------------


# def test_plan_matmul_no_cost():
#     """a @ b — both inputs satisfy matmul stick constraints, cost = 0."""
#     a, b = [torch.randn((S, S), dtype=torch.float16) for _ in range(2)]
#     _run(lambda a, b: a @ b, a, b)
#     _verify(0)


# def test_plan_matmul_x_wrong_stick():
#     """a.t() @ b — x input needs restickify to move stick to reduction dim."""
#     a, b = [torch.randn((S, S), dtype=torch.float16) for _ in range(2)]
#     _run(lambda a, b: a.t() @ b, a, b)
#     _verify(S * S)


# def test_plan_matmul_y_wrong_stick():
#     """a @ b.t() — y input needs restickify to move stick to generated dim."""
#     a, b = [torch.randn((S, S), dtype=torch.float16) for _ in range(2)]
#     _run(lambda a, b: a @ b.t(), a, b)
#     _verify(S * S)


# def test_plan_adds_then_matmul_x():
#     """(a + b.t() + c.t() + d.t()) @ e.
#     Upstream optimal: stick=d0, 1 restickify of a (cost S*S).
#     Matmul x always needs forced restickify (reduction var is a fresh kernel var).
#     Total = 2*S*S.
#     """
#     a, b, c, d, e = [torch.randn((S, S), dtype=torch.float16) for _ in range(5)]
#     _run(lambda a, b, c, d, e: (a + b.t() + c.t() + d.t()) @ e, a, b, c, d, e)
#     _verify(2 * S * S)


# def test_plan_adds_then_matmul_y():
#     """a @ (b + c.t()) — y input has upstream conflict.
#     Beam finds upstream stick=d1 (restickify b.t(), cost S*S) so y satisfies
#     matmul constraint with no extra cost. Total = S*S.
#     With K=1 greedy may pick d0 upstream, paying extra S*S at the matmul.
#     """
#     a, b, c = [torch.randn((S, S), dtype=torch.float16) for _ in range(3)]
#     _run(lambda a, b, c: a @ (b + c.t()), a, b, c)
#     _verify(S * S)


# def test_plan_adds_then_matmul_y_long_chain():
#     """a @ (b + c.t() + d.t() + e.t()) — majority transposed (d0) going into y.
#     Upstream optimal: stick=d0, restickify b (cost S*S).
#     Matmul y needs d1 → forced extra S*S. Total = 2*S*S.
#     """
#     a, b, c, d, e = [torch.randn((S, S), dtype=torch.float16) for _ in range(5)]
#     _run(lambda a, b, c, d, e: a @ (b + c.t() + d.t() + e.t()), a, b, c, d, e)
#     _verify(2 * S * S)


# def test_plan_matmul_x_and_y_conflict():
#     """a.t() @ (b + c.t()) — x wrong stick (S*S) + y upstream conflict.
#     Beam picks d1 for y (no extra matmul cost). Total = 2*S*S.
#     """
#     a, b, c = [torch.randn((S, S), dtype=torch.float16) for _ in range(3)]
#     _run(lambda a, b, c: a.t() @ (b + c.t()), a, b, c)
#     _verify(2 * S * S)


# def test_plan_matmul_then_adds():
#     """(a @ b) + c.t() — matmul output has stick=generated_var.
#     Downstream pointwise sees conflict with c.t(). Beam picks the matmul
#     output's stick, so c.t() is restickified. Total = S*S.
#     """
#     a, b, c = [torch.randn((S, S), dtype=torch.float16) for _ in range(3)]
#     _run(lambda a, b, c: (a @ b) + c.t(), a, b, c)
#     _verify(S * S)


# def test_plan_matmul_then_long_adds():
#     """(a @ b) + c.t() + d.t() — matmul output (d1) vs two transposed inputs (d0).
#     Optimal: keep d1, restickify c.t() once. Total = S*S.
#     """
#     a, b, c, d = [torch.randn((S, S), dtype=torch.float16) for _ in range(4)]
#     _run(lambda a, b, c, d: (a @ b) + c.t() + d.t(), a, b, c, d)
#     _verify(S * S)


# def test_plan_chained_matmuls():
#     """(a @ b) @ c — second matmul x is a matmul output (row-major, stick on
#     generated_var/cols).  When accessed as x in the second matmul the reduction
#     dimension maps to the column direction, so stickify injects no restickify.
#     Both the plan cost and actual restickify count are 0.
#     """
#     a, b, c = [torch.randn((S, S), dtype=torch.float16) for _ in range(3)]
#     _run(lambda a, b, c: (a @ b) @ c, a, b, c)
#     _verify(0)


# -- multiple restickifies ---------------------------------------------------


def test_plan_two_independent_conflicts():
    """(a+b.t()) + (e.t()+f.t()+g) — 2 restickifies from two separate conflicts.

    First add: a(d1) vs b.t()(d0) — 1 restickify.
    Second add chain: e.t()(d0)+f.t()(d0)+g(d1) — optimal stick=d0, 1 restickify of g.
    Final add sees both results; plan picks consistent stick across both groups.
    Total = 2*S*S.
    """
    a, b, e, f, g = [torch.randn((S, S), dtype=torch.float16) for _ in range(5)]
    _run(lambda a, b, e, f, g: (a + b.t()) + (e.t() + f.t() + g), a, b, e, f, g)
    _verify(2 * S * S)


# -- fan-out: intermediate buffer consumed by two downstream ops ---------------


def test_plan_fanout_intermediate():
    """buf = a + b.t(); (buf + c) + (buf + d.t()) — buf consumed twice.

    Both paths from buf introduce one restickify each (either b.t() or d.t.
    depending on chosen stick). Optimal cost = 2*S*S.
    Catches: last_use pruning bug — if buf were dropped from state after its
    first consumer, the second consumer would fall back to _stick_var and
    undercount the conflict.
    """
    a, b, c, d = [torch.randn((S, S), dtype=torch.float16) for _ in range(4)]

    def fn(a, b, c, d):
        buf = a + b.t()
        return buf + c + (buf + d.t())

    _run(fn, a, b, c, d)
    _verify(2 * S * S)


# -- diamond: same CSE'd buffer read twice in one add -------------------------


def test_plan_diamond():
    """buf = a + b.t(); buf + buf — same intermediate read twice.

    Both reads resolve to the same stick var, so no second conflict.
    Cost = S*S (one restickify for b.t() in the first op).
    Catches: candidate_vars deduplication — if the set comprehension were
    broken, two identical vars could appear as a false conflict.
    """
    a, b = [torch.randn((S, S), dtype=torch.float16) for _ in range(2)]

    def fn(a, b):
        buf = a + b.t()
        return buf + buf

    _run(fn, a, b)
    _verify(S * S)


# -- non-square matmul: cost formula uses layout.size, not dep.ranges ---------


# def test_plan_matmul_rect_x_wrong_stick():
#     """(64x128).t() @ (64x192) — x wrong stick, cost = 64*128 not 128*128.

#     Catches: forced_cost formula. If x_buf.get_layout().size were replaced by
#     prod(dep.ranges.values()) the result would include the reduction dim,
#     giving 128*64*64 instead of 128*64.
#     """
#     M, K, N = 64, 128, 192
#     a = torch.randn((M, K), dtype=torch.float16)
#     b = torch.randn((M, N), dtype=torch.float16)
#     _run(lambda a, b: a.t() @ b, a, b)
#     _verify(M * K)


# -- non-matmul reduction (sum) passthrough -----------------------------------


def test_plan_sum_passthrough():
    """(a + b.t()).sum(0) + c — reduction between two pointwise stages.

    The sum is a passthrough op; its output stick is recorded in state.
    The downstream add sees sum_out + c; if the sum's recorded stick is wrong
    or missing, the add would miscalculate the conflict.
    Cost = S*S (one restickify for b.t() in the upstream add).
    """
    a, b = [torch.randn((S, S), dtype=torch.float16) for _ in range(2)]
    c = torch.randn((S,), dtype=torch.float16)
    _run(lambda a, b, c: (a + b.t()).sum(0) + c, a, b, c)
    _verify(S * S)


# -- two matmuls with wrong inputs added together -----------------------------


# def test_plan_two_matmuls_wrong_inputs():
#     """(a.t() @ b) + (c @ d.t()) — each matmul has one wrong-stick input.

#     a.t() costs S*S for x; d.t() costs S*S for y. No upstream conflict.
#     The two matmul outputs share the same generated_var naming so the
#     downstream add sees no conflict. Total = 2*S*S.
#     """
#     a, b, c, d = [torch.randn((S, S), dtype=torch.float16) for _ in range(4)]
#     _run(lambda a, b, c, d: (a.t() @ b) + (c @ d.t()), a, b, c, d)
#     _verify(2 * S * S)


# -- both matmul inputs have upstream conflicts --------------------------------


# def test_plan_matmul_both_inputs_upstream_conflict():
#     """(a + b.t()) @ (c + d.t()) — both inputs have upstream stick conflicts.

#     Upstream op1 (x): conflict d0/d1, each costs S*S.  Beam picks d1 (row-major)
#     so the matmul x stick lands on the reduction dim — no extra matmul cost.
#     Upstream op2 (y): conflict d0/d1.  Beam picks d1 = generated_var — no extra
#     matmul y cost.
#     Total optimal = 2*S*S (one restickify per upstream pointwise op).
#     """
#     a, b, c, d = [torch.randn((S, S), dtype=torch.float16) for _ in range(4)]
#     _run(lambda a, b, c, d: (a + b.t()) @ (c + d.t()), a, b, c, d)
#     _verify(2 * S * S)
