"""CPU oracles for the routed-expert path, so the MoE changes are covered without a device.

Why this file exists
--------------------
The MoE work of the last rounds — the sort-once expert loop, the fused SwiGLU, the ``moe_strict``
bounds-check caching — was verified **on the device** against ``reference_moe``.  But every existing
MoE test module imports ``tilelang`` at collection time, so on a CPU box the whole area is
*uncollected*: a future edit to ``_routed_sorted`` could break it and nothing local would notice.

``fused_experts`` needs only torch (the ``flags``/``torch_npu`` imports inside it are lazy), and
``testing.moe_reference.reference_moe`` is a straight-line fp32 oracle, so this checks the three
paths against each other on the CPU:

* the baseline per-expert loop,
* the sort-once loop (default on),
* the oracle.

Both switches are exercised, because a test that only covers the new path cannot tell "the new path
is right" from "both paths changed together".
"""

from __future__ import annotations

import pytest
import torch

from tileinfer.testing.moe_reference import reference_moe
from tileinfer.vllm.ops import moe


@pytest.fixture(autouse=True)
def _clean_moe_state(monkeypatch):
    """Order independence: the module keeps a process-wide validated-bounds set and reads switches
    per call, so a preceding test can leak into this file (observed: this file passes alone but one
    case fails when run after the reference/attention suites)."""
    moe._VALIDATED_ID_BOUNDS.clear()
    monkeypatch.delenv("TILEINFER_FLAG_WATCH", raising=False)
    monkeypatch.delenv("TILEINFER_MOE_STRICT", raising=False)
    monkeypatch.delenv("TILEINFER_MOE_SORTED", raising=False)
    monkeypatch.delenv("TILEINFER_MOE_FP32", raising=False)
    monkeypatch.delenv("TILEINFER_MOE_SWIGLU", raising=False)
    yield
    moe._VALIDATED_ID_BOUNDS.clear()


def _case(tokens=6, hidden=8, inter=4, experts=4, top_k=2, seed=0):
    torch.manual_seed(seed)
    w13 = torch.randn(experts, 2 * inter, hidden) * 0.3
    w2 = torch.randn(experts, hidden, inter) * 0.3
    x = torch.randn(tokens, hidden) * 0.2
    ids = torch.stack([torch.randperm(experts)[:top_k] for _ in range(tokens)]).to(torch.int32)
    w = torch.rand(tokens, top_k)
    w = w / w.sum(dim=1, keepdim=True)
    return x, w13, w2, ids, w


def _run(x, w13, w2, ids, w, **kwargs):
    return moe.fused_experts(x, w13, w2, w, ids, activation="silu", **kwargs)


def test_sorted_and_loop_agree_with_the_oracle(monkeypatch):
    x, w13, w2, ids, w = _case()
    want = reference_moe(x, w13, w2, ids, w)

    monkeypatch.setenv("TILEINFER_MOE_SORTED", "0")
    monkeypatch.delenv("TILEINFER_FLAG_WATCH", raising=False)
    loop = _run(x, w13, w2, ids, w)

    monkeypatch.setenv("TILEINFER_MOE_SORTED", "1")
    monkeypatch.setenv("TILEINFER_MOE_BMM", "0")
    sorted_path = _run(x, w13, w2, ids, w)

    monkeypatch.setenv("TILEINFER_MOE_BMM", "1")
    batched = _run(x, w13, w2, ids, w)

    assert torch.allclose(loop, want, atol=1e-5), "the baseline loop drifted from the oracle"
    assert torch.allclose(sorted_path, want, atol=1e-5), "the sort-once loop drifted from the oracle"
    assert torch.allclose(batched, want, atol=1e-5), "the batched (bmm) path drifted from the oracle"
    assert torch.equal(loop, sorted_path), (
        "loop and sort-once run the *same* per-expert matmul calls, so they must agree bit for bit"
    )
    # The batched path uses one ``bmm`` for all experts instead of one matmul each; different BLAS
    # kernels are allowed to differ in the last fp32 bit, so this is a tolerance, not equality.
    assert torch.allclose(batched, sorted_path, atol=1e-6), (
        "batching must not change the result beyond fp32 re-association"
    )


def test_duplicate_expert_ids_accumulate(monkeypatch):
    """Two slots on the same expert must add, not overwrite."""
    monkeypatch.delenv("TILEINFER_FLAG_WATCH", raising=False)
    monkeypatch.setenv("TILEINFER_MOE_SORTED", "1")
    x, w13, w2, _ids, w = _case(tokens=2, top_k=2)
    ids = torch.tensor([[1, 1], [2, 2]], dtype=torch.int32)  # every slot duplicates its expert
    want = reference_moe(x, w13, w2, ids, w)
    got = _run(x, w13, w2, ids, w)
    assert torch.allclose(got, want, atol=1e-5)


def test_dropped_slots_contribute_nothing(monkeypatch):
    """A negative id means "no expert" and must be skipped by both paths.

    **The device path currently fails this and is switched off by default** (§77: it leaves a live
    slot unwritten, so ``torch.empty`` reads through as ``3.7e+19``).  The assertion below is
    therefore also the specification the device path has to meet before
    ``TILEINFER_MOE_DEVICE_PATH`` may be enabled.
    """
    monkeypatch.delenv("TILEINFER_FLAG_WATCH", raising=False)
    x, w13, w2, ids, w = _case(tokens=2, top_k=2)
    ids[0, 1] = -1
    ids[1, 0] = -1
    outs = {}
    for value in ("0", "1"):
        monkeypatch.setenv("TILEINFER_MOE_SORTED", value)
        outs[value] = _run(x, w13, w2, ids, w)
    assert torch.allclose(outs["0"], outs["1"], atol=1e-6), (
        "the sorted path parks dropped slots in a synthetic bucket; it must agree with the loop"
    )
    # And a dropped slot must behave like the same slot routed to an expert with router weight 0.
    # Phrased without ``reference_moe`` so the case cannot inherit state from the oracle -- and this
    # assertion is what exposed the uninitialised-buffer bug in ``_routed_sorted_into`` (§46.2).
    x, w13, w2, _ids, w = _case(tokens=2, top_k=2)
    zero_w = w.clone()
    zero_w[1, 0] = 0.0
    dropped = torch.tensor([[1, 1], [-1, 2]], dtype=torch.int32)
    routed_to_zero = torch.tensor([[1, 1], [1, 2]], dtype=torch.int32)
    outs = {}
    for value in ("0", "1"):
        monkeypatch.setenv("TILEINFER_MOE_SORTED", value)
        outs[value] = (
            _run(x, w13, w2, dropped, zero_w),
            _run(x, w13, w2, routed_to_zero, zero_w),
        )
    for value in ("0", "1"):
        assert torch.allclose(outs[value][0], outs[value][1], atol=1e-6), (
            f"sorted={value}: a dropped slot must behave like a zero-weight slot"
        )


def test_apply_router_weight_on_input_uses_the_loop(monkeypatch):
    """The sorted path folds the weight into the single combine, so it must not take over here."""
    monkeypatch.delenv("TILEINFER_FLAG_WATCH", raising=False)
    x, w13, w2, ids, w = _case(tokens=4, top_k=1)
    outs = {}
    for value in ("0", "1"):
        monkeypatch.setenv("TILEINFER_MOE_SORTED", value)
        outs[value] = _run(x, w13, w2, ids, w, apply_router_weight_on_input=True)
    assert torch.allclose(outs["0"], outs["1"], atol=1e-6)


def test_strict_switch_does_not_change_the_numbers(monkeypatch):
    monkeypatch.delenv("TILEINFER_FLAG_WATCH", raising=False)
    x, w13, w2, ids, w = _case()
    outs = {}
    for value in ("1", "0"):
        monkeypatch.setenv("TILEINFER_MOE_STRICT", value)
        moe._VALIDATED_ID_BOUNDS.clear()
        outs[value] = _run(x, w13, w2, ids, w)
    assert torch.equal(outs["1"], outs["0"]), (
        "caching the bounds check must not change a single value"
    )


def test_out_of_range_ids_are_rejected_in_strict_mode(monkeypatch):
    monkeypatch.delenv("TILEINFER_FLAG_WATCH", raising=False)
    monkeypatch.setenv("TILEINFER_MOE_STRICT", "1")
    moe._VALIDATED_ID_BOUNDS.clear()
    x, w13, w2, ids, w = _case(tokens=2, experts=4)
    ids[0, 0] = 9
    with pytest.raises(ValueError, match="topk_ids contains expert id"):
        _run(x, w13, w2, ids, w)


def test_silu_and_mul_gate_first_layout():
    """SwiGLU over ``[gate | up]``: the product uses the first half as the gate."""
    g = torch.tensor([[1.0, -1.0]])
    u = torch.tensor([[2.0, 3.0]])
    got = moe.silu_and_mul(torch.cat([g, u], dim=-1))
    want = torch.nn.functional.silu(g) * u
    assert torch.allclose(got, want, atol=1e-6)
