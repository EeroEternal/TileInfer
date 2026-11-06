"""DeepSeek-V4.1 attention layer vs the official reference, on CPU (bit-exact contract).

Runs ``tools/dsv41/s2_layer_check.py``'s harness for one layer with synthetic weights in the
checkpoint's own dtypes and requires **max|diff| == 0** and identical top-k index lists.  That is a
strong statement: it pins the Compressor / Indexer / candidate-block / sliding-window / sparse-attn
semantics to the official ``inference/model.py`` implementation itself (its six tilelang kernels are
swapped for the torch transcription in ``tools/dsv41/ref_cpu.py``).

Skipped when the checkpoint is not present (this test needs the official ``inference/`` sources; it
never needs the 476 GB weights).
"""

from __future__ import annotations

import os
import sys

import pytest

MODEL_DIR = "/home/models/DeepSeek-V4.1-Flash"
INFERENCE = os.path.join(MODEL_DIR, "inference")

pytestmark = pytest.mark.skipif(
    not os.path.isdir(INFERENCE),
    reason=f"official inference sources not present at {INFERENCE}",
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def _run(layers, steps):
    from tileinfer.models.dsv41 import Dsv41Config
    from tools.dsv41.s2_layer_check import official_model, run_chain

    cfg = Dsv41Config.from_json(os.path.join(MODEL_DIR, "config.json"))
    official = official_model(MODEL_DIR)
    return run_chain(
        official, cfg, layers, steps, device="cpu", max_seq_len=max(sum(steps) + 8, 256), verbose=False
    )


@pytest.mark.parametrize("layer_id", [2, 20])
def test_layer_matches_the_reference_bit_for_bit(layer_id):
    """Layer 2 = ratio-2 kv/index source; layer 20 = ratio-1 candidate source."""
    results = _run([layer_id], [8, 1, 1])
    assert results, "no comparison rows"
    for row in results:
        assert row.max_abs == 0.0, (
            f"layer {row.layer_id} step {row.step}: max|diff|={row.max_abs:.3e} "
            f"(must be bit-exact on CPU)"
        )
        assert row.idx_match, f"layer {row.layer_id} step {row.step}: top-k index lists differ"


def test_consumers_reuse_the_published_stream():
    """Layers 3..5 read what layer 2 published: a wrong shared-stream wiring shows up here."""
    results = _run([2, 3, 4, 5], [8, 1])
    assert [(r.layer_id, r.step) for r in results] == [
        (2, 0), (3, 0), (4, 0), (5, 0), (2, 1), (3, 1), (4, 1), (5, 1)
    ]
    assert all(r.max_abs == 0.0 for r in results)
    assert all(r.idx_match for r in results)


def test_ring_buffer_wrap_matches():
    """A 200-token prefill crosses the 128-slot window ring: the rotation path must agree.

    Unlike the short-sequence cases above this one is *not* required to be bit-exact: across a
    200-step prefill the fp32 reductions accumulate in a slightly different order between the two
    implementations and one element can land 1 bf16 ulp away (measured 3.9e-3 at |x| ~ 2).  The
    contract here is the numeric bar plus the discrete index list.
    """
    results = _run([2], [200, 1])
    assert len(results) == 2
    for row in results:
        assert row.max_abs <= 1e-2, f"layer {row.layer_id} step {row.step}: {row.max_abs:.3e}"
        assert row.idx_match, f"layer {row.layer_id} step {row.step}: index lists differ"
