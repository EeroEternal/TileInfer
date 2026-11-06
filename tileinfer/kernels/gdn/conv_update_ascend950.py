"""Fused GDN decode conv update: 4-tap conv + bias + SiLU + state shift, one kernel.

Why: ``causal_conv1d_update`` costs **0.72 ms per layer** (x30 = 21.6 ms of a ~103 ms decode step,
§81) for ~5e4 MACs — it is seven small operators at this device's ~0.1 ms per-operator floor.  One
kernel removes the floor.

Shapes (Qwen3.6 linear attention, decode = one token per sequence):

* ``X``      ``[C]``          this step's token, packed ``[q|k|v|z]`` channels
* ``State``  ``[L, C, W-1]``  conv cache, **mutated in place**; line ``Idx[0]`` belongs to this request
* ``Weight`` ``[C, W]``       depthwise taps
* ``Bias``   ``[C]``          fp32
* ``Out``    ``[C]``          conv + bias + optional SiLU

Per channel ``c``::

    acc = Bias[c] + sum_j State[idx, c, j] * Weight[c, j]  +  X[c] * Weight[c, W-1]
    Out[c] = silu(acc) if act else acc
    State[idx, c, 0..W-3] = State[idx, c, 1..W-2];  State[idx, c, W-2] = X[c]

Each core owns a **contiguous** channel block: §82 measured that strided stores on the last axis are
what made the torch version expensive, and the same trap applies here.

Structure mirrors ``chunk_wu_ascend950.py``: a plain factory returns the ``T.prim_func`` and
``tilelang.compile`` wraps it — ``tilelang.jit`` cannot decorate the factory because it re-parses the
decorated source and loses the enclosing scope (measured: ``NameError: name 'C' is not defined``).

**Experimental and default-off** (``TILEINFER_GDN_CONV_KERNEL=1``): it must be exercised on the device
against ``tests/test_gdn_causal_conv1d.py`` before it may become a default (C2/E4).
"""

from __future__ import annotations

import torch

__all__ = [
    "build_conv_update_kernel",
    "gdn_conv_update_kernel",
    "gdn_conv_update_kernel_available",
    "NUM_CORES",
]

NUM_CORES = 28  # AI-vector core count the sibling GDN kernels target


def gdn_conv_update_kernel(
    channels: int,
    width: int,
    lines: int,
    dtype: str = "bfloat16",
    cores: int = NUM_CORES,
):
    """Build the ``prim_func`` for one exact shape (tilelang caches per spec, so shapes are explicit).

    SiLU is applied unconditionally: that is what the GDN layer uses in production, and a Python
    ``if`` inside a ``prim_func`` makes tilelang fall back to source re-parsing, which loses the
    factory's locals (measured: ``NameError: name 'C' is not defined``).  Conditional variants must
    use ``T.If``/``T.Then`` like the sibling kernel instead.
    """
    import tilelang.ascend.language as T

    C = channels
    W = width
    L = lines
    taps = W - 1
    per = (C + cores - 1) // cores

    @T.prim_func
    def main(
        X: T.Tensor((C,), dtype),
        State: T.Tensor((L, C, taps), dtype),
        Weight: T.Tensor((C, W), dtype),
        Bias: T.Tensor((C,), "float32"),
        line: T.int32,
        Out: T.Tensor((C,), dtype),
    ):
        with T.Kernel(cores) as bx:
            xt = T.alloc_shared((per,), dtype)
            st = T.alloc_shared((per, taps), dtype)
            wt = T.alloc_shared((per, W), dtype)
            bt = T.alloc_shared((per,), "float32")
            ot = T.alloc_shared((per,), dtype)

            base = bx * per
            T.copy(X[base : base + per], xt[0:per])
            T.copy(State[line, base : base + per, 0:taps], st[0:per, 0:taps])
            T.copy(Weight[base : base + per, 0:W], wt[0:per, 0:W])
            T.copy(Bias[base : base + per], bt[0:per])

            # Every ``T.Parallel`` must sit inside a VF block on Ascend NPU (measured:
            # "Parallel loops outside VF blocks are not supported on Ascend NPU"); the sibling kernel
            # wraps them in ``T.SimtVF``, so the same idiom is used here.
            with T.SimtVF(threads=128):
                for i in T.Parallel(per):
                    # Fully unrolled over the taps: a `T.serial` loop inside `T.Parallel` makes the
                    # accumulator an SSA value crossing a region boundary, which tilelang rejects
                    # ("Immutable variable `acc` is used outside its defining region").  `W` is a
                    # compile-time constant of this model (4), so unrolling costs nothing.
                    acc = (
                        bt[i]
                        + st[i, 0] * wt[i, 0]
                        + st[i, 1] * wt[i, 1]
                        + st[i, 2] * wt[i, 2]
                        + xt[i] * wt[i, 3]
                    )
                    ot[i] = T.Cast(dtype, acc / (1.0 + T.exp(0.0 - acc)))

                T.copy(ot[0:per], Out[base : base + per])
                # Shift in place, oldest column first: new[j] = old[j+1] reads ahead of the write.
                for i in T.Parallel(per):
                    st[i, 0] = st[i, 1]
                for i in T.Parallel(per):
                    st[i, 1] = st[i, 2]
                for i in T.Parallel(per):
                    st[i, 2] = xt[i]
            T.copy(st[0:per, 0:taps], State[line, base : base + per, 0:taps])

    return main


_KERNEL_CACHE: dict = {}


def build_conv_update_kernel(**spec):
    """Compile (and cache) the exact-shape kernel.

    Shapes are published into this module's globals *around* the compile: ``tilelang.compile``
    re-parses the ``prim_func`` source and resolves its annotations against the module globals, so a
    factory-local (``C = channels``) is invisible and the build dies with
    ``NameError: name 'C' is not defined`` (measured, §89).  The sibling kernels avoid this by using
    only module-level names in annotations; publishing around a narrow window is the local fix, and
    the previous values are restored in ``finally`` so two shapes cannot collide.
    """
    import sys

    import tilelang

    key = ("gdn_conv_update",) + tuple(sorted(spec.items()))
    kernel = _KERNEL_CACHE.get(key)
    if kernel is not None:
        return kernel

    module = sys.modules[__name__]
    channels = int(spec["channels"])
    width = int(spec["width"])
    cores = int(spec.get("cores", NUM_CORES))
    published = {
        "C": channels,
        "W": width,
        "L": int(spec["lines"]),
        "taps": width - 1,
        "cores": cores,
        "per": (channels + cores - 1) // cores,
        "dtype": str(spec.get("dtype", "bfloat16")),
    }
    saved = {name: module.__dict__.get(name) for name in published}
    module.__dict__.update(published)
    try:
        kernel = tilelang.compile(
            gdn_conv_update_kernel(**spec),
            out_idx=[],
            pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True},
        )
    finally:
        for name, value in saved.items():
            if value is None:
                module.__dict__.pop(name, None)
            else:
                module.__dict__[name] = value
    _KERNEL_CACHE[key] = kernel
    return kernel


def gdn_conv_update_kernel_available() -> bool:
    """Whether the experimental kernel *can* be built here (tilelang + the Ascend dialect)."""
    try:
        import tilelang  # noqa: F401
        import tilelang.ascend.language as T  # noqa: F401
    except Exception:  # pragma: no cover - no tilelang on a CPU dev box
        return False
    return True
