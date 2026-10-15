"""H29 实验:对 h26 融合内核本体(生产发射参数)做 strip 物理序单 hot 反推。

背景:mini 探针(h27/h29 探针)与 h26 numeric 三种 pack 全 FAIL 矛盾——外推失效。
本实验直接测 h26 moe_gate_up_swiglu_align(迷你几何 E=1,I=256,H=128,单 K-slice,
发射代码路径与生产逐语句相同):gate strip 全 1 → y = σ(1)·sem_u(线性可解码);
x=单位阵 [128,128](rows_slots=128=num_tiles 4×32),up strip 两轮编码(row/col)。

输出:P[槽(c,k)] = up strip flat 坐标 j(flat[j]=码),即 pack 契约的**实测**答案。

用法::

    .venv/bin/python benchmarks/probes/h29b_fused_align_probe.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch
import torch_npu  # noqa: F401

import tilelang
import h26_moe_prenz_align as h26

E_MINO, I_MINO, H_MINO = 1, 256, 128
BN, BK, BM = 256, 128, 32
NT = I_MINO // BN
STRIP = BN * H_MINO
ROWS_SLOTS = 128  # 4 tiles × 32
SIG1 = 0.7310585786300049  # σ(1)


def _build():
    return tilelang.compile(
        h26.moe_gate_up_swiglu_align(
            num_tiles_max=4, rows_slots=ROWS_SLOTS, num_experts=E_MINO,
            inter=I_MINO, hidden=H_MINO, block_m=BM, block_n=BN, block_k=BK,
        ), out_idx=[],
        pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True},
    )


def _run(kern, up_flat_code: np.ndarray) -> np.ndarray:
    """up_flat_code: [STRIP] bf16 精确码 → 返回 sem_u[c,k] 解码值 [BN, BK]。"""
    dev = "npu"
    wnz = torch.ones(E_MINO * 2 * NT * STRIP, dtype=torch.bfloat16, device=dev)
    up_base = (E_MINO * NT + 0) * STRIP + STRIP  # gate strip=0,up strip=1
    wnz[STRIP:2 * STRIP] = torch.from_numpy(up_flat_code).to(torch.bfloat16).to(dev)
    x = torch.eye(ROWS_SLOTS, dtype=torch.bfloat16, device=dev)
    ids = torch.zeros(4, dtype=torch.int32, device=dev)
    sids = torch.arange(ROWS_SLOTS, dtype=torch.int32, device=dev)
    y = torch.empty((ROWS_SLOTS, I_MINO), dtype=torch.bfloat16, device=dev)
    kern(x, wnz, ids, sids, y)
    torch.npu.synchronize()
    # nt=0 → y[:, 0:BN]=σ(1)·sem_u;行 m=k(x=I)
    yv = y.float().cpu().numpy()[:, :BN] / SIG1
    return yv.T  # [BN(c), BK(k=m)]


def main() -> int:
    dev = "npu"
    kern = _build()
    j = np.arange(STRIP, dtype=np.int64)

    # row 轮:码=0.5*((j%128)+2) → 解码 r=j%128
    code_r = (0.5 * ((j % BK).astype(np.float32) + 2.0)).astype(np.float32)
    sem_r = _run(kern, code_r)
    r = np.round(sem_r * 2).astype(np.int64) - 2

    # col 轮:码=1+(j//128) → 解码 cc=j//128 ∈[0,256)
    code_c = (1.0 + (j // BK).astype(np.float32)).astype(np.float32)
    sem_c = _run(kern, code_c)
    cc = np.round(sem_c).astype(np.int64) - 1

    ok = bool((r >= 0).all() and (r < BK).all() and (cc >= 0).all() and (cc < BN).all())
    print(f"decode in-range: {ok}")
    if not ok:
        print(f"r min/max={r.min()}/{r.max()} cc min/max={cc.min()}/{cc.max()}")
        return 1
    P = (r * BK + cc).astype(np.int64).reshape(-1)  # P[c*BK+k] = strip flat j
    out = Path(__file__).resolve().parent / "h29b_perm_fused_align.npy"
    np.save(out, P)
    identity = np.array_equal(P, np.arange(STRIP, dtype=np.int64))
    # 转置 pack 序:flat_t[k*BN+c]=W[c,k] → sem[c,k]=flat[c*BK+k] → P[c*BK+k]=?
    #   若 pack 后 align 语义=G[k,c](nd2nz 实测),恒等 pack(ND 序)应使 sem[c,k]=flat[c*BK+k] 对应 G[c,k]——判据直接看 P
    P2 = P.reshape(BN, BK)
    print("P2[:4,:8]=\n", P2[:4, :8])
    print("P2[0,:8] =", P2[0, :8].tolist())
    print("P2[:8,0] =", P2[:8, 0].tolist())
    print("IDENTITY(恒等):", identity)
    bij = np.array_equal(np.sort(P), np.arange(STRIP, dtype=np.int64))
    print("BIJECTION:", bij)
    print(f"SAVED: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
