"""H29 设备探针:生产几何(strided GM slice)下的 nd2nz 物理序实测。

E76 缺口:探针用连续 ND [BN,BK] 实测"转置连续",但生产内核加载的是带行 stride 的
GM slice(T.copy(W13[eid, n0:n0+BN, k0:k0+BK], w_l1),行 stride=H=2048/I=512),
三种 pack(compact-NZ/转置/恒等)numeric 全 FAIL 证明 strided 序 ≠ 连续序。
本探针按生产几何实测:GM 大张量 strided slice → L1(nd2nz)→ gemm 单 hot 反推。

- row 轮:W[n,k]=0.5*(k+2)(seg 限段),x=单位阵 → 解码 k;
- col 轮:W[n,k]=1+n,x=单位阵 → 解码 n;
- P[槽 c*BK+k] = k*STRIDE? 不——P 以 [BN, BK] 视图 flat(n*BK+k)为 GM 坐标。

用法(设备,NPU 空闲窗)::

    .venv/bin/python benchmarks/probes/h29_strided_l1_layout_probe.py --shape gu
    .venv/bin/python benchmarks/probes/h29_strided_l1_layout_probe.py --shape dn --seg 0..3
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch
import torch_npu  # noqa: F401

import tilelang
import tilelang.ascend.language as T

SHAPES = {
    "gu": dict(BN=256, BK=128, STRIDE=2048),  # gate_up: W13 slice [256,128], 行 stride=H
    "dn": dict(BN=64, BK=512, STRIDE=512),    # down: W2 slice [64,512], 行 stride=I
}
BM = 128


def _kernel(bn: int, bk: int, stride: int):

    @T.prim_func
    def main(
        Wbig: T.Tensor((bn, stride), "bfloat16"),
        X: T.Tensor((BM, bk), "bfloat16"),
        Y: T.Tensor((BM, bn), "bfloat16"),
    ):
        with T.Kernel(1) as bx:
            x_l1 = T.alloc_l1((BM, bk), "bfloat16")
            w_l1 = T.alloc_l1((bn, bk), "bfloat16")
            ga = T.alloc_l0a((BM, bk), "bfloat16")
            gb = T.alloc_l0b((bn, bk), "bfloat16")
            acc = T.alloc_l0c((BM, bn), "float32")
            o_f32 = T.alloc_shared((BM // 2, bn), "float32")
            o_cast = T.alloc_shared((BM // 2, bn), "bfloat16")

            T.copy(X[0:BM, 0:bk], x_l1)
            T.copy(Wbig[0:bn, 0:bk], w_l1)  # 生产同款:strided GM slice → L1(nd2nz)
            T.copy(x_l1, ga)
            T.copy(w_l1, gb)
            T.gemm(ga, gb, acc, transpose_B=True, clear_accum=True)
            T.dual_copy(acc, o_f32)
            with T.SimtVF(threads=128):
                for r, c in T.Parallel(BM // 2, bn):
                    o_cast[r, c] = T.Cast("bfloat16", o_f32[r, c])
            T.dual_copy(o_cast[0:BM // 2, 0:bn], Y[0:BM, 0:bn])

    return main


def _meas(kern, wbig: torch.Tensor, axis: str, seg: int, bk: int, bn: int) -> np.ndarray:
    """一轮编码实验 → 解码 sem[c, k](int64 [bn, bk],越界段=-1)。"""
    w = torch.zeros((bn, wbig.shape[1]), dtype=torch.bfloat16, device=wbig.device)
    if axis == "row":
        enc = (np.arange(bk, dtype=np.float32) + 2) * 0.5
        w[:bn, :bk] = torch.from_numpy(np.broadcast_to(enc[None, :], (bn, bk))).to(torch.bfloat16).to(wbig.device)
    else:
        enc = np.arange(bn, dtype=np.float32) + 1.0
        w[:bn, :bk] = torch.from_numpy(np.broadcast_to(enc[:, None], (bn, bk))).to(torch.bfloat16).to(wbig.device)
    if seg >= 0:  # 段外清零(row 轮分段用)
        w[:bn, :bk] = 0
        lo, hi = seg * 128, seg * 128 + 128
        if axis == "row":
            enc_s = (np.arange(hi - lo, dtype=np.float32) + 2) * 0.5
            w[:bn, lo:hi] = torch.from_numpy(np.broadcast_to(enc_s[None, :], (bn, hi - lo))).to(torch.bfloat16).to(wbig.device)
        else:
            enc_c = np.arange(bn, dtype=np.float32) + 1.0
            w[:bn, lo:hi] = torch.from_numpy(np.broadcast_to(enc_c[:, None], (bn, hi - lo))).to(torch.bfloat16).to(wbig.device)
    wbig.copy_(w)

    x = torch.eye(BM, dtype=torch.bfloat16, device=wbig.device)
    if bk > BM:  # x=移位单位阵切片:x[m,k]=1 iff k==seg*128+m 的推广——此处 x 为 [BM,bk]
        x = torch.zeros((BM, bk), dtype=torch.bfloat16, device=wbig.device)
        lo = seg * 128 if seg >= 0 else 0
        for m in range(min(BM, bk - lo)):
            x[m, lo + m] = 1.0
    y = torch.empty((BM, bn), dtype=torch.bfloat16, device=wbig.device)
    kern(wbig, x, y)
    torch.npu.synchronize()
    yv = y.float().cpu().numpy()[:BM]
    n_keep = min(BM, bk)
    sem = np.full((bn, bk), -1.0)
    if bk <= BM:
        sem = yv[:bk].T  # [c, k]
    else:
        lo = seg * 128 if seg >= 0 else 0
        sem[:, lo:lo + n_keep] = yv[:n_keep].T
    if axis == "row":
        return np.round(sem * 2).astype(np.int64) - 2
    return np.round(sem).astype(np.int64) - 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shape", choices=list(SHAPES), required=True)
    ap.add_argument("--seg", type=int, default=-1, help="dn 形状的 128 宽段号(0..3);-1=全段(gu)")
    args = ap.parse_args()
    sp = SHAPES[args.shape]
    BN, BK, STRIDE = sp["BN"], sp["BK"], sp["STRIDE"]
    dev = "npu"
    kern = tilelang.compile(_kernel(BN, BK, STRIDE), out_idx=[])

    seg = args.seg
    wbig = torch.zeros((BN, STRIDE), dtype=torch.bfloat16, device=dev)
    rows = _meas(kern, wbig, "row", seg, BK, BN)
    cols = _meas(kern, wbig, "col", seg, BK, BN)
    ks = slice(seg * 128, seg * 128 + 128) if seg >= 0 else slice(0, BK)
    r = rows[:, ks]
    c = cols[:, ks]
    sanity = bool((r >= 0).all() and (r < r.shape[1]).all() and (c >= 0).all() and (c < BN).all())
    k_off = seg * 128 if seg >= 0 else 0
    print(f"[{args.shape} seg={seg}] decode in-range: {sanity}")
    if not sanity:
        print(f"  rows min/max={r.min()}/{r.max()} cols min/max={c.min()}/{c.max()}")
        print(f"  rows uniq(first 16)={np.unique(r)[:16].tolist()}")
        print(f"  cols uniq(first 16)={np.unique(c)[:16].tolist()}")
        return 1
    # P 以 [BN, BK] 视图 flat(n*BK+k)为 GM 坐标;槽 (c,k) → GM flat = r[c,k]*BK + c[c,k]
    if seg >= 0:
        cc = np.broadcast_to(c[:, :1], r.shape)  # col 轮全列同码,取首列广播
    else:
        cc = c
    P = (r.astype(np.int64) * BK + cc.astype(np.int64)).reshape(-1)
    out = Path(__file__).resolve().parent / f"h29_perm_{args.shape}_seg{seg}.npy"
    np.save(out, P)
    print(f"  rows[:2,:8]={r[:2,:8].tolist()}")
    print(f"  cols[:2,:8]={c[:2,:8].tolist()}")
    print(f"  P[:8]={P[:8].tolist()}")
    bij = np.array_equal(np.sort(P), np.arange(BN * r.shape[1]))
    print(f"  bijection(on segment): {bij}")
    print(f"SAVED: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
