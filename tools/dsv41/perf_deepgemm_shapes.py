"""判定 deep_gemm 的 fp8×fp4 GEMM 对我们真实形状是否真的执行（而不是静默跳过）。

背景：上一轮 M=512,N=2048,K=5120 调用后 `d` 仍为全 0、耗时 0.027 ms（对 5.4 GFLOP 不可能），
怀疑 `get_gemm_desc` 在 `gemm_early_return(...)` 处返回 nullopt，函数静默 return。
本脚本用**哨兵值**填充 `d`（7.0），调用后检查是否被覆写，并对多组形状给出数值相对误差。

用法：
  export LD_PRELOAD=/home/userlist/zxs/TileInfer/.work/gcc13/lib64/libstdc++.so.6
  bash tools/dsv41/patch_deepgemm_headers.sh
  .venv/bin/python tools/dsv41/perf_deepgemm_shapes.py
"""

from __future__ import annotations

import statistics
import sys
import time

import torch

FP4_LUT = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
     -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0]
)
BK = 32
DEV = "npu:0"
SENTINEL = 7.0


def quant_fp8_block32(x, bk=BK):
    m, k = x.shape
    blk = x.view(m, k // bk, bk)
    sf = blk.abs().amax(dim=-1, keepdim=True).clamp_min(1e-9) / 448.0
    # e8m0 语义：sf 必须是 2 的整数次幂（内核要求 fp32 的 23 位尾数为 0）
    sf = torch.pow(2.0, torch.round(torch.log2(sf)))
    return (blk / sf).clamp(-448, 448).to(torch.float8_e4m3fn).reshape(m, k), sf.squeeze(-1).contiguous(), sf


def quant_fp4_block32(w, bk=BK):
    n, k = w.shape
    blk = w.view(n, k // bk, bk)
    sf = blk.abs().amax(dim=-1, keepdim=True).clamp_min(1e-9) / 6.0
    sf = torch.pow(2.0, torch.round(torch.log2(sf)))   # e8m0：2 的整数次幂
    idx = ((blk / sf).clamp(-6, 6).unsqueeze(-1) - FP4_LUT).abs().argmin(-1).to(torch.int32)
    idx = idx.reshape(n, k)
    return ((idx[:, 0::2] << 4) | idx[:, 1::2]).to(torch.uint8), sf.squeeze(-1).contiguous(), sf


def reference(a_q, a_sf3, b_packed, b_sf3, m, n, k):
    a_ref = (a_q.float().view(m, k // BK, BK) * a_sf3).reshape(m, k)
    nb = b_packed.to(torch.int32)
    nib = torch.empty(n, k, dtype=torch.long)
    nib[:, 0::2] = nb >> 4
    nib[:, 1::2] = nb & 0xF
    b_ref = (FP4_LUT[nib].view(n, k // BK, BK) * b_sf3).reshape(n, k)
    return a_ref @ b_ref.t()


def one(deep_gemm, m: int, n: int, k: int, reps=(10, 110)):
    torch.manual_seed(0)
    x = torch.randn(m, k)
    w = torch.randn(n, k)
    a_q, a_sf, a_sf3 = quant_fp8_block32(x)
    b_packed, b_sf, b_sf3 = quant_fp4_block32(w)

    a_q_d, a_sf_d = a_q.to(DEV), a_sf.to(DEV)
    b_d, b_sf_d = b_packed.view(torch.int8).to(DEV), b_sf.to(DEV)
    d = torch.full((m, n), SENTINEL, dtype=torch.bfloat16, device=DEV)

    try:
        deep_gemm.fp8_fp4_gemm_nt((a_q_d, a_sf_d), (b_d, b_sf_d), d, None, recipe=(1, 1, BK))
        torch.npu.synchronize()
    except Exception as exc:  # noqa: BLE001
        print(f"  M={m:<5} N={n:<5} K={k:<5} CALL-FAILED {str(exc).splitlines()[0][:90]}")
        return

    wrote = not bool(torch.all(d == SENTINEL))
    ref = reference(a_q, a_sf3, b_packed, b_sf3, m, n, k)
    got = d.float().cpu()
    rel = float((got - ref).abs().max()) / max(float(ref.abs().max()), 1e-6)

    def t(nn_):
        ts = []
        for _ in range(5):
            torch.npu.synchronize()
            t0 = time.perf_counter()
            for _ in range(nn_):
                deep_gemm.fp8_fp4_gemm_nt((a_q_d, a_sf_d), (b_d, b_sf_d), d, None, recipe=(1, 1, BK))
            torch.npu.synchronize()
            ts.append((time.perf_counter() - t0) / nn_ * 1e3)
        return statistics.median(ts)

    n1, n2 = reps
    t1, t2 = t(n1), t(n2)
    per = (t2 * n2 - t1 * n1) / (n2 - n1)
    print(f"  M={m:<5} N={n:<5} K={k:<5} wrote={wrote!s:<5} rel={rel:.3e}  per-iter={per:.4f} ms")


def main() -> int:
    import deep_gemm
    print(f"deep_gemm {getattr(deep_gemm, '__version__', '?')}  (sentinel {SENTINEL} -> 若未被覆写即为静默跳过)")
    # 我们真实的 MoE 形状在最后三组：M=1/6（decode）、M=512（prefill 批量）
    for m, n, k in [
        (512, 2048, 5120),
        (64, 2304, 5120),
        (256, 2304, 5120),
        (1, 2304, 5120),
        (6, 2304, 5120),
    ]:
        one(deep_gemm, m, n, k)
    return 0


if __name__ == "__main__":
    sys.exit(main())
