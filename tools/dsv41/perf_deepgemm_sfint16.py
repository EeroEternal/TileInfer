"""deep_gemm fp8×fp4 GEMM：用 int16 (packed ue8m0x2) 的 scale-factor 再测。

DeepInfer 的 `PLATFORM.fp8_sf_type` 在 Ascend 侧是 **torch.int16**（见 platform.py:138），
不是 fp32 —— 这解释了此前"写入了但值全错、耗时与规模无关"的现象。

e8m0 语义：字节 = 指数 + 127（即 2^e 存 e+127）；两个 e8m0 字节打包进一个 int16（低字节在前），
所以 sf 的形状是 `[ceil(M/gran_m), ceil(K/gran_k/2)]`（K 轴折半）。

用法：
  export LD_PRELOAD=/home/userlist/zxs/TileInfer/.work/gcc13/lib64/libstdc++.so.6
  bash tools/dsv41/patch_deepgemm_headers.sh
  .venv/bin/python tools/dsv41/perf_deepgemm_sfint16.py
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


def sf_e8m0_bytes(sf: torch.Tensor) -> torch.Tensor:
    """2 的幂 -> e8m0 字节（指数 + 127）。"""
    exp = torch.round(torch.log2(sf.clamp_min(1e-30)))
    return (exp + 127.0).to(torch.int16).clamp(0, 255).to(torch.uint8)


def pack_sf_int16(sf_bytes: torch.Tensor) -> torch.Tensor:
    """[rows, cols] 的 e8m0 字节 -> [rows, cols/2] 的 int16（低字节在前）。"""
    rows, cols = sf_bytes.shape
    lo = sf_bytes[:, 0::2].to(torch.int32)
    hi = sf_bytes[:, 1::2].to(torch.int32)
    packed = (lo | (hi << 8)).to(torch.int16)
    return packed.contiguous()


def quant_fp8_block32(x, bk=BK):
    m, k = x.shape
    blk = x.view(m, k // bk, bk)
    sf = blk.abs().amax(dim=-1, keepdim=True).clamp_min(1e-9) / 448.0
    sf = torch.pow(2.0, torch.round(torch.log2(sf)))
    return (blk / sf).clamp(-448, 448).to(torch.float8_e4m3fn).reshape(m, k), sf.squeeze(-1).contiguous()


def quant_fp4_block32(w, bk=BK):
    n, k = w.shape
    blk = w.view(n, k // bk, bk)
    sf = blk.abs().amax(dim=-1, keepdim=True).clamp_min(1e-9) / 6.0
    sf = torch.pow(2.0, torch.round(torch.log2(sf)))
    idx = ((blk / sf).clamp(-6, 6).unsqueeze(-1) - FP4_LUT).abs().argmin(-1).to(torch.int32)
    idx = idx.reshape(n, k)
    return ((idx[:, 0::2] << 4) | idx[:, 1::2]).to(torch.uint8), sf.squeeze(-1).contiguous()


def reference(a_q, a_sf, b_packed, b_sf, m, n, k):
    a_ref = (a_q.float().view(m, k // BK, BK) * a_sf.unsqueeze(-1)).reshape(m, k)
    nb = b_packed.to(torch.int32)
    nib = torch.empty(n, k, dtype=torch.long)
    nib[:, 0::2] = nb >> 4
    nib[:, 1::2] = nb & 0xF
    b_ref = (FP4_LUT[nib].view(n, k // BK, BK) * b_sf.unsqueeze(-1)).reshape(n, k)
    return a_ref @ b_ref.t()


def one(deep_gemm, m, n, k):
    torch.manual_seed(0)
    x, w = torch.randn(m, k), torch.randn(n, k)
    a_q, a_sf = quant_fp8_block32(x)
    b_packed, b_sf = quant_fp4_block32(w)

    a_q_d = a_q.to(DEV)
    a_sf_d = pack_sf_int16(sf_e8m0_bytes(a_sf)).to(DEV)
    b_d = b_packed.view(torch.int8).to(DEV)
    b_sf_d = pack_sf_int16(sf_e8m0_bytes(b_sf)).to(DEV)
    d = torch.full((m, n), 7.0, dtype=torch.bfloat16, device=DEV)

    try:
        deep_gemm.fp8_fp4_gemm_nt((a_q_d, a_sf_d), (b_d, b_sf_d), d, None, recipe=(1, 1, BK))
        torch.npu.synchronize()
    except Exception as exc:  # noqa: BLE001
        print(f"  M={m:<5} N={n:<5} K={k:<5} CALL-FAILED {str(exc).splitlines()[0][:100]}")
        return

    wrote = not bool(torch.all(d == 7.0))
    ref = reference(a_q, a_sf, b_packed, b_sf, m, n, k)
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

    t1, t2 = t(10), t(110)
    per = (t2 * 110 - t1 * 10) / 100
    ok = "PASS" if rel < 5e-2 else "MISMATCH"
    print(f"  M={m:<5} N={n:<5} K={k:<5} sfA{tuple(a_sf_d.shape)} sfB{tuple(b_sf_d.shape)} "
          f"wrote={wrote!s:<5} rel={rel:.3e} per={per:.4f} ms  {ok}")


def main() -> int:
    import deep_gemm
    print(f"deep_gemm {getattr(deep_gemm, '__version__', '?')}  sf=int16 (packed ue8m0x2)")
    for m, n, k in [(512, 2048, 5120), (6, 2304, 5120), (1, 2304, 5120)]:
        one(deep_gemm, m, n, k)
    return 0


if __name__ == "__main__":
    sys.exit(main())
