"""Probe: Ascend MX block-scaled GEMM with packed FP4 weights.

This is the P1 go/no-go prototype.  The existing path dequantises the whole
expert weight to bf16 (``unpack_fp4_lut`` + broadcast scale) and then calls
``torch.bmm``.  The Ascend backend exposes ``T.gemm_blockscaled`` which applies
per-32-element MX scales inside the MAD, so the packed uint8 weights and the
e8m0 scales can be fed directly.

The probe answers three questions, in order:

1. does FP4 x FP4 blockscaled GEMM produce the same result as the CPU
   reference on a small tile?
2. what does a realistic expert tile (M=16, N=2304, K=5120) cost versus the
   current unpack + bmm path?
3. is mixed FP8 activation x FP4 weight accepted by the backend?  (If yes it
   avoids quantising the activation to FP4; if no the decode path uses FP4/FP4.)
"""

import time

import torch
import torch_npu
import tilelang
import tilelang.ascend.language as T

from tools.dsv41 import ref_cpu

FP4 = "float4_e2m1fn"
FP8 = "float8_e4m3fn"


def blockscaled_kernel(M: int, N: int, K: int, a_dtype: str, b_dtype: str, BN: int, BK: int):
    """One N tile per block, K reduction in BK chunks, FP32 accumulator."""
    assert K % BK == 0 and BK % 64 == 0 and N % BN == 0

    a_packed = a_dtype == FP4
    b_packed = b_dtype == FP4
    a_shape = (M, K // 2) if a_packed else (M, K)
    b_shape = (N, K // 2) if b_packed else (N, K)

    @T.prim_func
    def main(
        A: T.Tensor(a_shape, "int8" if a_packed else "float8_e4m3fn"),
        B: T.Tensor(b_shape, "int8"),
        SA: T.Tensor((K // 64, M), "uint16"),
        SB: T.Tensor((K // 64, N), "uint16"),
        C: T.Tensor((M, N), "float32"),
    ):
        with T.Kernel(N // BN) as bx:
            a_l1 = T.alloc_l1((M, BK), a_dtype)
            b_l1 = T.alloc_l1((BN, BK), b_dtype)
            sa_l1 = T.alloc_l1((M, BK // 64), "uint16")
            sb_l1 = T.alloc_l1((BN, BK // 64), "uint16")
            acc = T.alloc_l0c((M, BN), "float32")
            a_data = T.view(A, (M, K), a_dtype) if a_packed else A
            b_data = T.view(B, (N, K), b_dtype)
            for k in T.serial(K // BK):
                T.copy(a_data[0:M, k * BK : (k + 1) * BK], a_l1)
                T.copy(b_data[bx * BN : (bx + 1) * BN, k * BK : (k + 1) * BK], b_l1)
                T.copy(SA[k * (BK // 64) : (k + 1) * (BK // 64), 0:M], sa_l1, transpose=True)
                T.copy(SB[k * (BK // 64) : (k + 1) * (BK // 64), bx * BN : (bx + 1) * BN], sb_l1, transpose=True)
                T.gemm_blockscaled(
                    a_l1, b_l1, acc, sa_l1, sb_l1, transpose_B=True, clear_accum=(k == 0)
                )
            T.copy(acc, C[0:M, bx * BN : (bx + 1) * BN])

    return main


def _sf_u16(scale):
    """[rows, K/32] e8m0 bytes -> [K/64, rows] uint16 (two scales per 64 K)."""
    if scale.dtype != torch.float8_e8m0fnu:
        scale = scale.to(torch.float8_e8m0fnu)
    return scale.view(torch.uint8).view(torch.uint16).T.contiguous()


def _ref_fp4(packed, scale, M, K, dtype=torch.float32):
    w = ref_cpu.unpack_fp4_lut(packed, M, K, dtype=dtype)
    s = ref_cpu.scale_to_float(scale.view(torch.float8_e8m0fnu))
    return (w.reshape(M, K // 32, 32) * s.unsqueeze(-1)).reshape(M, K)


def _pack(x):
    return ref_cpu.quantize_weight_fp4_block32(x)


def check_fp4_fp4():
    M, N, K = 32, 32, 256
    torch.manual_seed(0)
    x = torch.randn(M, K) * 0.25
    w = torch.randn(N, K) * 0.25
    pa, sa = _pack(x)
    pb, sb = _pack(w)
    out = torch.empty(M, N, dtype=torch.float32, device="npu")
    k = tilelang.compile(blockscaled_kernel(M, N, K, FP4, FP4, 16, 256), target="ascend", out_idx=-1)
    out = k(pa.npu().view(torch.int8), pb.npu().view(torch.int8),
            _sf_u16(sa).npu(), _sf_u16(sb).npu()).cpu()
    ref = _ref_fp4(pa, sa, M, K) @ _ref_fp4(pb, sb, N, K).T
    diff = (out.cpu() - ref).abs().max().item()
    print(f"FP4xFP4 M={M} N={N} K={K} maxdiff={diff:.3e}")
    return diff


def check_fp8_fp4():
    M, N, K = 32, 32, 256
    torch.manual_seed(1)
    x = torch.randn(M, K) * 0.25
    w = torch.randn(N, K) * 0.25
    qa, sa = ref_cpu.act_quant(x, 32, "ue8m0", torch.float8_e8m0fnu, False)
    pb, sb = _pack(w)
    out = torch.empty(M, N, dtype=torch.float32, device="npu")
    try:
        k = tilelang.compile(blockscaled_kernel(M, N, K, FP8, FP4, 16, 256), target="ascend", out_idx=-1)
        out = k(qa.npu(), pb.npu().view(torch.int8),
                _sf_u16(sa).npu(), _sf_u16(sb).npu()).cpu()
        torch.npu.synchronize()
    except Exception as e:  # noqa: BLE001
        print(f"FP8xFP4 unsupported: {str(e)[:300]}")
        return None
    ref = (qa.float() * ref_cpu.scale_to_float(sa).repeat_interleave(32, dim=-1)) @ _ref_fp4(pb, sb, N, K).T
    diff = (out.cpu() - ref).abs().max().item()
    print(f"FP8xFP4 M={M} N={N} K={K} maxdiff={diff:.3e}")
    return diff


def check_shape(M, N, K, BN, BK):
    torch.manual_seed(7)
    x = torch.randn(M, K) * 0.25
    w = torch.randn(N, K) * 0.25
    pa, sa = _pack(x)
    pb, sb = _pack(w)
    k = tilelang.compile(blockscaled_kernel(M, N, K, FP4, FP4, BN, BK), target="ascend", out_idx=-1)
    out = k(pa.npu().view(torch.int8), pb.npu().view(torch.int8),
            _sf_u16(sa).npu(), _sf_u16(sb).npu()).cpu()
    ref = _ref_fp4(pa, sa, M, K) @ _ref_fp4(pb, sb, N, K).T
    diff = (out - ref).abs().max().item()
    print(f"check M={M} N={N} K={K} BN={BN} BK={BK} maxdiff={diff:.3e}")
    return diff


def check_npu_quant_gemm(M, N, K):
    """Quantise the activation with torch_npu MXFP4 and feed it to the kernel."""
    torch.manual_seed(13)
    x = torch.randn(M, K, device="npu", dtype=torch.bfloat16)
    w = torch.randn(N, K) * 0.25
    pb, sb = _pack(w)
    q, s = torch_npu.npu_dynamic_mx_quant(x, dst_type=296, block_size=32)
    k = tilelang.compile(blockscaled_kernel(M, N, K, FP4, FP4, 128, 512), target="ascend", out_idx=-1)
    sa_t = s.reshape(M, K // 64, 2).view(torch.uint16).reshape(M, K // 64).T.contiguous()
    out = k(q.view(torch.int8), pb.npu().view(torch.int8),
            sa_t, _sf_u16(sb).npu()).cpu()
    xd = ref_cpu.unpack_fp4_lut(q, M, K, dtype=torch.float32)
    sf = s.reshape(M, K // 32).view(torch.float8_e8m0fnu)
    xd = (xd.reshape(M, K // 32, 32) * ref_cpu.scale_to_float(sf).unsqueeze(-1)).reshape(M, K)
    ref = xd.cpu().float() @ _ref_fp4(pb, sb, N, K).T
    diff = (out - ref).abs().max().item()
    print(f"npu-quant M={M} N={N} K={K} maxdiff={diff:.3e}")
    return diff


def _deq_npu(q, s, M, K):
    xd = ref_cpu.unpack_fp4_lut(q, M, K, dtype=torch.float32)
    sf = s.reshape(M, K // 32).view(torch.float8_e8m0fnu)
    return (xd.reshape(M, K // 32, 32) * ref_cpu.scale_to_float(sf).unsqueeze(-1)).reshape(M, K)


def check_quality_res(M, N, K):
    """Two-term FP4 residual: x ~= x0 + x1, both MXFP4, two block-scaled GEMMs."""
    torch.manual_seed(11)
    x = torch.randn(M, K, device="npu", dtype=torch.bfloat16)
    w = torch.randn(N, K) * 0.25
    exact = x.float().cpu() @ w.T
    pb, sb = _pack(w)
    q0, s0 = torch_npu.npu_dynamic_mx_quant(x, dst_type=296, block_size=32)
    x0 = _deq_npu(q0, s0, M, K)
    r = (x.float() - x0).to(torch.bfloat16)
    q1, s1 = torch_npu.npu_dynamic_mx_quant(r, dst_type=296, block_size=32)
    k = tilelang.compile(blockscaled_kernel(M, N, K, FP4, FP4, 128, 512), target="ascend", out_idx=-1)
    wb = pb.npu().view(torch.int8)
    sw = _sf_u16(sb).npu()
    out = k(q0.view(torch.int8), wb, s0.reshape(M, K // 64, 2).view(torch.uint16).reshape(M, K // 64).T.contiguous(), sw)
    out = out + k(q1.view(torch.int8), wb, s1.reshape(M, K // 64, 2).view(torch.uint16).reshape(M, K // 64).T.contiguous(), sw)
    out = out.cpu()
    err = (out - exact).norm() / exact.norm()
    print(f"quality fp4+residual: rel_fro={err.item():.4e} maxabs={(out - exact).abs().max().item():.4e}")
    return err


def check_quality(M, N, K):
    """FP4 activation error vs the current FP8-activation + dequant-weight path."""
    torch.manual_seed(11)
    x = torch.randn(M, K)
    w = torch.randn(N, K) * 0.25
    exact = x @ w.T
    # current path: FP8 activation, dequantised bf16 weight, bf16 bmm
    qa, sa = ref_cpu.act_quant(x, 32, "ue8m0", torch.float8_e8m0fnu, False)
    xf = (qa.float() * ref_cpu.scale_to_float(sa).repeat_interleave(32, dim=-1)).to(torch.bfloat16)
    pb, sb = _pack(w)
    wf = ref_cpu.unpack_fp4_lut(pb, N, K, dtype=torch.bfloat16)
    wf = (wf.reshape(N, K // 32, 32) * ref_cpu.scale_to_float(sb).to(torch.bfloat16).unsqueeze(-1)).reshape(N, K)
    cur = torch.bmm(xf.unsqueeze(0), wf.transpose(0, 1).unsqueeze(0))[0].float()
    # new path: FP4 activation, packed weight, blockscaled GEMM
    pa4, sa4 = _pack(x)
    k = tilelang.compile(blockscaled_kernel(M, N, K, FP4, FP4, 128, 512), target="ascend", out_idx=-1)
    new = k(pa4.npu().view(torch.int8), pb.npu().view(torch.int8),
            _sf_u16(sa4).npu(), _sf_u16(sb).npu()).cpu()
    for name, y in (("current", cur), ("fp4act", new)):
        err = (y - exact).norm() / exact.norm()
        mx = (y - exact).abs().max().item()
        print(f"quality {name}: rel_fro={err.item():.4e} maxabs={mx:.4e}")
    d = (new - cur).abs()
    print(f"quality new-vs-current: rel_fro={(d.norm() / cur.norm()).item():.4e} maxabs={d.max().item():.4e}")


def bench(M, N, K, BN, BK, a_dtype=FP4, iters=50):
    torch.manual_seed(2)
    x = torch.randn(M, K) * 0.25
    w = torch.randn(N, K) * 0.25
    if a_dtype == FP4:
        pa, sa = _pack(x)
        a = pa.npu().view(torch.int8)
        sa_t = _sf_u16(sa).npu()
    else:
        qa, sa = ref_cpu.act_quant(x, 32, "ue8m0", torch.float8_e8m0fnu, False)
        a = qa.npu()
        sa_t = _sf_u16(sa).npu()
    pb, sb = _pack(w)
    b = pb.npu().view(torch.int8)
    sb_t = _sf_u16(sb).npu()
    k = tilelang.compile(blockscaled_kernel(M, N, K, a_dtype, FP4, BN, BK), target="ascend", out_idx=-1)
    for _ in range(5):
        k(a, b, sa_t, sb_t)
    torch.npu.synchronize()
    ev0, ev1 = torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)
    ev0.record()
    for _ in range(iters):
        k(a, b, sa_t, sb_t)
    ev1.record()
    ev1.synchronize()
    dt = ev0.elapsed_time(ev1) / iters
    print(f"blockscaled {a_dtype}xFP4 M={M} N={N} K={K} BN={BN} BK={BK}: {dt * 1e3:.3f} us  ({dt:.5f} ms)")
    return dt


def bench_torch(M, N, K, iters=20):
    """Current batched path: unpack + scale + bmm for one expert tile."""
    torch.manual_seed(2)
    w = torch.randn(N, K) * 0.25
    pb, sb = _pack(w)
    packed, scale = pb.npu(), sb.npu()
    xq, xs = ref_cpu.act_quant(torch.randn(M, K) * 0.25, 32, "ue8m0", torch.float8_e8m0fnu, False)
    xq, xs = xq.npu(), xs.npu()
    # warm
    wf = ref_cpu.unpack_fp4_lut(packed, N, K, dtype=torch.bfloat16)
    wf = (wf.reshape(N, K // 32, 32) * ref_cpu.scale_to_float(scale).to(torch.bfloat16).unsqueeze(-1)).reshape(N, K)
    xf = (xq.float() * ref_cpu.scale_to_float(xs).repeat_interleave(32, dim=-1)).to(torch.bfloat16)
    for _ in range(3):
        torch.bmm(xf.unsqueeze(0), wf.transpose(0, 1).unsqueeze(0))
    torch.npu.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        wf = ref_cpu.unpack_fp4_lut(packed, N, K, dtype=torch.bfloat16)
        wf = (wf.reshape(N, K // 32, 32) * ref_cpu.scale_to_float(scale).to(torch.bfloat16).unsqueeze(-1)).reshape(N, K)
        torch.bmm(xf.unsqueeze(0), wf.transpose(0, 1).unsqueeze(0))
    torch.npu.synchronize()
    dt = (time.perf_counter() - t0) / iters * 1e3
    print(f"torch unpack+bmm M={M} N={N} K={K}: {dt:.3f} ms")
    return dt


def main():
    print("npu", torch.npu.is_available())
    check_fp4_fp4()
    check_fp8_fp4()
    # realistic expert shapes: w13 = [2*2304, 5120], w2 = [5120, 2304]
    for (M, N, K, BN, BK) in [
        (32, 512, 5120, 128, 512),
        (32, 2304, 5120, 128, 512),
        (32, 576, 2304, 64, 256),
        (32, 5120, 2304, 128, 256),
    ]:
        try:
            bench(M, N, K, BN, BK)
        except Exception as e:  # noqa: BLE001
            print(f"bench M={M} N={N} K={K} ERR {str(e)[:300]}")
    bench_torch(32, 2304, 5120)
    bench_torch(32, 5120, 2304)


if __name__ == "__main__":
    main()
