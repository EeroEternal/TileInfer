"""H27 设备探针:实测 MoE 权重 L1 物理序(GEMM 单 hot 反推)。

背景(E75/H27):h26 align 直载数值错乱(rel_y~4.6),E70「L1 物理序=compact-NZ」静态
反推被推翻;tilelang codegen 无 asc_copy_l12ub,L1 无法直接读回。本探针用**可逆编码 +
单位阵输入的 GEMM** 间接测出两种加载路径下 L1 内容在 gemm 语义视图下的排列:

- 轮 A(行编码):W[k,c] = 0.5*(k+2),x = I → y[k,c] = W_sem[c,k] → 解出每个语义槽
  (c,k) 读到的 GM 行索引;
- 轮 B(列编码):W[k,c] = 1+c,x = I → 解出 GM 列索引;
- 合成完整置换 P[c*BK+k] = k_row*BK + c_col(以 [BN,BK] ND 平坦序为 GM 坐标),对两种
  路径各测一遍:
    * ``nd2nz``:T.copy(GM ND → L1)——生产内核的加载,其物理序 = 「正确序」权威;
    * ``align``:asc_copy_gm2l1_align 平载 GM flat——h26 直载路径,理论=恒等。

判读(CPU 侧自动,见输出):
    ALIGN_EQ_ND2NZ   两路径物理序一致 → h26 失败不在布局对齐假设,另查 burst 语义
    ALIGN_IDENTITY   align=恒等平载 → packer 修正公式 = nd2nz 实测物理序
    其他             打印前 16 槽映射供归因

**已裁决(E76,2026-10-15)**:nd2nz 实测物理序 = 转置连续序(列 c 外轴、行 k 内轴,
32768/32768),E70 compact-NZ 静态反推被推翻;align 侧在 ND 序输入下 rows 解码=槽列号
(0..255,超 BK 范围)与「恒等平载」精确自洽。packer 修正 = 每 [BN,K] slice
``transpose(-2,-1)`` 后平坦化(见 h26_moe_prenz_align.py 的 pack_gate_up/pack_down)。

用法(设备,NPU 空闲窗)::

    source .agent_loop/run/env.sh
    .venv/bin/python benchmarks/probes/h27_l1_layout_probe.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch
import torch_npu  # noqa: F401

import tilelang
import tilelang.ascend.language as T
from tilelang.language.tir.op import call_extern

BN = 256
BK = 128
BM = 128  # == BK:单位阵一次放下(BM 不影响权重 L1 slice 内部几何)
BURSTS = BN * BK * 2 // 4096
BLEN = 4096


def _align_kernel():

    @T.prim_func
    def main(
        Wflat: T.Tensor((BN * BK,), "bfloat16"),
        X: T.Tensor((BM, BK), "bfloat16"),
        Y: T.Tensor((BM, BN), "bfloat16"),
    ):
        with T.Kernel(1) as bx:
            x_l1 = T.alloc_l1((BM, BK), "bfloat16")
            w_l1 = T.alloc_l1((BN, BK), "bfloat16")
            ga = T.alloc_l0a((BM, BK), "bfloat16")
            gb = T.alloc_l0b((BN, BK), "bfloat16")
            acc = T.alloc_l0c((BM, BN), "float32")
            o_f32 = T.alloc_shared((BM // 2, BN), "float32")
            o_cast = T.alloc_shared((BM // 2, BN), "bfloat16")

            T.copy(X[0:BM, 0:BK], x_l1)
            T.evaluate(call_extern("void", "asc_set_gm2l1_loop_size", 1, 1))
            T.evaluate(call_extern("void", "asc_set_gm2l1_loop1_stride", 0, 0))
            T.evaluate(call_extern("void", "asc_set_gm2l1_loop2_stride", 0, 0))
            T.evaluate(call_extern(
                "void", "asc_copy_gm2l1_align",
                T.access_ptr(w_l1, "r"),
                T.access_ptr(Wflat[0], "r"),
                BURSTS, BLEN, 0, 0, 0, 0, BLEN, BLEN))
            T.copy(x_l1, ga)
            T.copy(w_l1, gb)
            T.gemm(ga, gb, acc, transpose_B=True, clear_accum=True)
            T.dual_copy(acc, o_f32)
            with T.SimtVF(threads=128):
                for r, c in T.Parallel(BM // 2, BN):
                    o_cast[r, c] = T.Cast("bfloat16", o_f32[r, c])
            T.dual_copy(o_cast[0:BM // 2, 0:BN], Y[0:BM, 0:BN])

    return main


def _nd2nz_kernel():

    @T.prim_func
    def main(
        W: T.Tensor((BN, BK), "bfloat16"),
        X: T.Tensor((BM, BK), "bfloat16"),
        Y: T.Tensor((BM, BN), "bfloat16"),
    ):
        with T.Kernel(1) as bx:
            x_l1 = T.alloc_l1((BM, BK), "bfloat16")
            w_l1 = T.alloc_l1((BN, BK), "bfloat16")
            ga = T.alloc_l0a((BM, BK), "bfloat16")
            gb = T.alloc_l0b((BN, BK), "bfloat16")
            acc = T.alloc_l0c((BM, BN), "float32")
            o_f32 = T.alloc_shared((BM // 2, BN), "float32")
            o_cast = T.alloc_shared((BM // 2, BN), "bfloat16")

            T.copy(X[0:BM, 0:BK], x_l1)
            T.copy(W[0:BN, 0:BK], w_l1)
            T.copy(x_l1, ga)
            T.copy(w_l1, gb)
            T.gemm(ga, gb, acc, transpose_B=True, clear_accum=True)
            T.dual_copy(acc, o_f32)
            with T.SimtVF(threads=128):
                for r, c in T.Parallel(BM // 2, BN):
                    o_cast[r, c] = T.Cast("bfloat16", o_f32[r, c])
            T.dual_copy(o_cast[0:BM // 2, 0:BN], Y[0:BM, 0:BN])

    return main


def _meas(kernel, w_input, dev: str, axis: str) -> np.ndarray:
    """跑一轮编码实验,返回解码后的 sem[c, k](int64 [BN, BK])。"""
    if axis == "row":
        enc = (np.arange(BK, dtype=np.float32) + 2) * 0.5  # 1.0..64.0,bf16 精确
        w = np.broadcast_to(enc[None, :], (BN, BK)).copy()
    else:
        enc = np.arange(BN, dtype=np.float32) + 1.0  # 1..256,bf16 精确
        w = np.broadcast_to(enc[:, None], (BN, BK)).copy()
    w_t = torch.from_numpy(w).to(torch.bfloat16).to(dev).contiguous()
    if w_input == "flat":
        w_t = w_t.reshape(-1).contiguous()
    x = torch.eye(BK, dtype=torch.bfloat16, device=dev)
    y = torch.empty((BM, BN), dtype=torch.bfloat16, device=dev)
    kernel(w_t, x, y)
    torch.npu.synchronize()
    yv = y.float().cpu().numpy()[:BK]  # y[k, c] = W_sem[c, k](x=I)
    sem = yv.T  # [BN, BK] = [c, k]
    if axis == "row":
        return np.round(sem * 2).astype(np.int64) - 2
    return np.round(sem).astype(np.int64) - 1


def main() -> int:
    dev = "npu"
    ok = True
    perms = {}
    for mode, kern_fn, w_input in (("nd2nz", _nd2nz_kernel, "nd"), ("align", _align_kernel, "flat")):
        kern = tilelang.compile(kern_fn(), out_idx=[])
        rows = _meas(kern, w_input, dev, "row")
        cols = _meas(kern, w_input, dev, "col")
        sanity = bool((rows >= 0).all() and (rows < BK).all() and (cols >= 0).all() and (cols < BN).all())
        print(f"[{mode}] decode in-range: {sanity}")
        if not sanity:
            ok = False
            # 取证:打印原始解码值样本与模式统计(不 continue,仍保留超出部分供归因)
            rc_ = np.clip(rows, -1, 1 << 20)
            cc_ = np.clip(cols, -1, 1 << 20)
            print(f"[{mode}] rows uniq(first 32)={np.unique(rows)[:32].tolist()}")
            print(f"[{mode}] cols uniq(first 32)={np.unique(cols)[:32].tolist()}")
            print(f"[{mode}] rows min/max={rows.min()}/{rows.max()} cols min/max={cols.min()}/{cols.max()}")
            P_bad = (rc_.astype(np.int64) * BK + cc_.astype(np.int64)).reshape(-1)
            np.save(Path(__file__).resolve().parent / f"h27_perm_{mode}_bad.npy", P_bad)
            perms[mode] = P_bad
            continue
        P = (rows.astype(np.int64) * BK + cols.astype(np.int64)).reshape(-1)
        perms[mode] = P
        np.save(Path(__file__).resolve().parent / f"h27_perm_{mode}.npy", P)

    if "nd2nz" in perms and "align" in perms:
        identity = np.arange(BN * BK, dtype=np.int64)
        align_is_identity = bool(np.array_equal(perms["align"], identity))
        equal = bool(np.array_equal(perms["align"], perms["nd2nz"]))
        diff = int((perms["align"] != perms["nd2nz"]).sum())
        print(f"ALIGN_IDENTITY: {align_is_identity}")
        print(f"ALIGN_EQ_ND2NZ: {equal} (diff slots={diff}/{BN * BK})")
        if equal:
            print("H27 READOUT: 布局对齐假设成立,h26 失败需另查(burst/stride 语义)")
        elif align_is_identity:
            print("H27 READOUT: align=恒等平载;packer 修正=按 nd2nz 实测物理序重排 GM flat")
        else:
            print("nd2nz 前 16 槽映射:", perms["nd2nz"][:16].tolist())
            print("align 前 16 槽映射:", perms["align"][:16].tolist())
    print("PROBE:", "OK" if ok else "DECODE_FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
