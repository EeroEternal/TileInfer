"""H26 离线(无 NPU)探针:GM→L1 免 nd2nz 直载(``asc_copy_gm2l1_align``)能否经
TileLang Python 层原生表达。

背景(E36/E68/D61):
- tilelang 的 GM→L1 代码生成硬编码只发 ``asc_copy_gm2l1_nd2nz/dn2nz``
  (codegen_ascend.cc: ``EmitGmToL1Copy_``),nd2nz 转换带宽 ~50GB/s/核是 M-MoE
  0.813x 的根因;
- CANN c_api 已有免转换的 ``asc_copy_gm2l1_align``(cube_datamove.h:1084),
  DeepGEMM-Ascend 的 ``copy_gm_to_l1<kSrcIsNZ=true>`` 即用它直载 GM 内预 NZ 权重;
- UB→L1 的 raw 旁路(``asc_copy_ub2l1``,EmitNd2NzPostCopy_)证明 L1 物理序 =
  compact-NZ 序(bf16 C0=16,列组 g 占 N*32 字节连续,列组间连续),
  因此「GM 侧按 compact-NZ 预打包 + plain burst 直载」字节级成立。

本探针不碰设备,只做离线 lower,验证三件事:
1. ``call_extern`` 原始调用能穿过 tilelang 前端与各 pass 不被改写;
2. 生成的 CCE 源含 ``asc_copy_gm2l1_align`` 且权重路径无 ``gm2l1_nd2nz``;
3. 指针实参按 ``__cbuf__/__gm__ bfloat16_t*`` 声明(与 c_api bf16 重载匹配)。

用法::

    source .agent_loop/run/env.sh
    python benchmarks/probes/h26_gm2l1_align_probe.py
"""

import importlib
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import tilelang.ascend.language as T
from tilelang.language.tir.op import call_extern
from tilelang.tools.compile_only import compile_kernel_source

N, K, M = 128, 128, 32
BURSTS = K // 16          # 列组数(bf16 C0=16)
GROUP_BYTES = N * 32      # 每列组 N 行 x 32B


def direct_kernel():
    """GM(pre-NZ compact) -> L1 raw burst;X 走常规 nd2nz(小操作数,无关紧要)。"""

    @T.prim_func
    def main(
        Wnz: T.Tensor((N, K), "bfloat16"),
        X: T.Tensor((M, K), "bfloat16"),
        Out: T.Tensor((M, N), "bfloat16"),
    ):
        with T.Kernel(1) as bx:
            w_l1 = T.alloc_l1((N, K), "bfloat16")
            a_l1 = T.alloc_l1((M, K), "bfloat16")
            a_l0 = T.alloc_l0a((M, K), "bfloat16")
            b_l0 = T.alloc_l0b((N, K), "bfloat16")
            acc = T.alloc_l0c((M, N), "float32")
            T.evaluate(call_extern("void", "asc_set_gm2l1_loop_size", 1, 1))
            T.evaluate(call_extern("void", "asc_set_gm2l1_loop1_stride", 0, 0))
            T.evaluate(call_extern("void", "asc_set_gm2l1_loop2_stride", 0, 0))
            T.evaluate(
                call_extern(
                    "void", "asc_copy_gm2l1_align",
                    T.access_ptr(w_l1, "r"), T.access_ptr(Wnz, "r"),
                    BURSTS, GROUP_BYTES,
                    0, 0, 0, 0,
                    GROUP_BYTES, GROUP_BYTES,
                )
            )
            T.copy(X[0:M, 0:K], a_l1)
            T.copy(a_l1, a_l0)
            T.copy(w_l1, b_l0)
            T.gemm(a_l0, b_l0, acc, transpose_B=True, clear_accum=True)
            T.copy(acc, Out)

    return main


def baseline_kernel():
    """现行模式:GM(ND) -> L1(经 nd2nz 转换),作指令混合对照。"""

    @T.prim_func
    def main(
        W: T.Tensor((N, K), "bfloat16"),
        X: T.Tensor((M, K), "bfloat16"),
        Out: T.Tensor((M, N), "bfloat16"),
    ):
        with T.Kernel(1) as bx:
            w_l1 = T.alloc_l1((N, K), "bfloat16")
            a_l1 = T.alloc_l1((M, K), "bfloat16")
            a_l0 = T.alloc_l0a((M, K), "bfloat16")
            b_l0 = T.alloc_l0b((N, K), "bfloat16")
            acc = T.alloc_l0c((M, N), "float32")
            T.copy(W[0:N, 0:K], w_l1)
            T.copy(X[0:M, 0:K], a_l1)
            T.copy(a_l1, a_l0)
            T.copy(w_l1, b_l0)
            T.gemm(a_l0, b_l0, acc, transpose_B=True, clear_accum=True)
            T.copy(acc, Out)

    return main


def _report(label, fn):
    try:
        src = compile_kernel_source(fn(), target="ascend")
    except Exception as exc:  # noqa: BLE001
        print(f"[{label}] LOWER FAILED: {type(exc).__name__}: {str(exc)[:400]}")
        return None
    align = len(re.findall(r"asc_copy_gm2l1_align", src))
    nd2nz = len(re.findall(r"asc_copy_gm2l1_nd2nz", src))
    dn2nz = len(re.findall(r"asc_copy_gm2l1_dn2nz", src))
    print(f"[{label}] bytes={len(src)} gm2l1_align={align} gm2l1_nd2nz={nd2nz} gm2l1_dn2nz={dn2nz}")
    for line in src.splitlines():
        if "asc_copy_gm2l1" in line:
            print("      ", line.strip()[:200])
    for pat in (r"__cbuf__\s+\w+\s*\*\s*\w+", r"__gm__\s+\w+\s*\*\s*\w+"):
        m = re.search(pat, src)
        if m:
            print(f"       decl: {m.group(0)}")
    return src


def main():
    print("=== baseline: GM(ND) -> L1(nd2nz) ===")
    _report("baseline", baseline_kernel)
    print("=== H26: GM(pre-NZ) -> L1(asc_copy_gm2l1_align raw) ===")
    src = _report("h26_direct", direct_kernel)
    if src is None:
        return 1
    ok = "asc_copy_gm2l1_align" in src and "Wnz" in src
    print("LOWER:", "OK" if ok else "MISSING-PTR")
    if not ok:
        return 1

    import tilelang as _tl

    tpl = Path(_tl.__file__).resolve().parent
    dump = Path("/tmp/opencode/h26_direct.asc")
    dump.write_text(src.replace(
        "#include <tl_templates/",
        f"#include <{tpl}/src/tl_templates/",
    ))
    from tilelang.contrib import bisheng

    try:
        blob = bisheng.compile_ascend(
            dump.read_text(), target_format="o",
            options=[f"-I{tpl / 'src'}"],
        )
        print(f"BISHENG_COMPILE: OK bytes={len(blob)}")
    except Exception as exc:  # noqa: BLE001
        print(f"BISHENG_COMPILE: FAILED {type(exc).__name__}: {str(exc)[:600]}")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
