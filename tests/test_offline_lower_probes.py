"""CPU 契约测试:T4/T5 的离线 lower 机制探针(与 ``benchmarks/probes/`` 同源)。

T4(PM-2,full M tile)与 T5(SIMD softmax)都属"机制是否可表达"一类,结论读自生成的 CCE,
无需设备。本测试把这两个探针的核心断言固化,防止后端回归导致探针失效。

* **T4 更正**:``next-tasks.md`` 曾把 full M tile 列为未做,实际 commit ``5730b66``(PM-2)
  已用 ``RowThr`` 实现(``block_q // 2`` workaround 已移除,见 docs/known-issues.md)。
  本测试断言内核里**不再**存在 ``block_q // 2`` 的半区规划痕迹,并保留 RowThr 契约。
* **T5**:SIMT softmax 降到标量线程循环(``scalar_for`` 多、无 ``simd_inst::``),SIMD softmax
  降到 ``vlds_norm/vcmax/vsub/vexp/vcgmax/vmul/vsts_norm`` 向量内在函数。

CPU 机器上后端仍可 lower(不跑设备),故本测试不标 npu;但若后端/工具链缺失则跳过。
"""

from __future__ import annotations

import pytest

# 后端可 lower(仅编译期,不上设备)才跑;缺工具链自动跳过。
try:  # pragma: no cover
    import tilelang.ascend.language as _T  # noqa: F401
    from tilelang.tools.compile_only import compile_kernel_source  # noqa: F401

    _HAVE_LOWER = True
except Exception:  # pragma: no cover
    _HAVE_LOWER = False

pytestmark = pytest.mark.skipif(not _HAVE_LOWER, reason="tilelang ascend lower 不可用")


def test_simd_softmax_lowers_to_vector_intrinsics():
    """T5:SIMD softmax 必须降到向量内在函数(证机制可表达)。"""
    from benchmarks.probes.simd_softmax_lower import simd_softmax_kernel
    from tilelang.tools.compile_only import compile_kernel_source

    src = compile_kernel_source(simd_softmax_kernel(), target="ascend")
    for op in ("vlds_norm", "vcmax", "vsub", "vexp", "vcgmax", "vmul", "vsts_norm"):
        assert op in src, f"SIMD softmax 未降到 {op}"


def test_simt_softmax_lowers_to_scalar_loops():
    """T5 对照:SIMT softmax 降到标量线程循环(无向量内在函数)。"""
    from benchmarks.probes.simd_softmax_lower import simt_softmax_kernel
    from tilelang.tools.compile_only import compile_kernel_source

    src = compile_kernel_source(simt_softmax_kernel(), target="ascend")
    assert "simd_inst::" not in src, "SIMT 路径不应出现向量内在函数"
    assert "for (" in src, "SIMT 路径应有标量线程循环"


def test_simd_device_softmax_lowers():
    """T5 设备探针的 SIMD softmax 必须在 CPU 上也能 lower(锁住内在函数选择,防回归)。

    设备数值等价由 ``benchmarks/probes/simd_softmax_device.py`` 在 NPU 上验证;这里只保证
    lower 不退化。含两档宽度:BC=64(单寄存器)与 BC=128(真实 page_size,跨寄存器归约合并
    ``vmax``/``vadd``——这是 T5 集成到 decode 内核的关键形状)。
    """
    from benchmarks.probes.simd_softmax_device import _simd_softmax_64, _simd_softmax_128
    from tilelang.tools.compile_only import compile_kernel_source

    base = ("vlds_norm", "vcmax", "vdupv", "vsub", "vexp", "vcadd", "vdiv", "vsts_norm")
    src64 = compile_kernel_source(_simd_softmax_64(32), target="ascend")
    for op in base:
        assert op in src64, f"BC=64 SIMD softmax 未降到 {op}"

    src128 = compile_kernel_source(_simd_softmax_128(32), target="ascend")
    for op in (*base, "vmax", "vadd"):  # 跨寄存器归约合并
        assert op in src128, f"BC=128 SIMD softmax 未降到 {op}"


def test_simd_paged_softmax_lowers():
    """T5 分页 masked SIMD softmax 必须 lower(锁住集成蓝本的算子集,防回归)。

    设备数值等价由 ``benchmarks/probes/simd_softmax_paged_device.py`` 验证;这里保证
    masking(`vci` lane 索引 + `vcmps` int 比较 + `vsel`)与运行期页循环可 lower——
    这是 decode 内核集成的关键(避开 f32 select/sitofp 两个后端限制)。
    """
    from benchmarks.probes.simd_softmax_paged_device import paged_softmax_simd
    from tilelang.tools.compile_only import compile_kernel_source

    src = compile_kernel_source(paged_softmax_simd(16, 128, 3), target="ascend")
    for op in ("vci", "vcmps", "vsel", "vcmax", "vmax", "vcadd", "vadd", "vexp", "vdiv",
               "vdupv", "vlds_norm", "vsts_norm"):
        assert op in src, f"T5 分页 SIMD softmax 未降到 {op}"



def test_prefill_uses_full_m_tile():
    """T4(PM-2,5730b66):prefill 内核用 full block_q tile,无 `block_q // 2` 半区规划。

    只检查**代码体**(剥掉 docstring):``ROWS = BR // 2`` 是 `dual_copy` 的 AIV 半分约定
    (常量 BR//2,与 block_q 无关),docstring 里对旧 workaround 的*解释*文字不算规划痕迹。
    """
    import inspect
    import re

    from tileinfer.kernels.attention import paged_decode_ascend950 as k

    src = inspect.getsource(k.paged_prefill_ascend950)
    # 去掉 docstring(首个三引号块),只看真实代码。
    body = re.sub(r'""".*?"""', "", src, count=1, flags=re.S)
    assert "BR = block_q" in body, "full M tile:BR 必须等于完整 block_q"
    assert "block_q // 2" not in body, "PM-2 已移除 block_q//2 半区规划"
    assert "RowThr" in body, "RowThr(每行因果阈值)是 full-M-tile 方案的契约"

