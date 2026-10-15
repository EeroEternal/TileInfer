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


def test_simd_online_softmax_lowers():
    """T5 逐页 online SIMD softmax 核心(= decode 内核形态)必须 lower。

    设备数值等价由 ``benchmarks/probes/simd_softmax_online_device.py`` 验证;这里锁住
    running (m,l) 的两处跨页累积(vdupv 广播 + vadd/vmul/vcadd)可 lower。
    """
    from benchmarks.probes.simd_softmax_online_device import perpage_online_simd
    from tilelang.tools.compile_only import compile_kernel_source

    src = compile_kernel_source(perpage_online_simd(16, 128, 8), target="ascend")
    for op in ("vlds_norm", "vcmax", "vmax", "vsub", "vexp", "vcadd", "vadd", "vmul",
               "vdupv", "vsts_norm"):
        assert op in src, f"T5 online SIMD softmax 未降到 {op}"


def test_decode_simd_softmax_gate():
    """T5 集成开关：``TILEINFER_SIMD_SOFTMAX`` 让 decode 内核在 SIMT/SIMD softmax 间切换。

    默认关闭时内核不含 ``vcmax``(与改造前一致);置 1 时含 SIMD 归约内在函数。两路径都必须在
    CPU 上可 lower。设备等价由 ``tests/test_tilelang_ascend950_decode.py``(off/on 各 25 用例)
    与 ``benchmarks/probes/simd_softmax_*_device.py`` 共同保证。
    """
    import os

    from tilelang.tools.compile_only import compile_kernel_source
    import tileinfer.kernels.attention.paged_decode_ascend950 as K

    spec = dict(batch=1, kv_heads=2, group=8, dim=128, page_size=128,
                num_pages_cap=16, threads=128)
    saved = os.environ.pop("TILEINFER_SIMD_SOFTMAX", None)
    try:
        assert K._simd_softmax_enabled() is False
        src_off = compile_kernel_source(K.paged_decode_ascend950(**spec), target="ascend")
        assert "vcmax" not in src_off, "默认路径不应含 SIMD 归约内在函数"
        sspec = dict(batch=3, kv_heads=2, group=8, dim=128, page_size=128, num_pages_cap=16,
                     num_tiles=4, max_splits=2, threads=128)
        s_off = compile_kernel_source(K.paged_decode_split(**sspec), target="ascend")
        assert "vcmax" not in s_off, "split 默认路径不应含 SIMD 归约内在函数"
        pspec = dict(batch=1, kv_heads=2, group=8, dim=128, page_size=128, num_pages_cap=16,
                     num_tiles=4, threads=128)
        sspec_prefill = dict(batch=1, kv_heads=2, group=8, dim=128, page_size=128,
                             num_pages_cap=16, num_tiles=4, max_splits=2, threads=128)
        p_off = compile_kernel_source(K.paged_prefill_ascend950(**pspec), target="ascend")
        assert "vcmax" not in p_off, "prefill 默认路径不应含 SIMD 归约内在函数"

        os.environ["TILEINFER_SIMD_SOFTMAX"] = "1"
        assert K._simd_softmax_enabled() is True
        src_on = compile_kernel_source(K.paged_decode_ascend950(**spec), target="ascend")
        for op in ("vci", "vcmps", "vsel", "vcmax", "vmax", "vcadd", "vadd", "vexp"):
            assert op in src_on, f"T5 decode SIMD 路径未降到 {op}"

        # split kernel: same gate + same SIMD softmax (per-tile online over `plen` pages).
        s_on = compile_kernel_source(K.paged_decode_split(**sspec), target="ascend")
        for op in ("vcmps", "vcmax", "vcadd", "vexp"):
            assert op in s_on, f"T5 split SIMD 路径未降到 {op}"

        # prefill main kernel: causal + tail mask via two chained vsel (vcmp on kv_pos).
        p_on = compile_kernel_source(K.paged_prefill_ascend950(**pspec), target="ascend")
        for op in ("vcmps", "vcmp", "vcmax", "vcadd", "vexp", "vadds"):
            assert op in p_on, f"T5 prefill SIMD 路径未降到 {op}"

        # prefill split / head / frontier: same gate, same SIMD softmax.  split + frontier carry the
        # causal mask; head carries only the scalar tail mask.
        for name, sp in (
            ("prefill_split", K.paged_prefill_split_ascend950),
            ("prefill_head", K.paged_prefill_split_head_ascend950),
            ("prefill_frontier", K.paged_prefill_split_frontier_ascend950),
        ):
            on = compile_kernel_source(sp(**sspec_prefill), target="ascend")
            for op in ("vcmps", "vcmax", "vcadd", "vexp"):
                assert op in on, f"T5 {name} SIMD 路径未降到 {op}"
    finally:
        os.environ.pop("TILEINFER_SIMD_SOFTMAX", None)
        if saved is not None:
            os.environ["TILEINFER_SIMD_SOFTMAX"] = saved


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

