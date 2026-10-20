"""Fused top-p/top-k 采样内核(移植自 deepseek/DeepInfer `fused_sample.py`)。

## 这个内核做什么

vLLM 原生 Ascend 采样路径求 top-p/top-k 阈值的方式是**对整张词表排序**:

    probs = softmax(logits)          # 全词表 float32 物料化(V≈248K)
    sorted_probs, idx = probs.sort() # O(V log V)
    cumsum -> 找 top-p 截断点;第 k 大 -> top-k 阈值
    multinomial / 逆 CDF 采样

DeepInfer 完全不做排序:**用「三分位点二分搜索」直接逼近那个标量阈值**,并把
temperature / softmax / 阈值 / mask / 采样**融进一个 kernel**。

核心洞察:

1. top-p 与 top-k 都只是「按概率值切一刀」,需要的只是**阈值标量**,不是有序排列;
2. 在值域上二分:每轮取区间 `[lo, hi]` 的 1/4、1/2、3/4 三点,一次扫描同时统计
   **三组** `(概率和, token 数)` —— 一轮把区间缩到 1/4,16 轮 ≈ 4^-16;
3. 阈值附近存在**同值项**,必须控制保留几个,否则 top-p 有效质量会偏。

## 本实现的关键简化:在 exp 空间搜索,不做逐元素除法

令 `e_i = exp(x_i·inv_t − max)`,则 `p_i = e_i / total`,而

    比较 p_i >= q  等价于  e_i >= q·total

于是搜索区间固定为 `[MIN_PROB, 1.0]`,top-p 的目标质量写成 `top_p · total`,
**全程没有一次 `vdiv`**,省掉 24.8 万次除法。

## 实现约定(tilelang 0.1.15,每条都是踩出来的)

1. **SIMD 块内只用向量值**:标量状态一律放 shared,`vld(..., "BRC_B32")` 读、
   `vsts(..., "ONEPT_B32")` 写。混用报 `Incompatible types in BufferStore`。
2. **向量不能在 `T.serial` 迭代间自赋值**(`Immutable variable ... used outside
   its defining region`)。累积一律走 shared。
3. **`MODE_MERGING` 必须赋给可变 SIMD 寄存器**(`alloc_local`),不能直接作为
   `vsts` 的入参。
4. **不允许"块内读 shared 又写回同一 shared"**:编译器会把 barrier 提到分支外,
   留下竞争(`[ThreadSync] Hoisting sync out of an if ... the race remains`),
   运行时 `device error type 3 / 507035`。因此每个 SIMD 块只写 `red_ub` 中转,
   跨 chunk 的累加在随后的 `SimtVF` 块里做。
5. **int32 的 `vcadd` 归约 + `vsts` 写 int32 shared 会崩**(507035)。所有归约
   (含 token 计数)统一用 **float32**;词表 ≤ 2^24 时 float32 可精确表示每个整数。
6. **`if` 用 Python 常量**(`do_sample` / `_stage_limit`)在构建期求值;不要用
   `and` 组合多个 Python 常量后当条件,DSL 可能误当 TIR 布尔而静默跳过整块。

另:不用 `T.dynamic`(0.1.15 在 tensor 形状里 `NameError`),batch 为编译期常量;
不用 `annotate_buffer_versions`(0.1.15 无此 API)。

## 数值语义

给定 logits 与种子,输出应与「top-p/top-k 过滤后按概率做逆 CDF 抽样」分布一致。
**不保证与原生 `multinomial` 逐个随机数相同**(随机数流不同),故测试用统计口径
(大样本频率)+ 阈值一致性双重验证。

## 开关

`TILEINFER_OPS` 含 `sampling` 才安装;包络外(vocab 太小 / 需要 logprobs /
非 float32 logits)逐调用回退原生实现。任何异常都回退,绝不破坏服务。
"""

from __future__ import annotations

import tilelang
import tilelang.ascend.language as T

__all__ = [
    "fused_sample_plan",
    "fused_sample_tl",
    "MAX_SEARCH_ROUNDS",
    "MIN_VOCAB_FOR_FUSED_SAMPLE",
]

#: 词表下限:低于此值原生排序本身不贵,融合收益盖不住搜索固定开销。
MIN_VOCAB_FOR_FUSED_SAMPLE = 16384

#: 三分位二分搜索轮数(每轮缩到 1/4,16 轮 ≈ 4.3e-10 相对精度)
MAX_SEARCH_ROUNDS = 16

#: SIMD 向量长度(64 个 float32 lane)
_VL = 64

#: 概率下限,避免把阈值压到 0
_MIN_PROB = 2.0**-32

#: Ascend 950 的 vector core 数
_NUM_CORES = 40


def _block_size(vocab_size: int, cap: int) -> int:
    """选一个能整除 vocab、且不超过 `cap` 的分块大小(递减 64 搜索)。"""
    block = min(vocab_size, cap)
    block = block // _VL * _VL or _VL
    while block >= _VL:
        if vocab_size % block == 0:
            return block
        block -= _VL
    return _VL


def fused_sample_plan(vocab_size: int, batch: int, ub_cap: int = 8192) -> dict:
    """分块规划(纯 host 算术,便于单测覆盖,不需要设备)。

    `ub_cap` 默认 **8192** 而非 16384:16384 个 float32 = 64KB 会把 UB 占满,
    叠加标量缓冲后越界,运行时表现为 `device error type 3 / 507035`(实测)。
    """
    if vocab_size < _VL or vocab_size % _VL != 0:
        raise ValueError(
            f"fused_sample 要求 vocab_size 必须是 {_VL} 的正整数倍,got {vocab_size}"
        )
    if batch < 1:
        raise ValueError(f"batch 必须 >= 1,got {batch}")
    blk = _block_size(vocab_size, ub_cap)
    return {
        "blk": blk,
        "n_chunks": vocab_size // blk,
        "n_vec": blk // _VL,
        "num_cores": min(batch, _NUM_CORES),
    }


def fused_sample_tl(
    vocab_size: int,
    batch: int,
    do_sample: bool,
    ub_cap: int = 8192,
    _stage_limit: int = 99,
):
    """构建融合采样内核。

    参数
    ----
    vocab_size : 词表大小,必须是 64 的正整数倍(本模型 248320 满足)。
    batch      : 行数(请求数),编译期常量(见模块 docstring)。
    do_sample  : True = 按概率随机抽;False = 纯 argmax(greedy)。
    ub_cap     : 单个 UB 分块上限(float32 元素数)。
    _stage_limit : 诊断用,只跑到第 N 阶段(正常 99 = 全跑)。

    张量约定
    --------
    ``logits``      ``[batch, vocab]`` float32(内核内做 temperature 缩放)
    ``temperatures`` ``[batch]`` float32
    ``top_p``       ``[batch]`` float32,``<1.0`` 才启用
    ``top_k``       ``[batch]`` int32,``>0 且 <vocab`` 才启用
    ``seed``        ``[batch]`` int64
    ``out_tokens``  ``[batch]`` int32
    """
    plan = fused_sample_plan(vocab_size, batch, ub_cap)
    blk = plan["blk"]
    n_chunks = plan["n_chunks"]
    n_vec = plan["n_vec"]

    @T.prim_func
    def main(
        logits: T.Tensor((batch, vocab_size), T.float32),
        temperatures: T.Tensor((batch,), T.float32),
        top_p: T.Tensor((batch,), T.float32),
        top_k: T.Tensor((batch,), T.int32),
        seed: T.Tensor((batch,), T.int64),
        out_tokens: T.Tensor((batch,), T.int32),
    ):
        with T.Kernel(batch) as row:
            vals_ub = T.alloc_shared((blk,), T.float32)
            # [0]=inv_t [1]=row_max [2]=total [3]=keep_mass
            # [4]=run(逆CDF 已累加) [5]=best(贪心最大 e) [6]=抽样目标
            st_ub = T.alloc_shared((8,), T.float32)
            bounds_ub = T.alloc_shared((4,), T.float32)  # [0]=lo [1]=hi
            quart_ub = T.alloc_shared((4,), T.float32)  # 三个分位点
            # 三个分位点对应的 (质量和, 数量);[3..5] 是本次中转,全 float
            stat_ub = T.alloc_shared((6,), T.float32)
            tgt_ub = T.alloc_shared((4,), T.float32)  # [0]=质量目标 [1]=数量目标
            pick_ub = T.alloc_shared((4,), T.int32)
            done_ub = T.alloc_shared((4,), T.int32)
            # 向量归约中转:SIMD 块**只写这里**,标量块从这里读
            red_ub = T.alloc_shared((4,), T.float32)

            # ================= 阶段 1:inv_t 与 row_max =================
            with T.SimtVF(threads=1):
                st_ub[0] = T.float32(1.0) / T.max(temperatures[row], T.float32(1e-6))
                st_ub[1] = T.float32(-1e38)
            for c in T.serial(n_chunks):
                T.copy(logits[row, c * blk : (c + 1) * blk], vals_ub)
                with T.SimdVF():
                    full = T.simd.pset(32)
                    one = T.simd.pset(32, "PAT_VL1")
                    inv = T.simd.vld(st_ub[0], "BRC_B32")
                    acc = T.simd.alloc_local((1,), T.float32)
                    acc[0] = T.simd.vdup(-1e38, T.float32, full)
                    for g in T.serial(n_vec):
                        acc[0] = T.simd.vmax(
                            acc[0],
                            T.simd.vcmax(T.simd.vmul(T.simd.vld(vals_ub[g * _VL]), inv), full),
                            full,
                        )
                    T.simd.vsts(red_ub[0], acc[0], one, "ONEPT_B32")
                with T.SimtVF(threads=1):
                    st_ub[1] = T.max(st_ub[1], red_ub[0])

            # ================= 阶段 2:total = Σ exp(x·inv_t − max) =================
            with T.SimtVF(threads=1):
                st_ub[2] = T.float32(0.0)
            for c in T.serial(n_chunks):
                T.copy(logits[row, c * blk : (c + 1) * blk], vals_ub)
                with T.SimdVF():
                    full = T.simd.pset(32)
                    one = T.simd.pset(32, "PAT_VL1")
                    inv = T.simd.vld(st_ub[0], "BRC_B32")
                    mx = T.simd.vld(st_ub[1], "BRC_B32")
                    acc = T.simd.alloc_local((1,), T.float32)
                    acc[0] = T.simd.vdup(0.0, T.float32, full)
                    for g in T.serial(n_vec):
                        acc[0] = T.simd.vadd(
                            acc[0],
                            T.simd.vexp(
                                T.simd.vsub(T.simd.vmul(T.simd.vld(vals_ub[g * _VL]), inv), mx)
                            ),
                            full,
                        )
                    T.simd.vsts(red_ub[0], acc[0], one, "ONEPT_B32")
                with T.SimtVF(threads=1):
                    st_ub[2] = st_ub[2] + red_ub[0]

            # ================= 阶段 3:三分位二分搜索阈值(exp 空间) =================
            if _stage_limit >= 3:
                with T.SimtVF(threads=1):
                    # top-p 目标质量(exp 空间)、top-k 目标数量
                    tgt_ub[0] = T.max(T.float32(_MIN_PROB), top_p[row]) * st_ub[2]
                    tgt_ub[1] = T.cast(
                        T.Select(
                            T.min(top_k[row] > 0, top_k[row] < vocab_size),
                            T.max(1, top_k[row]),
                            vocab_size,
                        ),
                        T.float32,
                    )
                with T.SimdVF():
                    one = T.simd.pset(32, "PAT_VL1")
                    T.simd.vsts(
                        bounds_ub[0], T.simd.vdup(_MIN_PROB, T.float32, one), one, "ONEPT_B32"
                    )
                    T.simd.vsts(bounds_ub[1], T.simd.vdup(1.0, T.float32, one), one, "ONEPT_B32")

                for _r in T.serial(MAX_SEARCH_ROUNDS):
                    # 三个四分位点(float32 位模式上做整数插值,避免浮点除法)
                    with T.SimdVF():
                        full = T.simd.pset(32)
                        one = T.simd.pset(32, "PAT_VL1")
                        lo_i = T.reinterpret(T.simd.vld(bounds_ub[0], "BRC_B32"), T.int32x64)
                        hi_i = T.reinterpret(T.simd.vld(bounds_ub[1], "BRC_B32"), T.int32x64)
                        width = T.simd.vsub(hi_i, lo_i, full)
                        q0 = T.simd.vadd(lo_i, T.simd.vshrs(width, 2, full), full)
                        q1 = T.simd.vadd(lo_i, T.simd.vshrs(width, 1, full), full)
                        q2 = T.simd.vsub(
                            hi_i, T.simd.vshrs(T.simd.vadds(width, 3, full), 2, full), full
                        )
                        T.simd.vsts(quart_ub[0], T.reinterpret(q0, T.float32x64), one, "ONEPT_B32")
                        T.simd.vsts(quart_ub[1], T.reinterpret(q1, T.float32x64), one, "ONEPT_B32")
                        T.simd.vsts(quart_ub[2], T.reinterpret(q2, T.float32x64), one, "ONEPT_B32")

                    with T.SimtVF(threads=1):
                        stat_ub[0] = T.float32(0.0)
                        stat_ub[1] = T.float32(0.0)
                        stat_ub[2] = T.float32(0.0)

                    for c in T.serial(n_chunks):
                        T.copy(logits[row, c * blk : (c + 1) * blk], vals_ub)
                        with T.SimdVF():
                            full = T.simd.pset(32)
                            one = T.simd.pset(32, "PAT_VL1")
                            inv = T.simd.vld(st_ub[0], "BRC_B32")
                            mx = T.simd.vld(st_ub[1], "BRC_B32")
                            q0v = T.simd.vld(quart_ub[0], "BRC_B32")
                            q1v = T.simd.vld(quart_ub[1], "BRC_B32")
                            q2v = T.simd.vld(quart_ub[2], "BRC_B32")
                            a0 = T.simd.alloc_local((1,), T.float32)
                            a1 = T.simd.alloc_local((1,), T.float32)
                            a2 = T.simd.alloc_local((1,), T.float32)
                            c0 = T.simd.alloc_local((1,), T.float32)
                            c1 = T.simd.alloc_local((1,), T.float32)
                            c2 = T.simd.alloc_local((1,), T.float32)
                            a0[0] = T.simd.vdup(0.0, T.float32, full)
                            a1[0] = T.simd.vdup(0.0, T.float32, full)
                            a2[0] = T.simd.vdup(0.0, T.float32, full)
                            c0[0] = T.simd.vdup(0.0, T.float32, full)
                            c1[0] = T.simd.vdup(0.0, T.float32, full)
                            c2[0] = T.simd.vdup(0.0, T.float32, full)
                            onef = T.simd.vdup(1.0, T.float32, full)
                            for g in T.serial(n_vec):
                                e = T.simd.vexp(
                                    T.simd.vsub(
                                        T.simd.vmul(T.simd.vld(vals_ub[g * _VL]), inv), mx
                                    )
                                )
                                h0 = T.simd.vcmp(e, q0v, full, "ge")
                                h1 = T.simd.vcmp(e, q1v, full, "ge")
                                h2 = T.simd.vcmp(e, q2v, full, "ge")
                                z = T.simd.vdup(0.0, T.float32, full)
                                # 掩码置零后无条件累加(不依赖 MODE_MERGING 语义)
                                a0[0] = T.simd.vadd(a0[0], T.simd.vsel(e, z, h0), full)
                                a1[0] = T.simd.vadd(a1[0], T.simd.vsel(e, z, h1), full)
                                a2[0] = T.simd.vadd(a2[0], T.simd.vsel(e, z, h2), full)
                                c0[0] = T.simd.vadd(c0[0], T.simd.vsel(onef, z, h0), full)
                                c1[0] = T.simd.vadd(c1[0], T.simd.vsel(onef, z, h1), full)
                                c2[0] = T.simd.vadd(c2[0], T.simd.vsel(onef, z, h2), full)
                            T.simd.vsts(red_ub[0], a0[0], one, "ONEPT_B32")
                            T.simd.vsts(red_ub[1], a1[0], one, "ONEPT_B32")
                            T.simd.vsts(red_ub[2], a2[0], one, "ONEPT_B32")
                            # 数量同样用 float 归约(见 docstring 约束 5)
                            T.simd.vsts(stat_ub[3], T.simd.vcadd(c0[0], full), one, "ONEPT_B32")
                            T.simd.vsts(stat_ub[4], T.simd.vcadd(c1[0], full), one, "ONEPT_B32")
                            T.simd.vsts(stat_ub[5], T.simd.vcadd(c2[0], full), one, "ONEPT_B32")
                        with T.SimtVF(threads=1):
                            stat_ub[0] = stat_ub[0] + red_ub[0]
                            stat_ub[1] = stat_ub[1] + red_ub[1]
                            stat_ub[2] = stat_ub[2] + red_ub[2]

                    # 收窄:取"仍满足质量目标且数量目标"的最高分位点
                    with T.SimtVF(threads=1):
                        ok0 = T.min(stat_ub[0] >= tgt_ub[0], stat_ub[3] >= tgt_ub[1])
                        ok1 = T.min(stat_ub[1] >= tgt_ub[0], stat_ub[4] >= tgt_ub[1])
                        ok2 = T.min(stat_ub[2] >= tgt_ub[0], stat_ub[5] >= tgt_ub[1])
                        if ok2:
                            bounds_ub[0] = quart_ub[2]
                        elif ok1:
                            bounds_ub[0] = quart_ub[1]
                        elif ok0:
                            bounds_ub[0] = quart_ub[0]
                        else:
                            bounds_ub[1] = quart_ub[2]
            else:
                # 诊断模式跳过搜索:阈值置 0,等价"全保留"
                with T.SimtVF(threads=1):
                    bounds_ub[0] = T.float32(0.0)

            # ================= 阶段 4:选择 =================
            if _stage_limit >= 5:
                # ---- 4a:保留集质量 keep_mass = Σ e(e >= 阈值) ----
                # 采样需要它做归一化(逆 CDF 的目标 = u · keep_mass)。
                if do_sample:
                    with T.SimtVF(threads=1):
                        st_ub[3] = T.float32(0.0)
                    for c in T.serial(n_chunks):
                        T.copy(logits[row, c * blk : (c + 1) * blk], vals_ub)
                        with T.SimdVF():
                            full = T.simd.pset(32)
                            one = T.simd.pset(32, "PAT_VL1")
                            inv = T.simd.vld(st_ub[0], "BRC_B32")
                            mx = T.simd.vld(st_ub[1], "BRC_B32")
                            thr = T.simd.vld(bounds_ub[0], "BRC_B32")
                            acc = T.simd.alloc_local((1,), T.float32)
                            acc[0] = T.simd.vdup(0.0, T.float32, full)
                            for g in T.serial(n_vec):
                                e = T.simd.vexp(
                                    T.simd.vsub(
                                        T.simd.vmul(T.simd.vld(vals_ub[g * _VL]), inv), mx
                                    )
                                )
                                # 未命中置 0,再**无条件**累加(不依赖 MODE_MERGING 语义)
                                em = T.simd.vsel(
                                    e, T.simd.vdup(0.0, T.float32, full),
                                    T.simd.vcmp(e, thr, full, "ge"),
                                )
                                acc[0] = T.simd.vadd(acc[0], em, full)
                            T.simd.vsts(red_ub[0], acc[0], one, "ONEPT_B32")
                        with T.SimtVF(threads=1):
                            st_ub[3] = st_ub[3] + red_ub[0]

                if not do_sample:
                    # ---- 贪心:取最大 e,同值取最小 token id ----
                    # 方案:逐向量 SIMD 求最大,标量域只保留**严格更大**者;
                    # 向量内位置用 vci 生成 lane 索引 + vcmp 掩码 + vsel 求出。
                    # 不写 64 次标量循环(那样编译产物会膨胀,且容易踩 barrier 问题)。
                    with T.SimtVF(threads=1):
                        st_ub[5] = T.float32(-1e38)  # 全局最大 e
                        pick_ub[0] = -1  # 全局最大所在的 token id
                    for c in T.serial(n_chunks):
                        T.copy(logits[row, c * blk : (c + 1) * blk], vals_ub)
                        for g in T.serial(n_vec):
                            base = c * blk + g * _VL
                            with T.SimdVF():
                                full = T.simd.pset(32)
                                one = T.simd.pset(32, "PAT_VL1")
                                inv = T.simd.vld(st_ub[0], "BRC_B32")
                                mx = T.simd.vld(st_ub[1], "BRC_B32")
                                thr = T.simd.vld(bounds_ub[0], "BRC_B32")
                                e = T.simd.vexp(
                                    T.simd.vsub(
                                        T.simd.vmul(T.simd.vld(vals_ub[g * _VL]), inv), mx
                                    )
                                )
                                hit = T.simd.vcmp(e, thr, full, "ge")
                                em = T.simd.vsel(e, T.simd.vdup(-1e38, T.float32, full), hit)
                                T.simd.vsts(vals_ub[g * _VL], em)
                                # 向量最大
                                T.simd.vsts(
                                    red_ub[0],
                                    T.simd.vdupv(T.simd.vcmax(em, full), full, "POS_LOWEST"),
                                    one,
                                    "ONEPT_B32",
                                )
                                # 该向量内"等于向量最大"的最小 lane 索引(存 red_ub[1])
                                vmaxv = T.simd.vdupv(
                                    T.simd.vcmax(em, full), full, "POS_LOWEST"
                                )
                                eqm = T.simd.vcmp(em, vmaxv, full, "eq")
                                lanes = T.simd.vci(0, T.int32, "INC_ORDER")
                                sentinel = T.simd.vdup(_VL, T.int32, full)
                                first = T.simd.vsel(lanes, sentinel, eqm)
                                T.simd.vsts(
                                    red_ub[1],
                                    T.simd.vdupv(
                                        T.simd.vcvt(
                                            T.simd.vcmin(first, full),
                                            T.float32,
                                            full,
                                            round="ROUND_Z",
                                        ),
                                        full,
                                        "POS_LOWEST",
                                    ),
                                    one,
                                    "ONEPT_B32",
                                )
                            with T.SimtVF(threads=1):
                                better = red_ub[0] > st_ub[5]
                                st_ub[5] = T.Select(better, red_ub[0], st_ub[5])
                                pick_ub[0] = T.Select(
                                    better,
                                    c * blk + g * _VL + T.cast(red_ub[1], T.int32),
                                    pick_ub[0],
                                )
                    with T.SimtVF(threads=1):
                        pick_ub[0] = T.max(pick_ub[0], 0)
                else:
                    # ---- 采样:期望值归一化后做逆 CDF ----
                    with T.SimtVF(threads=1):
                        T.rng_init(seed[row], 0)
                        st_ub[3] = T.max(st_ub[3], T.float32(1e-30))
                        st_ub[6] = (
                            T.cast(T.rng_rand(), T.float32)
                            / T.float32(4294967296.0)
                            * st_ub[3]
                        )
                        st_ub[4] = T.float32(0.0)  # run
                        pick_ub[0] = 0
                        done_ub[0] = 0
                    # 采样选择
                    for c in T.serial(n_chunks):
                        T.copy(logits[row, c * blk : (c + 1) * blk], vals_ub)
                        for g in T.serial(n_vec):
                            # 本向量的 e(未命中置 0)与向量和
                            with T.SimdVF():
                                full = T.simd.pset(32)
                                one = T.simd.pset(32, "PAT_VL1")
                                inv = T.simd.vld(st_ub[0], "BRC_B32")
                                mx = T.simd.vld(st_ub[1], "BRC_B32")
                                thr = T.simd.vld(bounds_ub[0], "BRC_B32")
                                e = T.simd.vexp(
                                    T.simd.vsub(
                                        T.simd.vmul(T.simd.vld(vals_ub[g * _VL]), inv), mx
                                    )
                                )
                                hit = T.simd.vcmp(e, thr, full, "ge")
                                em = T.simd.vsel(e, T.simd.vdup(0.0, T.float32, full), hit)
                                T.simd.vsts(vals_ub[g * _VL], em)
                                T.simd.vsts(
                                    red_ub[0],
                                    T.simd.vdupv(T.simd.vcadd(em, full), full, "POS_LOWEST"),
                                    one,
                                    "ONEPT_B32",
                                )
                            # 无分支逆 CDF
                            with T.SimtVF(threads=1):
                                spans = st_ub[4] + red_ub[0] >= st_ub[6]
                                inside = T.min(done_ub[0] == 0, spans)
                                for lane in T.serial(_VL):
                                    idx = c * blk + g * _VL + lane
                                    proceed = T.min(inside, done_ub[0] == 0)
                                    nxt = st_ub[4] + vals_ub[g * _VL + lane]
                                    crossed = T.min(proceed, nxt >= st_ub[6])
                                    pick_ub[0] = T.Select(crossed, idx, pick_ub[0])
                                    done_ub[0] = T.Select(crossed, 1, done_ub[0])
                                    st_ub[4] = T.Select(proceed, nxt, st_ub[4])
                                # 本向量不跨目标:整段累加
                                st_ub[4] = T.Select(
                                    T.min(done_ub[0] == 0, inside == 0),
                                    st_ub[4] + red_ub[0],
                                    st_ub[4],
                                )
            else:
                # 诊断模式:输出 max 的整数值
                with T.SimtVF(threads=1):
                    pick_ub[0] = T.cast(st_ub[1], T.int32)

            with T.SimtVF(threads=1):
                out_tokens[row] = pick_ub[0]

    return main
