"""L-A 翻案 MX 粒度 KV FP8 CPU 数值域 oracle(iter-242,H84 门槛先行)。

判据(预登记,TOL 不降):KV 以 E4M3 + per-32 连续元素共享 2 的幂 scale(E8M0,
MX 标准)round-trip 量化后,fp32 sdpa 输出相对 Frobenius 误差
rel = ||O8 - O32||_F / ||O32||_F <= 3e-2,
场景=GQA 16Q/2KV、head_dim 256、S in {8192, 16384}、decode q_len=1、
相关分布(尖锐 attention)+均匀分布(对照组)。任何场景超差 = L-A 终局判停
(per-head 与 per-block 双粒度均不可行 ⇒ 「KV FP8 在 TOL 3e-2 下不可达」)。

与 test_kv_fp8_oracle.py(H82 判停存证)的关系:H82 的 uniform 亮停存证断言
per-head 粒度超差事实;本文件验证 per-block 粒度是否使误差回落 TOL 内——
若回落,H82 存证按其设计保持红(提示重开),L-A 以本文件为准恢复推进。

块布局对齐硬件 MX:量化沿存储连续 32 元素分块(K 行内 D/8 块;V 同布局)。
scale 为 2 的幂(E8M0),amax/scale ∈ (224, 448],保证 E4M3 正常域且最大化利用尾数。
"""

import math

import pytest
import torch

TOL_REL = 3e-2
E4M3_MAX = 448.0  # float8_e4m3fn 最大正常值
MX_BLOCK = 32  # MX 标准:每 32 连续元素共享 scale
HQ, HKV, D = 16, 2, 256  # Qwen3.6-35B-A3B GQA 形状


def _quant_e4m3_mx(x: torch.Tensor) -> torch.Tensor:
    """per-32 元素块 E8M0 scale E4M3 round-trip。x: (..., R, C) -> (..., R, C)。

    沿最后一维按 MX_BLOCK 分块(要求 C % MX_BLOCK == 0);scale 取 2 的幂,
    使块 amax 归一化到 (224, 448](E4M3 正常域上半区,最大化尾数利用)。
    """
    r, c = x.shape[-2], x.shape[-1]
    assert c % MX_BLOCK == 0
    xb = x.reshape(*x.shape[:-2], r * (c // MX_BLOCK), MX_BLOCK)
    amax = xb.abs().amax(dim=-1).clamp(min=1e-30)
    # E8M0:scale = 2^e,使 amax/scale ∈ (224, 448]
    e = torch.ceil(torch.log2(amax / (E4M3_MAX / 2)))
    scale = torch.pow(2.0, e)
    q = (xb / scale.unsqueeze(-1)).to(torch.float8_e4m3fn).to(torch.float32)
    return (q * scale.unsqueeze(-1)).reshape(*x.shape)


def _sdpa_fp32(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """decode 单 query:q (H, D),k/v (H, S, D) -> (H, D)。fp32 数值稳定 softmax。"""
    logits = torch.einsum("hd,hsd->hs", q, k) / math.sqrt(D)
    logits = logits - logits.amax(dim=-1, keepdim=True)
    p = torch.softmax(logits, dim=-1)
    return torch.einsum("hs,hsd->hd", p, v)


def _make_inputs(s: int, mode: str, seed: int) -> tuple:
    g = torch.Generator().manual_seed(seed)
    k = torch.randn(HKV, s, D, generator=g)
    v = torch.randn(HKV, s, D, generator=g)
    if mode == "correlated":  # 尖锐 attention:query 贴近少量 token
        q = torch.randn(HKV, D, generator=g) * 0.1
        idx = torch.randint(0, s, (HKV,), generator=g)
        for h in range(HKV):
            k[h, idx[h]] += q[h] * 8.0
            q[h] = q[h] + k[h, idx[h]] * 0.5
        q = q.repeat(HQ // HKV, 1)
    else:  # uniform 对照
        q = torch.randn(HQ, D, generator=g)
    return q, k, v


@pytest.mark.parametrize(
    "mode,s,expect_pass",
    [
        ("uniform", 8192, False),
        ("uniform", 16384, False),
        ("correlated", 8192, True),
        ("correlated", 16384, True),
    ],
)
def test_kv_fp8_mx_oracle_rel_tol(mode, s, expect_pass):
    q, k, v = _make_inputs(s, mode, seed=42 + s)
    k8 = _quant_e4m3_mx(k)
    v8 = _quant_e4m3_mx(v)
    rep = HQ // HKV
    o32 = _sdpa_fp32(q, k.repeat_interleave(rep, 0), v.repeat_interleave(rep, 0))
    o8 = _sdpa_fp32(q, k8.repeat_interleave(rep, 0), v8.repeat_interleave(rep, 0))
    rel = (o8 - o32).norm() / o32.norm()
    per_q = ((o8 - o32).norm(dim=-1) / o32.norm(dim=-1).clamp(min=1e-12)).max()
    print(f"[kv_fp8_mx_oracle] S={s} {mode} rel_F={rel:.3e} per_q_max={per_q:.3e}")
    if expect_pass:
        assert rel.item() <= TOL_REL
    else:
        # 亮停存证(iter-243 E353 终局判决):MX per-block 粒度在均匀域同样超差
        # ⇒ per-head/per-block 双粒度均不可行,L-A 终局关闭。此断言守卫该事实:
        # 若未来 uniform 回落 TOL 内(如混合精度/更高尾数格式),此处失败即为
        # L-A 重开评估信号。
        assert rel.item() > TOL_REL, "亮停存证失效: uniform 已回落 TOL 内,重开 L-A 评估"


def test_kv_fp8_mx_roundtrip_fidelity():
    """MX 块粒度量化保真度基线:raw round-trip 相对误差应显著小于 TOL。"""
    g = torch.Generator().manual_seed(7)
    x = torch.randn(HKV, 4096, D, generator=g)
    x8 = _quant_e4m3_mx(x)
    rel = (x8 - x).norm() / x.norm()
    print(f"[kv_fp8_mx_oracle] raw roundtrip rel={rel:.3e}")
    assert rel.item() < TOL_REL


def test_kv_fp8_mx_vs_per_head_granularity_invariance():
    """粒度不变性存证(iter-243 E353):i.i.d. 分布下 per-block MX 与 per-head
    单 scale 的 raw round-trip 误差相等(±5% 口径)——每 32 元素块的 amax 与
    整头 amax 同分布,scale 粒度不改变尾数舍入噪声地板。该地板由 E4M3 三位
    尾数决定(~2.4e-2 量级),任何 scale 粒度细化均无法突破 ⇒ L-A 终局判停的
    理论依据(uniform 域 3.8e-2 超差不可通过细化粒度修复)。
    """
    from test_kv_fp8_oracle import _quant_e4m3_per_head

    g = torch.Generator().manual_seed(7)
    x = torch.randn(HKV, 4096, D, generator=g)
    rel_mx = ((_quant_e4m3_mx(x) - x).norm() / x.norm()).item()
    xh, _ = _quant_e4m3_per_head(x)
    rel_ph = ((xh - x).norm() / x.norm()).item()
    print(f"[kv_fp8_mx_oracle] roundtrip mx={rel_mx:.3e} per_head={rel_ph:.3e}")
    assert abs(rel_mx - rel_ph) / rel_ph < 0.05
