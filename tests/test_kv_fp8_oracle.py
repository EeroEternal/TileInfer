"""L-A KV FP8 CPU 数值域 oracle(iter-238,H82 门槛先行)。

判据(预登记,TOL 不降):E4M3 per-(layer,head) amax 校准 round-trip 量化 K/V 后,
fp32 sdpa 输出相对 Frobenius 误差 rel = ||O8 - O32||_F / ||O32||_F <= 3e-2,
场景=GQA 16Q/2KV、head_dim 256、S in {8192, 16384}、decode q_len=1、
相关分布(尖锐 attention)+均匀分布(对照组)。任何场景超差 = L-A 停。

事后 amax 校准(上帝视角)是量化精度的乐观上界;上界超差即无需再做在线校准设计。
本文件同时是判据存档:数值结论回写 docs/performance.md(设备域验证完成后)。
"""

import math

import pytest
import torch

TOL_REL = 3e-2
E4M3_MAX = 448.0  # float8_e4m3fn 最大正常值
HQ, HKV, D = 16, 2, 256  # Qwen3.6-35B-A3B GQA 形状


def _quant_e4m3_per_head(x: torch.Tensor) -> tuple:
    """per-head amax 校准 E4M3 round-trip。x: (H, S, D) -> (反量化, scale(H))"""
    amax = x.abs().amax(dim=(1, 2)).clamp(min=1e-12)  # (H,)
    scale = amax / E4M3_MAX
    q = (x / scale[:, None, None]).to(torch.float8_e4m3fn).to(torch.float32)
    return q * scale[:, None, None], scale


def _sdpa_fp32(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """decode 单 query:q (H, D),k/v (H, S, D) -> (H, D)。fp32 数值稳定 softmax。"""
    logits = torch.einsum("hd,hsd->hs", q, k) / math.sqrt(D)
    logits = logits - logits.amax(dim=-1, keepdim=True)
    p = torch.softmax(logits, dim=-1)
    return torch.einsum("hs,hsd->hd", p, v), p


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
        # H82 判停存档(iter-238,E341):per-(layer,head) amax 校准 E4M3 的乐观上界
        # 在均匀分布域超差(3.80e-2/4.05e-2 > TOL 3e-2),尖锐域贴线(2.62/2.66e-2,
        # per-query max 3.29/3.65e-2 已超单元素口径)。按预登记「任何场景超差即停、
        # 禁止降 TOL」→ L-A 首方案判停(disproven)。uniform 两例为亮停存证(断言
        # 超差事实本身):若未来翻案(per-token scale 等)使其回落线内,此存证会红,
        # 提示重开 L-A 评估。翻案路径见 HYPOTHESES H82 关闭记录。
        ("uniform", 8192, False),
        ("uniform", 16384, False),
        ("correlated", 8192, True),
        ("correlated", 16384, True),
    ],
)
def test_kv_fp8_oracle_rel_tol(s, mode, expect_pass):
    q, k, v = _make_inputs(s, mode, seed=42 + s)
    k8, _ = _quant_e4m3_per_head(k)
    v8, _ = _quant_e4m3_per_head(v)
    # GQA 展开 K/V 到 16 头后逐头计算
    rep = HQ // HKV
    o32, _ = _sdpa_fp32(q, k.repeat_interleave(rep, 0), v.repeat_interleave(rep, 0))
    o8, p8 = _sdpa_fp32(q, k8.repeat_interleave(rep, 0), v8.repeat_interleave(rep, 0))
    rel = (o8 - o32).norm() / o32.norm()
    per_q = ((o8 - o32).norm(dim=-1) / o32.norm(dim=-1).clamp(min=1e-12)).max()
    print(f"[kv_fp8_oracle] S={s} {mode} rel_F={rel:.3e} per_q_max={per_q:.3e}")
    if expect_pass:
        assert rel.item() <= TOL_REL
    else:
        assert rel.item() > TOL_REL, "亮停存证失效: uniform 场景已回落到 TOL 内,重开 L-A 评估"


def test_kv_fp8_quant_roundtrip_fidelity():
    """per-head scale 校准的量化保真度基线:round-trip 相对误差应显著小于 TOL。"""
    g = torch.Generator().manual_seed(7)
    x = torch.randn(HKV, 4096, D, generator=g)
    x8, scale = _quant_e4m3_per_head(x)
    rel = (x8 - x).norm() / x.norm()
    print(f"[kv_fp8_oracle] raw roundtrip rel={rel:.3e} scale={scale.tolist()}")
    assert rel.item() < TOL_REL
