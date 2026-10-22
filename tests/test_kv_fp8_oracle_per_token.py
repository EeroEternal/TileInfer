"""L-A2 KV FP8 per-token scale 翻案 oracle(iter-252,H88,D147 用户授权线)。

判据(与 H82/H84 同口径,TOL 不降):per-(head, token) amax 校准 E4M3 round-trip,
fp32 sdpa 相对 Frobenius 误差 <= 3e-2。预登记红线:任何场景超差 = L-A 线**永久关闭**
(含一切 scale 粒度变体,不再留翻案,任务书 L-A2 条款)。

判决(iter-252,E381):uniform 3.816e-2/3.907e-2 > TOL → FAIL;
与 per-head(E341: 3.80/4.05e-2)逐位同水平 ⇒ E353「i.i.d. 粒度不变性」机理
在 token 轴再证:per-token 行 amax 与整头 amax 同分布,scale 粒度不改变 E4M3
三位尾数的舍入噪声地板(raw round-trip 2.611e-2 ≈ per-head 2.642e-2)。
本文件为终局判停存证:uniform 断言超差事实本身,若失效提示口径漂移需人工复核。
"""

import math

import pytest
import torch

TOL_REL = 3e-2
E4M3_MAX = 448.0
HQ, HKV, D = 16, 2, 256


def _quant_e4m3_per_token(x: torch.Tensor) -> tuple:
    """per-(head, token) amax 校准 E4M3 round-trip。x: (H, S, D)。"""
    amax = x.abs().amax(dim=-1).clamp(min=1e-12)  # (H, S)
    scale = amax / E4M3_MAX
    q = (x / scale[..., None]).to(torch.float8_e4m3fn).to(torch.float32)
    return q * scale[..., None], scale


def _sdpa_fp32(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    logits = torch.einsum("hd,hsd->hs", q, k) / math.sqrt(D)
    logits = logits - logits.amax(dim=-1, keepdim=True)
    p = torch.softmax(logits, dim=-1)
    return torch.einsum("hs,hsd->hd", p, v), p


def _make_inputs(s: int, mode: str, seed: int) -> tuple:
    g = torch.Generator().manual_seed(seed)
    k = torch.randn(HKV, s, D, generator=g)
    v = torch.randn(HKV, s, D, generator=g)
    if mode == "correlated":
        q = torch.randn(HKV, D, generator=g) * 0.1
        idx = torch.randint(0, s, (HKV,), generator=g)
        for h in range(HKV):
            k[h, idx[h]] += q[h] * 8.0
            q[h] = q[h] + k[h, idx[h]] * 0.5
        q = q.repeat(HQ // HKV, 1)
    else:
        q = torch.randn(HQ, D, generator=g)
    return q, k, v


@pytest.mark.parametrize(
    "mode,s,expect_pass",
    [
        # H88 终局判停存证(iter-252,E381):per-token scale 与 per-head 数值同水平,
        # uniform 域超差事实存档(断言超差本身)。L-A 线永久关闭,无后续翻案路径。
        ("uniform", 8192, False),
        ("uniform", 16384, False),
        ("correlated", 8192, True),
        ("correlated", 16384, True),
    ],
)
def test_kv_fp8_per_token_oracle_rel_tol(s, mode, expect_pass):
    q, k, v = _make_inputs(s, mode, seed=42 + s)
    k8, _ = _quant_e4m3_per_token(k)
    v8, _ = _quant_e4m3_per_token(v)
    rep = HQ // HKV
    o32, _ = _sdpa_fp32(q, k.repeat_interleave(rep, 0), v.repeat_interleave(rep, 0))
    o8, _ = _sdpa_fp32(q, k8.repeat_interleave(rep, 0), v8.repeat_interleave(rep, 0))
    rel = (o8 - o32).norm() / o32.norm()
    per_q = ((o8 - o32).norm(dim=-1) / o32.norm(dim=-1).clamp(min=1e-12)).max()
    print(f"[kv_fp8_per_token] S={s} {mode} rel_F={rel:.3e} per_q_max={per_q:.3e}")
    if expect_pass:
        assert rel.item() <= TOL_REL
    else:
        assert rel.item() > TOL_REL, (
            "L-A2 终局判停存证失效: per-token uniform 场景已回落 TOL 内,"
            "与 E381 判决矛盾,须人工复核口径"
        )


def test_kv_fp8_per_token_roundtrip_floor():
    """per-token round-trip 地板与 per-head 同水平(E4M3 尾数决定,粒度无关)。"""
    g = torch.Generator().manual_seed(7)
    x = torch.randn(HKV, 4096, D, generator=g)
    x8, _ = _quant_e4m3_per_token(x)
    rel = (x8 - x).norm() / x.norm()
    print(f"[kv_fp8_per_token] raw roundtrip rel={rel:.3e}")
    assert 2e-2 < rel.item() < 3e-2
