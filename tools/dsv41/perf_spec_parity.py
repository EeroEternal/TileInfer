"""投机解码的 CPU 对拍：`投机生成` vs `逐 token 贪心生成`（不需要 NPU）。

为什么要有这个脚本（`docs/notes/dsv41-perf-plan.md` §21 的离线门）：

* 投机的**唯一**允许的效果是"更少的前向次数"，输出必须与逐 token 贪心**逐 token 相同**；
* 上车（NPU / vLLM）之前必须先在 CPU 上把这条性质证明掉，否则一旦上机出错很难判断是
  分块解码、逐位置 logits 还是接受逻辑的问题。

复用 `tests/test_dsv41_block_parity.py` 的脚手架：tiny 配置 + 官方 `inference/` 作为权重来源
（C4：官方代码只读、仅作 oracle）。

用法：
    .venv/bin/python tools/dsv41/perf_spec_parity.py [--max-new 8] [--k 4] [--n 3]

输出：
    plain  : 纯贪心生成的 token 序列与步数
    spec   : 投机生成的 token 序列与步数、接受率
    断言   : 两个序列必须逐 token 相同；否则非零退出
"""

from __future__ import annotations

import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-new", type=int, default=8)
    ap.add_argument("--k", type=int, default=4, help="每次提议的候选数")
    ap.add_argument("--n", type=int, default=3, help="n-gram 的 n")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    import torch

    from tileinfer.models.dsv41.block import ModelState, logits_all_positions, transformer_forward
    from tileinfer.models.dsv41.spec_decode import propose_ngram
    from tileinfer.models.dsv41.spec_session import acceptance_rate, run_speculative

    # 复用测试里的 tiny 配置与 oracle 权重脚手架（它在 CPU 上就能跑）
    from tests.test_dsv41_block_parity import tiny_config, _setup, _pair_blocks

    cfg = tiny_config()
    official = _setup()
    _args, pairs = _pair_blocks(official, cfg, [0])
    layer_id, _ref, _ours = pairs[0]

    # 用我们的 block 拼一个"单层模型"：只验证投机/贪心的等价性，不需要 40 层
    from tileinfer.models.dsv41.block import Dsv41Model  # noqa: F401  (类型提示用)
    from tileinfer.models.dsv41.modules import Dsv41Block

    

    # 用真实的模型类构造（transformer_forward 需要 embed/norm/layers/head 齐全），
    # 再把 oracle 权重灌进第 layer_id 层——与 tests/test_dsv41_block_parity.py 的做法一致。
    from tileinfer.vllm.models.dsv41 import build_model

    model = build_model(cfg, device="cpu")
    model.layers[layer_id].load_state_dict(_ours.state_dict())
    model.device = "cpu"

    state = ModelState(cfg=cfg, max_seq_len=32, bsz=1, device="cpu")
    gen = torch.Generator().manual_seed(args.seed)

    def forward_k(tokens):
        """一次前向吃 k 个候选，返回逐位置 logits（[k, vocab]）。

        ``start_pos`` 由调用方通过 ``state`` 里的进度决定；这里用 state 的属性记录已生成数，
        与真实接线（`start_pos += accepted + 1`）保持同一语义。
        """
        start = int(getattr(state, "_spec_pos", 0))
        cap = {}
        tok = torch.tensor([list(tokens)], dtype=torch.long)
        _logits, _extras = transformer_forward(
            model, tok, state, start, capture=cap, moe_fn=None
        )
        h_all = cap["h_all"]
        per_pos = logits_all_positions(model, h_all)      # [1, k, vocab]
        state._spec_pos = start + len(tokens)
        return per_pos[0]

    hist = [int(x) for x in torch.randint(1, cfg.vocab_size, (4,), generator=gen).tolist()]

    plain, pstats = run_speculative(forward_k, hist, args.max_new, propose_fn=None, k=0)

    # 重置进度，重跑投机臂
    state._spec_pos = 0
    state2 = ModelState(cfg=cfg, max_seq_len=32, bsz=1, device="cpu")
    state = state2

    def forward_k2(tokens):
        start = int(getattr(state, "_spec_pos", 0))
        cap = {}
        tok = torch.tensor([list(tokens)], dtype=torch.long)
        transformer_forward(model, tok, state, start, capture=cap, moe_fn=None)
        per_pos = logits_all_positions(model, cap["h_all"])
        state._spec_pos = start + len(tokens)
        return per_pos[0]

    spec, sstats = run_speculative(
        forward_k2, hist, args.max_new,
        propose_fn=lambda p: propose_ngram(p, args.k, args.n), k=args.k,
    )

    print(f"plain: {plain}  步数={pstats['steps']}")
    print(f"spec : {spec}  步数={sstats['steps']}  提议={sstats['proposed']} "
          f"接受={sstats['accepted']} 接受率={acceptance_rate(sstats):.2f}")

    if spec != plain:
        print("FAIL：投机输出与逐 token 贪心不一致")
        return 1
    if sstats["steps"] > pstats["steps"]:
        print("FAIL：投机反而增加了前向次数")
        return 1
    print("PASS：投机与贪心逐 token 相同")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
