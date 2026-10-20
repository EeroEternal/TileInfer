"""H74 诊断:decode 每步的 host/glue 段实际成本(决定是否值得做 kernel 化)。

背景
----
DeepInfer 用 `get_kv_index` 这类 Persistent 内核替换 Python 侧的页表展开与 ragged
重建。我们要先确认:我们每步的 glue 到底花多少时间?如果只有几微秒,做成内核
没有意义(反而多一次 launch 税)。

测量对象(纯 torch,不启动引擎):
  1. `_expand_page_table(block_tables[:num_reqs], sub)`     —— 页表展开
  2. `ragged_from_page_table_device(...)`                    —— ragged 元数据重建
  3. 三段 ragged 重建(与 vllm_ascend.py run() 内联版同构)
  4. `pad_batch`                                             —— padding
按 c1 / c8 两种 num_reqs,以及 sub ∈ {1, 16} 测。
"""

import sys
import time
from pathlib import Path

import torch
import torch_npu  # noqa: F401

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import tileinfer  # noqa: F401
from tileinfer.integrations.vllm_ascend import KERNEL_PAGE_SIZE, _expand_page_table
from tileinfer.metadata import pad_batch, ragged_from_page_table_device

torch.npu.set_device(torch.device("npu", 0))
DEV = torch.device("npu", 0)
ITERS = 200


def timeit(fn, iters=ITERS):
    for _ in range(10):
        fn()
    torch.npu.synchronize()
    best = 1e9
    for _ in range(3):
        t0 = time.perf_counter()
        for _ in range(iters):
            fn()
        torch.npu.synchronize()
        best = min(best, (time.perf_counter() - t0) / iters * 1e6)  # µs
    return best


def run(label, num_reqs, width, sub):
    bt = torch.randint(0, 1000, (num_reqs, width), dtype=torch.int32, device=DEV)
    seq = torch.randint(100, width * 128 * sub, (num_reqs,), dtype=torch.int32, device=DEV)

    t_exp = timeit(lambda: _expand_page_table(bt, sub))

    def ragged_full():
        exp = _expand_page_table(bt, sub)
        return ragged_from_page_table_device(exp, seq, KERNEL_PAGE_SIZE)

    t_rag = timeit(ragged_full)

    meta = ragged_from_page_table_device(_expand_page_table(bt, sub), seq, KERNEL_PAGE_SIZE)
    t_pad = timeit(lambda: pad_batch(meta, num_reqs * 2, KERNEL_PAGE_SIZE))

    # 三段内联重建(与 integration run() 同构):lens->pages->indptr,lt/ge,dst,scatter
    lens_buf = seq.clone()
    width_total = width * sub
    j = torch.arange(width_total, dtype=torch.int32, device=DEV)
    dst32 = torch.empty((num_reqs, width_total), dtype=torch.int32, device=DEV)
    valid = torch.empty((num_reqs, width_total), dtype=torch.bool, device=DEV)
    indptr = torch.empty((num_reqs + 1,), dtype=torch.int32, device=DEV)
    exp_t = _expand_page_table(bt, sub)

    def three_stage():
        pages = torch.div(lens_buf + KERNEL_PAGE_SIZE - 1, KERNEL_PAGE_SIZE, rounding_mode="floor")
        torch.cumsum(pages, dim=0, out=indptr[1:])
        torch.lt(j.unsqueeze(0), pages.unsqueeze(1), out=valid)
        torch.add(indptr[:-1].unsqueeze(1), j.unsqueeze(0), out=dst32)

    t_three = timeit(three_stage)

    print(f"[H74G] {label:22s} num_reqs={num_reqs:2d} width={width_total:4d} "
          f"expand={t_exp:7.2f}us ragged={t_rag:7.2f}us three_stage={t_three:7.2f}us "
          f"pad={t_pad:7.2f}us  total~{t_rag + t_three:7.2f}us", flush=True)


if __name__ == "__main__":
    print("[H74G] decode 每步 glue 成本(c1/c8,sub=1/16)", flush=True)
    for sub in (1, 16):
        for n, w in ((1, 64), (8, 64), (8, 256)):
            run(f"sub={sub}", n, w, sub)
    print("[H74G] DONE", flush=True)
