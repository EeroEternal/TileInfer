"""fused_sample 内核编译冒烟(只管能否编译,不管数值)。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import tileinfer  # noqa: F401
import tilelang
from tileinfer.kernels.sampling.fused_sample import fused_sample_tl

V = 16384  # 最小可编译词表(真实 248320 太大,冒烟用小块)


def build(do_sample):
    return tilelang.compile(
        fused_sample_tl(V, batch=2, do_sample=do_sample),
        out_idx=None,
    )


for ds in (False, True):
    try:
        build(ds)
        print(f"do_sample={ds}: COMPILE OK")
    except Exception as e:
        print(f"do_sample={ds}: FAIL {type(e).__name__}: {str(e)[:400]}")
