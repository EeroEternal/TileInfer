# H47 取证棒 v2(iter-095,E129-2 方案):进程内 LLM + torch.profiler 包住引擎初始化,
# 使 graph capture 期逐 op 执行的 host 栈可见,把在役 replay 中 240 次/步的
# aclnnInplaceCopy_SliceAiCore_Slice 拷贝链精确映射到 torch 源(qwen3_next.py/
# tilelang_ascend950.py/vllm_ascend.py 静态候选清单,E129-3)。
# 产物:.agent_loop/run/h47_capture_prof/(chrome trace,gitignored)+
#       h47_probe.log([H47V2] 可解析行)。
# 窗口依赖:需空闲 NPU(HBM<8%),由 h47_wait_chain.sh 调度,不自行等窗。
import os
import sys

os.environ.setdefault("VLLM_NO_USAGE_STATS", "1")
os.environ.setdefault("VLLM_DO_NOT_TRACK", "1")

OUT_DIR = os.environ.get(
    "H47_PROF_DIR", "/home/userlist/zxs/TileInfer/.agent_loop/run/h47_capture_prof"
)
LOG_PATH = os.environ.get(
    "H47_LOG", "/home/userlist/zxs/TileInfer/.agent_loop/run/h47_probe.log"
)
MODEL = "/home/models/Qwen3.6-35B-A3B"


def log(msg: str) -> None:
    with open(LOG_PATH, "a") as f:
        f.write(msg + "\n")


def main() -> int:
    import torch
    from torch.profiler import ProfilerActivity, profile

    from vllm import LLM, SamplingParams

    log("[H47V2] start pid=%d %s" % (os.getpid(), sys.version.split()[0]))
    activities = [ProfilerActivity.CPU]
    try:
        activities.append(ProfilerActivity.CUDA)
    except Exception:
        pass

    # capture 发生在 LLM() 构造内(profile_run + 全 bucket capture),profiler 必须先开。
    with profile(
        activities=activities,
        with_stack=True,
        record_shapes=True,
        profile_memory=False,
    ) as prof:
        llm = LLM(
            model=MODEL,
            served_model_name=MODEL,
            tensor_parallel_size=1,
            gpu_memory_utilization=0.90,
            max_model_len=1024,
            max_num_seqs=32,
            trust_remote_code=True,
            additional_config={"multistream_overlap_shared_expert": True},
        )
        sp = SamplingParams(temperature=0, max_tokens=4, ignore_eos=True)
        prompts = ["hello " + str(i) for i in range(16)]  # 触发 bs16 桶 decode
        llm.generate(prompts, sp)

    os.makedirs(OUT_DIR, exist_ok=True)
    trace_path = os.path.join(OUT_DIR, "h47_capture_trace.json")
    try:
        prof.export_chrome_trace(trace_path)
        log("[H47V2] chrome trace saved %s" % trace_path)
    except Exception as exc:  # noqa: BLE001
        log("[H47V2] chrome trace export failed: %r" % exc)

    # 按栈聚合:重点看 aten::copy_/slice/empty_like/zero_/fill_/add_ 的调用源。
    try:
        ka = prof.key_averages(group_by_stack_n=8)
        rows = sorted(ka, key=lambda e: e.self_cpu_time_total, reverse=True)
        targets = (
            "aten::copy_",
            "aten::slice",
            "aten::empty_like",
            "aten::zero_",
            "aten::fill_",
            "aten::add_",
            "aten::clone",
            "aten::detach",
        )
        for e in rows:
            name = e.key.split("##")[0].split(" (")[0]
            if name in targets and e.self_cpu_time_total > 0:
                stack = (e.stack or ["<no-stack>"])[0]
                log(
                    "[H47V2] %-18s cnt=%-7d self_cpu=%.1fms top_stack=%s"
                    % (name, e.count, e.self_cpu_time_total / 1000.0, stack)
                )
        log("[H47V2] --- top-30 self cpu ops ---")
        for e in rows[:30]:
            stack = (e.stack or ["<no-stack>"])[0]
            log(
                "[H47V2] %-40s cnt=%-7d self_cpu=%.1fms %s"
                % (e.key.split("##")[0][:40], e.count, e.self_cpu_time_total / 1000.0, stack)
            )
    except Exception as exc:  # noqa: BLE001
        log("[H47V2] key_averages failed: %r" % exc)
        return 2
    log("[H47V2] done")
    return 0


if __name__ == "__main__":
    rc = 1
    try:
        rc = main()
    except BaseException as exc:  # noqa: BLE001
        log("[H47V2] fatal: %r" % exc)
        rc = 3
    sys.exit(rc)
