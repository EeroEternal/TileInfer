import torch

try:
    import torch_npu  # noqa: F401
except Exception as exc:  # noqa: BLE001
    print("### torch_npu import failed:", exc)

npu = getattr(torch, "npu", None)
print("### torch.npu:", bool(npu))
for name in ("CUDAGraph", "graph", "graph_pool_handle", "Stream", "mem_get_info", "synchronize"):
    print(f"### torch.npu.{name}:", getattr(npu, name, None))

from tileinfer.vllm import graphs  # noqa: E402

aliases = getattr(graphs, "TORCH_CUDA_ALIASES", None)
print("### TORCH_CUDA_ALIASES:", aliases)
required = getattr(graphs, "REQUIRED_ALIASES", None)
print("### REQUIRED:", required)
install = getattr(graphs, "install", None)
print("### install:", install)
if install:
    try:
        result = install()
        print("### install() ->", result)
    except Exception as exc:  # noqa: BLE001
        print("### install() raised:", type(exc).__name__, exc)
    ready, reason = graphs.capture_path_ready()
    print("### after install, ready:", ready, "|", reason[:120])
