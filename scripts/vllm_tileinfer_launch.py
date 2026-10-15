"""Launch vLLM with the TileInfer backend registered.

    python scripts/vllm_tileinfer_launch.py serve <model> --attention-backend CUSTOM ...

Why a launcher instead of the `vllm` CLI: the backend has to be registered *before* vLLM resolves
`--attention-backend`, and an out-of-tree package can only do that by importing itself first.
`vllm_ascend` has to be importable already (it is, once the model's platform is selected) - importing
it here as well makes the ordering explicit.

T6 auto pre-warm: set ``TILEINFER_PREWARM=1`` (and ``TILEINFER_PREWARM_PAGES=<engine pages>`` so the
declared-capacity bucket matches the server's) to compile the whole kernel variant set - decode
buckets + the prefill ladder - **before** the engine starts, so the serving path never compiles
(``tileinfer.integrations.prewarm`` is the single source of truth for *what* to compile).  The pool
page count comes from the server's memory snapshot and drifts run to run, so it is passed explicitly;
omit it to skip pre-warm with a hint.
"""

from __future__ import annotations

import sys


def _serve_config(argv: list) -> dict:
    """Extract the pre-warm inputs from the ``serve`` argv (model path + a few options).

    No full vLLM arg parsing here: the launcher runs *before* vLLM resolves its config, so it reads
    the raw argv and the model's ``config.json`` only.  Missing options fall back to vLLM defaults.
    """
    model = None
    opts = {"max_model_len": 0, "max_num_seqs": 0, "max_num_batched_tokens": 0, "block_size": 0}
    # argv 里的选项名是连字符形式,opts 用下划线;此处建立映射。
    keymap = {k: k for k in opts}
    keymap.update({"--" + k.replace("_", "-"): k for k in opts})
    i = 0
    if argv and argv[0] == "serve":
        i = 1
    while i < len(argv):
        a = argv[i]
        if a in keymap and i + 1 < len(argv):
            try:
                opts[keymap[a]] = int(argv[i + 1])
            except ValueError:
                pass
            i += 2
            continue
        if a.startswith("--"):
            # option with a value (we skip its value) or a flag; only the known keys are captured above
            if i + 1 < len(argv) and not argv[i + 1].startswith("--"):
                i += 2
            else:
                i += 1
            continue
        if model is None:
            model = a
        i += 1
    return {"model": model, **opts}


def _auto_prewarm(argv: list) -> None:
    """Compile the declared kernel variant set before the engine starts (T6)."""
    import os
    import subprocess
    from pathlib import Path

    cfg = _serve_config(argv)
    model = cfg["model"]
    pages = int(os.environ.get("TILEINFER_PREWARM_PAGES", "0") or 0)
    if not model or not os.path.isdir(str(model)):
        print("[tileinfer] prewarm skipped: model path not a local directory "
              f"({model!r}); pass a local path or drop TILEINFER_PREWARM")
        return
    if pages <= 0:
        print("[tileinfer] prewarm skipped: set TILEINFER_PREWARM_PAGES=<engine KV pages> "
              "(the server's pool; find it in the startup log as 'GPU KV cache size'). "
              "Shell: python scripts/prewarm-tileinfer-kernels.py --plan-only --json ...")
        return

    import json

    from tileinfer.integrations.prewarm import build_prewarm_argv

    with open(Path(model) / "config.json", encoding="utf-8") as fh:
        raw = json.load(fh)
    tc = raw.get("text_config", raw)  # multimodal wrappers nest the LM config
    num_heads = int(tc.get("num_attention_heads") or tc.get("num_heads"))
    num_kv_heads = int(tc.get("num_key_value_heads") or num_heads)
    head_dim = int(tc.get("head_dim") or (int(tc["hidden_size"]) // num_heads))
    group = max(1, num_heads // max(1, num_kv_heads))
    max_model_len = cfg["max_model_len"] or int(tc.get("max_position_embeddings") or 0)
    # vLLM defaults when the flags are absent.
    max_num_seqs = cfg["max_num_seqs"] or 256
    max_batched = cfg["max_num_batched_tokens"] or 8192
    block = cfg["block_size"] or 128  # hybrid GDN models often use a larger engine block

    root = Path(__file__).resolve().parents[1]
    prefill_ktp = int(os.environ.get("TILEINFER_PREWARM_KV_TILE_PAGES", "0") or 0)
    args = build_prewarm_argv(
        num_kv_heads=num_kv_heads, group_size=group, head_size=head_dim,
        dtype=os.environ.get("TILEINFER_PREWARM_DTYPE", "bfloat16"),
        block_size=block, engine_pages=pages, max_num_reqs=max_num_seqs,
        max_num_batched_tokens=max_batched, max_model_len=max_model_len,
        prefill=True, prefill_kv_tile_pages=prefill_ktp,
    )
    cmd = [sys.executable, str(root / "scripts" / "prewarm-tileinfer-kernels.py"), *args]
    print(f"[tileinfer] pre-warming kernels (pages={pages}, group={group}, dim={head_dim}, "
          f"block={block}, seqs={max_num_seqs}, budget={max_batched}, len={max_model_len}) ...")
    rc = subprocess.run(cmd, cwd=str(root)).returncode
    if rc != 0:
        print(f"[tileinfer] pre-warm failed rc={rc}; the engine will fall back to lazy compile")
    else:
        print("[tileinfer] pre-warm done: the serving path should see zero compiles")


def main() -> int:
    import os

    import vllm_ascend  # noqa: F401  (its platform registers the Ascend backends)

    if os.environ.get("TILEINFER_DISABLE"):
        # the baseline: vLLM-Ascend's own (FIA) path, for an A/B comparison of the same prompt
        print("[tileinfer] TILEINFER_DISABLE is set - serving with the stock Ascend backends")
    else:
        from tileinfer.integrations import vllm_ascend as ti

        if not ti.install():
            raise SystemExit("TileInfer could not register its attention backend")
        from vllm.v1.attention.backends.registry import AttentionBackendEnum

        print("[tileinfer] CUSTOM backend resolves to:", AttentionBackendEnum.CUSTOM.get_class())

        if os.environ.get("TILEINFER_PREWARM") and sys.argv[1:2] == ["serve"]:
            _auto_prewarm(sys.argv[1:])

    from vllm.entrypoints.cli.main import main as vllm_main

    return vllm_main()


if __name__ == "__main__":
    raise SystemExit(main())
