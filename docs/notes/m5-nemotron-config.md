# Nemotron 目标 config 快照来源

- 文件：[`m5-nemotron-config.json`](m5-nemotron-config.json)
- 来源：`https://hf-mirror.com/nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16/raw/main/config.json`
- 获取时间：2026-11-12（离线阶段，仅取 config）
- 范围：**只取 config.json**，没有下载权重、tokenizer 或任何 tensor。
- 用途：核对层型、MTP 声明、上下文和 MoE 参数；不作为权重可用性的证据。

原始 config 使用 `layers_block_type` / `mtp_layers_block_type`，与 vLLM 0.23 的
`NemotronHConfig` 不兼容；转换方式见
[`m5-nemotron-declaration.md`](m5-nemotron-declaration.md) §2 与
[`m5-nemotron-runbook.md`](m5-nemotron-runbook.md) §1.1。
