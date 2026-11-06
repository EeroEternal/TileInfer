"""``legacy/`` 是冻结快照:pytest 不收集这里的任何文件。

主树跑 ``pytest tests`` 时本来就不会进来;这里挡住的是从仓库根目录直接 ``pytest`` 的情况,
免得把现网旧的 vLLM-Ascend 用例混进主树结果(它们需要 vllm / vllm_ascend / tilelang,
只有复现现网行为时才手动单文件执行)。
"""

from __future__ import annotations

#: 冻结目录:整棵子树都不参与收集。
collect_ignore_glob = ["*"]
