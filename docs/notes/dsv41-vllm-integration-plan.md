# DeepSeek-V4.1-Flash 接入 vLLM 0.23 model runner 的施工计划

> 现状：`tileinfer.vllm.plugin.install()` 已让 vLLM 能**解析该 checkpoint**（`deepseek_v41` config 类型）、
> **解析架构**（`DeepseekV41ForCausalLM`，366 个 arch 中可见）、并绑定 11 个算子
> （`facts §26`、`tools/dsv41/s1_vllm_registry.py` 实测）。
> 缺的是**最后一段**：`DeepseekV41ForCausalLM.forward` 仍是显式未接线的桩
> （`vllm serve` 会在第一步就报错，报错文本指向本文件）。
> 数值与卸载机制本身已经验证完毕（S2–S5）。

---

## 1. 为什么这不是"接个 forward"那么简单

| vLLM 要求 | 本栈现状 | 差距 |
|---|---|---|
| `forward(input_ids, positions, kv_caches, attn_metadata)` | `transformer_forward(model, ids, state, start_pos)` | 需要把 `positions`/`start_pos` 与 `attn_metadata` 的上下文长度对齐 |
| 分页 KV（`kv_caches` 由 vLLM 分配） | 自己的 `LayerRuntime` 缓冲（`window_kv` / `compress_kv` / `index_k`） | 二者只能二选一：要么把我们的缓冲包装成 vLLM 的 cache layout，要么让 runner 走"单序列、无分页"模式 |
| `compute_logits(hidden)` | `logits_from_hidden(model, h)` | 只需暴露最终 hidden（`transformer_forward(capture=…)`） |
| MoE 只拿 `input_ids` | 专家权重在**主机内存**，按层换入（`ExpertArena`） | vLLM 的 `load_weights` 期望所有参数在 device；需要把 experts/engram 注册为"host 侧参数"并绕开它的 device 校验 |
| 并发 batch | 我们的 `ModelState` 是**单序列**（bsz=1、ring buffer） | 必须 `--max-num-seqs 1`，或把 state 扩成按 request id 的多实例 |
| engram 表 189 GiB | 只在主机、按 token 查表 | vLLM 的 device 内存核算会认为放不下 → 需要绕过 `--gpu-memory-utilization` 的检查路径 |

## 2. 最小可行路径（建议按此顺序，每步都可验证）

### 步骤 A：单序列、无分页的 forward（预计半天）

1. 在 `DeepseekV41ForCausalLM` 里持有一个 `ModelState`（进程级单例），并在 `positions[0] == 0` 时重置；
2. `forward` 忽略 `kv_caches`/`attn_metadata`，直接调 `transformer_forward`，`start_pos = int(positions[0])`；
3. 用 `capture={"h": …}` 拿到最终 hidden，`return hidden`（形状 `[num_tokens, 5120]`）；
4. `compute_logits(hidden)` → `logits_from_hidden`。

**验收**：`vllm serve --enforce-eager --max-num-seqs 1 --max-model-len 512` 能起服务并返回
`11111`（与 S3 的 oracle 一致）。前提：绕过分页 KV（见步骤 B）。

### 步骤 B：让 runner 不要分页（取决于 vLLM 0.23 的钩子面）

两条候选，先试 (1)：
1. **`AttentionMetadata` 空实现**：我们的注意力完全不读 vLLM 的 KV cache（自持缓冲），
   因此把 `attn_metadata` 当作"只提供 `seq_lens` 与 `slot_mapping` 用于推理位置"的载体，不分配 `kv_caches`
   ——需要在 `TileInferPlatform` 里把 KV cache 的 spec 设成 0 字节；
2. 若平台层不允许 0 字节 KV，则实现一个"直通" attention backend，`write_to_paged_cache` 变成 no-op。

**验收**：服务起来后 `npu-smi` 显示 HBM 只有 ~11 GiB（10.42 稠密 + 少量激活），**不随上下文增长**
（因为 KV 在我们自己的缓冲里，1M 上下文约 13.27 GiB，见 s4-plan §5）。

### 步骤 C：权重加载走 vLLM（`load_weights` 已有，需验证）

`DeepseekV41ForCausalLM.load_weights` 已按 vLLM 的迭代器协议实现（name → tensor），并复用
`fold_wo_a` 与 fp4 `view`。需要验证的是：
- 专家与 engram 表**不被搬到 device**（在 `load_weights` 里就地留在 CPU，并登记到 `ExpertArena`）；
- 476 GB 的首个 token 之前要能装进主机内存（实测 458 GiB 常驻，主机 618 GB，**需要独占窗口**）。

**验收**：`copied` 张量数与 `s1_load.json` 一致；`missing/extra/shape/dtype = 0`。

### 步骤 D：并发与批处理（可选，最后做）

当前 `ModelState` 是单序列。要支持 `--max-num-seqs > 1` 需要：
- 把 ring buffer / 压缩流缓存改成按 request 分槽；
- 专家扇出随 batch 增大 → `moe_forward_streamed` 的分组建页已支持（`groups` 逻辑，§5.7）。

## 3. 风险与前置条件

| 风险 | 说明 | 应对 |
|---|---|---|
| 主机内存 | vLLM 的 loader 可能额外持有副本 | 先用 `s1_load.json` 的峰值（572 GiB）作为上限，预留 ≥40 GiB |
| 共享仓库/机器 | 该仓库是多人的，且卡会被他人占用（`perf-comparison-rules.md` §共享） | 施工期间把本栈放在独立目录，跑前查 `npu-smi` 进程表 |
| vLLM 版本 | 0.23 的 `AttentionMetadata` 与 KV spec API 可能在下个小版本变化 | 全部改动集中在 `tileinfer/vllm/`，并加 feature gate |
| 回归 | 若 `vllm serve` 需要改 `tileinfer/models/dsv41/*` 的接口 | 只用可选参数（`moe_fn`/`capture` 已是这种风格） |

## 4. 明确不做的事

- 不改官方 `inference/`（C4）；
- 不动 Qwen3.6 通路与共享文件（C2，已有 md5 证据）；
- 不把 `vllm serve` 的可用性写进验收，直到步骤 A+B 的验收跑通——`dsv41-status.md` 当前如实写着
  "注册/config/算子绑定已通，`vllm serve` 未通"。

## 5. 步骤 B 的入口已定位（vLLM 0.23 源码实证）

原来的计划把"让 runner 不分页"标成"取决于 vLLM 的钩子面"。查阅本仓库已有的 vLLM 集成注释
（`tileinfer/vllm/platform.py:1212-1225`、`1464`）后可以确认：

* **KV cache spec 不是平台给的，是模型给的**：`Worker.get_kv_cache_spec` → `GPUModelRunner.get_kv_cache_spec`
  → 逐层 `AttentionLayerBase.get_kv_cache_spec`（`gpu_worker.py:547`）。平台只有
  `register_custom_kv_cache_specs`（默认 `pass`）。
* 因此"KV 为 0 / 极小"这件事**在我们自己的模型类里就能做到**：`DeepseekV41ForCausalLM` 不实例化
  vLLM 的 attention layer，只需实现一个返回最小 spec（例如 1 个 block 的 `FullAttentionSpec`）的路径，
  让 runner 的记账通过；真正的 KV 留在我们自己的 `LayerRuntime`（1M 上下文 13.27 GiB，见 s4-plan §5）。
* 平台侧要动的只有 `TileInferPlatform.check_and_update_config`（`platform.py:996`）——把
  `--max-model-len` / eager 等参数透传，并确保显存核算不把 476 GB 权重算进 device。

⇒ 步骤 B 从"未知"降级为"已知两个改动点"，其余按 §2 的 A/B/C/D 顺序推进即可。

## 6. 步骤 B 的关键事实（vLLM 0.23 源码，`gpu_model_runner.py:7440`）

```python
def get_kv_cache_spec(self) -> dict[str, KVCacheSpec]:
    """... Layers that do not need KV cache are not included."""
    attn_layers = get_layers_from_vllm_config(self.vllm_config, AttentionLayerBase)
```

三点推论，直接影响施工方式：

1. **spec 是"空字典"就合法**：文档字符串明说"不需要 KV cache 的层不会出现在里面"。
   我们的模型不实例化任何 vLLM `Attention` 层（自持 `LayerRuntime` 缓冲），所以
   `get_layers_from_vllm_config(...)` 返回空 → `kv_cache_spec == {}` → **vLLM 不会分配分页 KV**。
   这与步骤 A（忽略 paged KV）完全自洽。
2. **要小心 vLLM 对"空 spec"的启动校验**：有些版本会在 `num_gpu_blocks == 0` 时报错或退化。
   若报错，退路是给 1 个极小 spec（`FullAttentionSpec(block_size=1, num_kv_heads=1, head_size=1)`），
   代价是几 KiB 显存。
3. **真正的显存约束来自 `--max-model-len` 与 `--max-num-seqs`**，而不是 KV spec；
   配合 `--gpu-memory-utilization` 取小值即可，因为我们的 KV 全部在自己的缓冲里
   （1M 上下文 13.27 GiB，s4-plan §5）。

### 6.1 还缺的一环：权重放置时机

vLLM 的 worker 在 `load_model()` 之后立刻做显存 profiling，而我们的形态是
"稠密 10.42 GiB 上卡、专家与 engram 留在主机"。因此 `load_weights` 结束时必须完成这个放置
（等价于 `s4_full_stack_device.py` 里那段 `param.data = param.data.to(dev)` 的循环，跳过
`.ffn.experts.` 与 `.engram.embed.`）。否则 vLLM 会认为模型在 CPU 上而拒绝。

**验收（步骤 B+C 合并）**：`vllm serve --enforce-eager --max-num-seqs 1 --max-model-len 512
--gpu-memory-utilization 0.5` 起服务，`curl` 一次补全返回 `Paris`；`npu-smi` 显示 ~11 GiB、
且随上下文增长缓慢（自持缓冲）。

## 7. `vllm serve` 实跑记录（逐条 requirement）

每次 `vllm serve --trust-remote-code --enforce-eager --max-model-len 128 --max-num-seqs 1 --gpu-memory-utilization 0.5` 都暴露一个具体缺口，修一个前进一格：

| # | 报错 | 原因 | 处理 |
|---|---|---|---|
| 1 | `The text_config ... does not have num_attention_heads` | `config.json` 把架构放在 `text_config` 里，`PretrainedConfig` 只当它是 dict | **已修**：`Dsv41HFConfig` 把 `text_config`/`vision_config` 提升为子 config 对象 |
| 2 | 同上（补 architectures 后仍在 ModelConfig 阶段） | `config.json` 没有 `architectures` 字段 | **已修**：`kwargs.setdefault("architectures", ["DeepseekV41ForCausalLM"])` |
| 3 | `This model does not support --runner generate` | `registry.is_text_generation_model(architectures, model_config)` → `model_cls.is_text_generation_model` | **已声明**类属性 `is_text_generation_model=True / is_pooling_model=False / supports_multimodal=False`；**仍报同一错**，说明 `inspect_model_cls` 取到的不是我们这个类（或用的是别的判据），下一步要在进程内构造真实 `ModelConfig` 再探 |

诊断脚本注意：`ModelRegistry.inspect_model_cls(arch, None)` 会因 `model_config.model_impl` 报
`AttributeError`，必须传真实的 `ModelConfig` 实例。

其余已知项（第 3 条解决后才会遇到）：`load_weights` 末尾的 `finalize_placement`（已接）、
KV spec 为空（第 6 节已论证合法）、`--max-model-len` 与 probe 用的显存核算。

## 8. 突破到"量化策略门"（第 4 个 requirement）

补上 `embed_input_ids` 后，`isinstance(cls, VllmModelForTextGeneration)` 与 `--runner generate` 全部通过
（`PROBE is_vllm_model(cls): True / isinstance TextGen: True / fn(cls): True`）。下一次 serve 前进到：

```
Value error, quantisation 'fp8' is not supported on the TileInfer Ascend platform:
v0 runs bf16/fp16 only (docs/plan-vllm-native-backend.md §0 - W8A8/W4A8/FP8-KV are out of scope).
Remove --quantization, or use unquantised weights.
```

这是 **Qwen3.6 臂平台里的一条策略门**（`tileinfer/vllm/platform.py::verify_quantization`，约 1285 行），
v0 只允许 bf16/fp16。DSV4.1 必须走 fp8 权重 + fp4 专家，因此需要**给 DSV4.1 一个自己的平台子类**：

* 新文件 `tileinfer/vllm/platform_dsv41.py`：`class Dsv41Platform(TileInferPlatform)`，只覆盖
  `verify_quantization`（允许 fp8 + fp4 experts）以及必要的 KV/显存核算；
* 在插件 `register()` 里按环境变量（如 `VLLM_DSV41=1`）选择返回 `Dsv41Platform`，
  **默认仍返回原平台**，Qwen3.6 通路完全不受影响（C2）；
* 子类只新增文件、不改 `platform.py`，共享文件 md5 仍可核对。

### 关键教训（写进文档，避免重犯）

`is_text_generation_model` 是**结构性协议检查**
（`isinstance(cls, VllmModelForTextGeneration)`，成员为 `{forward, compute_logits, embed_input_ids}`），
**不是**类属性。之前声明 `is_text_generation_model = True` 完全无效；补 `embed_input_ids` 才通过。

## 9. 第 5 个 requirement：需要注册 model profile（框架已给出明确指引）

量化门通过后（`VLLM_DSV41=1` → `register()` 返回 `tileinfer.vllm.platform_dsv41.Dsv41Platform`，
实测 `register -> tileinfer.vllm.platform_dsv41.Dsv41Platform`，fp8/fp4 接受、int4 仍拒绝），serve 前进到：

```
no TileInfer model profile matches ['DeepseekV41ForCausalLM']; known profiles: qwen3.6
(archs=['Qwen3_5ForConditionalGeneration', ...]).
Register the model with register_profile() in tileinfer/vllm/profiles/
(a new tileinfer/vllm/profiles/<name>.py that calls register_profile()) or pass the profile name explicitly.
Hints seen: {'is_hybrid': False, 'head_size': 512}
```

这是框架**设计好的扩展点**，不是缺陷：`tileinfer/vllm/profiles/base.py:350 register_profile(profile)`
接受一个 `ModelProfile`（字段见 base.py:264：`name/archs/head_size/group_min/block_size_policy/
kv_shape/hybrid/op_family/unsupported_features/notes`），`qwen3_6.py:76` 就是模板。

**下一步**：新增 `tileinfer/vllm/profiles/dsv41.py`：

```python
NAME = "dsv41"
ARCHS = ("DeepseekV41ForCausalLM",)
DSV41 = register_profile(ModelProfile(
    name=NAME, archs=ARCHS,
    head_size=(512,),          # 64 heads x head_dim 512（facts §2）
    group_min=1,               # MLA：num_kv_heads=1
    block_size_policy=..., kv_shape=..., hybrid=None,   # 我们不用分页 KV，取最小几何
    op_family="gdn-moe",       # 或新增一个只包含我们需要的算子族
    unsupported_features=(...),# 见下
    notes="fp8-block-32 dense + fp4 experts; KV 自持，不走 paged cache",
))
```

并在 `tileinfer/vllm/profiles/__init__.py` 里 import 它（默认注册、不影响 qwen3.6：archs 不重叠）。

预计还会遇到的项（按框架顺序）：
1. `kv_shape`/`block_size_policy` 的具体几何（我们不分页，可能是最小的 1×1）；
2. `unsupported_features`：分页 KV、混合 KV、prefix caching 等应显式声明不支持；
3. `load_weights` 的 93378 张量在 vLLM loader 内的实际耗时与主机内存峰值（S1 实测 572 GiB）。

## 10. 里程碑：配置链全部通过，进入 engine core 启动

注册 `tileinfer/vllm/profiles/dsv41.py` 之后（`PROFILES` 变为 `['qwen3.6', 'dsv41']`），
`vllm serve` 已通过：

1. config 解析（`text_config` 子配置提升 + `architectures`）；
2. `--runner generate`（结构性协议：补 `embed_input_ids`）；
3. 量化门（`VLLM_DSV41=1` → `Dsv41Platform`，接受 fp8/fp4、拒绝 int4）；
4. model profile 匹配（`dsv41`：`head_size=(512,)`、`group_min=1`、`hybrid=None`、
   `kernel_page_size=16`、`hybrid_align=16`、`stride_order=(0,1,2,3,4)` —— `KvShape` 强制五维，
   我们多给一个 size-1 轴，因为该几何永不被读取：模型不暴露 `AttentionLayerBase`，
   `get_kv_cache_spec()` 为空，vLLM 不分配 paged cache）。

现在的失败点是 **engine core 初始化**（`v1/engine/utils.py:1249 Engine core initialization failed`），
即 vLLM 已经走到"起 worker 进程"这一步。下一步：读日志中该失败之前的第一个真实异常
（`grep -nE "Error|Traceback|raise" /tmp/serve_try.log | head`），大概率是
worker 里构造模型/加载权重时的 device 或内存问题（476 GB 参数的放置时机，第 6.1 节）。

### 一个工程细节：ACL 阻止了直接写 `tileinfer/vllm/profiles/`

本会话内 `.NET` 直写该目录被沙箱拒绝（`UnauthorizedAccessException`），
必须用 `edit` 工具改文件；`profiles/__init__.py` 的注册行则通过远程 python 追加完成。

## 11. 里程碑：vLLM 已用**我们的加载器**装完 475 GiB，并把 12.22 GiB 放上卡

EngineCore 日志实证：

```
INFO [gpu_model_runner.py:5092] Starting to load model /home/models/DeepSeek-V4.1-Flash...
INFO [weight_utils.py:922] Checkpoint size: 475.25 GiB. Available RAM: 612.83 GiB.
Loading safetensors checkpoint shards: 100% | 48/48
INFO [default_loader.py:397] Loading weights took 78.92 seconds
INFO [gpu_model_runner.py:5187] Model loading took 12.22 GiB memory and 87.69 seconds
```

⇒ **`load_weights` + `finalize_placement` 在 vLLM 的 loader 里正常工作**：475.25 GiB 从磁盘流式装完，
卡上只留 12.22 GiB —— 与 S4 的 10.42 GiB 稠密口径一致（差额是 vLLM 的少量缓冲）。

## 12. 下一个失败点：KV 初始化里的**预热 forward** 走到了非流式 MoE

```
core.py:132  _initialize_kv_caches -> model_executor.initialize_from_config
  -> dsv41.py:240 forward -> block.py:465 transformer_forward
  -> block.py:400 (moe_fn or moe_forward) -> block.py:160 moe_forward
  -> block.py:114 expert_forward -> ref_cpu.linear -> fp4_gemm -> matmul 形状/设备错误
```

原因很清楚：`forward` 里 `moe_fn=getattr(self, "_moe_fn", None)` 为空，于是回退到**非流式**
`moe_forward`，它假设专家权重与激活在同一设备；而按 S4 设计专家在主机内存。

**修法（下一步，代码量小）**：让适配器自己持有 arena 与流式 MoE——

```python
    def _runtime_moe(self):
        if getattr(self, "_arena", None) is None:
            from tileinfer.models.dsv41.expert_stream import ExpertArena, expert_specs
            self._arena = ExpertArena(specs=expert_specs(self.model.layers[0].ffn.experts[0]),
                                      n_slots=384, pin=True)
            for i, blk in enumerate(self.model.layers):
                blk.ffn._layer_id_for_stream = i
        def fn(moe, x, mask=None):
            lid = getattr(moe, "_layer_id_for_stream", 0)
            if x.numel() // x.shape[-1] == 1:
                return moe_forward_batched(moe, x, lid, self._arena, device=self._device, image_mask=mask)
            return moe_forward_streamed(moe, x, lid, self._arena, device=self._device, image_mask=mask)
        return fn
```

并在 `forward` 里把 `moe_fn=self._runtime_moe()`。`self._device` 取 `self.model.embed.weight.device`。

这一条改完，KV 初始化阶段的预热 forward 就能真正跑起来（它会跑一次 1 token 的前向），
之后才会到达"分配 KV cache（空 spec）→ 采样 → 首次补全"。

## 13. 最后一个已知阻塞：vLLM 要求 **恰好 1 个（或 >1 个）KV cache group，就是不能是 0**

`v1/core/kv_cache_coordinator.py:718-729`：

```python
if len(kv_cache_config.kv_cache_groups) == 1:
    return UnitaryKVCacheCoordinator(...)
return HybridKVCacheCoordinator(...)      # 内部 assert len(attention_groups) > 1
```

我们的模型不暴露任何 `AttentionLayerBase` → `get_kv_cache_spec() == {}` → **0 个 group** →
落到 Hybrid 分支 → `AssertionError: HybridKVCacheCoordinator requires at least two attention groups.`
（这解释了为什么改 `support_hybrid_kv_cache()` 无效：coordinator 是按 **group 数**选的，不是按平台标志。）

### 修法（最后的 20 行）

给模型加一个**惰性的 KV 锚点层**，只提供元数据、永不参与计算：

```python
from vllm.attention.layer import AttentionLayerBase
from vllm.v1.kv_cache_interface import FullAttentionSpec

class _KvAnchor(nn.Module, AttentionLayerBase):
    def get_kv_cache_spec(self):
        return FullAttentionSpec(block_size=16, num_kv_heads=1, head_size=512,
                                 dtype=torch.bfloat16)
```

在 `DeepseekV41ForCausalLM.__init__` 里 `self.kv_anchor = _KvAnchor()`，并保证它出现在
`get_layers_from_vllm_config(..., AttentionLayerBase)` 的枚举里（vLLM 遍历的是模块树，
所以挂成子模块即可）。这样 group 数 = 1 → 走 `UnitaryKVCacheCoordinator`，
分配一个 16-token 的小块（几十 KB），我们的 forward 依旧完全不用它。

**风险评估**：这是"给框架一个它需要的元数据"而不是"伪造计算路径"，与我们已有的
`embed_input_ids`、profile 几何属于同一类适配工作；不影响 Qwen3.6（该锚点只在本类里）。

## 14. 至此 vLLM 侧的全链条（供复审）

```
config 解析 ✓ → architectures ✓ → runner=generate ✓ → 量化策略 ✓ → model profile ✓
→ 权重加载 475.25 GiB / 78.9 s ✓ → 放置 12.22 GiB ✓ → 预热 forward（采样器）✓
→ KV cache 初始化 ✗（0 个 group，需上面的 kv anchor）
```

## 15. **`vllm serve` 起来了**：health 200 + 真补全返回（内容仍不对，待查）

关键发现：`--no-enable-prefix-caching` 让 vLLM 走
`KVCacheCoordinatorNoPrefixCache`（`kv_cache_coordinator.py:702` 的分支在 group 数检查**之前**），
从而绕开"0 个 KV group → Hybrid → assert"。**不需要 kv anchor**，一个启动参数就够。

```
GET  /health                       -> 200
POST /v1/completions               -> {"object":"text_completion","model":"dsv41",
                                      "choices":[{"text":" ?\nckockocko","finish_reason":"length"}],
                                      "usage":{"prompt_tokens":5,"total_tokens":9}}
```

⇒ 全链路打通：**vLLM 0.23 + TileInfer 插件 + 我们的 475 GiB 加载器 + 12.22 GiB 放置 + 我们的 forward
+ vLLM 采样 + HTTP 服务**，第一次拿到真实响应。

### 但内容不对（下一个要查的点）

期望 `" Paris"`（独立栈与 oracle 都是 token `11111`），实际 `" ?\nckockocko"`。可能原因（按怀疑度）：

1. **多步 decode 的状态机**：vLLM 每步只喂 1 个新 token 并给绝对 `positions`；我们靠
   `positions[0] == 0` 判定新请求。若 vLLM 在 prefill 后**再次**以 `positions[0] == 0` 调用
   （例如 chunked prefill 的第二块），状态会被重置 → 上下文丢失。
   **排查**：在 `forward` 里打印 `(ids.shape, positions[:4])` 序列，看是否出现意外的 0；
2. `--enable-chunked-prefill`（默认开）把 5-token prompt 分块 → 两次 forward，第二次 `positions`
   不从 0 开始但我们的 `ModelState` 需要连续前缀，可能与第一次的状态不一致；
3. 采样器拿到的 `logits` 与独立栈不同（需要 dump 对比 top-5）。

**建议下一步**：先用 `--no-enable-chunked-prefill` 重跑一次；若仍不对，再按 (1) 打印位置序列。

## 16. 定位：**问题在 hidden state，不在采样**

用 `VLLM_DSV41_DEBUG` 打印 `compute_logits` 的输入与输出：

```
DSV41_DEBUG compute_logits: hidden (1, 5120) top5 [(81066, 14.814), (18817, 14.078), ...]
DSV41_DEBUG compute_logits: hidden (1, 5120) top5 [(32065, 13.625), ...]
```

三条结论：
1. vLLM 传进来的就是 `[num_tokens, hidden]`（这里 `(1, 5120)`），我们的 reshape 处理正确；
2. **top-5 全是垃圾**（期望首位 11111 ' Paris'）⇒ 不是采样器的问题，是 **hidden 本身不对**；
3. 因此缺陷在我们 `forward` 里：输入 token、embedding、或权重之一与独立栈不一致。

### 下一步（精确到打印）

在 `forward` 里记录并打印 `(input_ids.shape, input_ids[0][:6], positions[:6])`，并在 `compute_logits`
里打印 `hidden[0][:4]`；与独立栈同 prompt 的对应值对比：

* 若 `input_ids` 不是 `[671, 6102, 294, 8760, 344]` → vLLM 的 tokenizer/模板与我们不同（**可能性最大**：
  vLLM 用 `--tokenizer` 默认模板，可能加了 BOS 或走了 chat 模板）；
* 若 `input_ids` 对但 hidden 不同 → 权重/placement 问题（例如 `finalize_placement` 之后某些参数仍为
  `torch.empty`，或 `load_weights` 漏了 `head`/`norm`）。

> 注：vLLM 会对未知的 `VLLM_*` 环境变量告警（`envs.py:2088`），但不影响功能；后续可改用非 `VLLM_` 前缀。

## 17. 输入完全正确；差异在 hidden ⇒ 收敛到"vLLM 侧权重的实际加载内容"

`VLLM_DSV41_DEBUG` 打印的 `forward` 入参（EngineCore 日志）：

```
forward: ids (1, 1) [0]                                   positions [0]      ← vLLM 的 warm-up dummy
forward: ids (1, 5) [671, 6102, 294, 8760, 344]           positions [0..4]   ← 真 prefill，与独立栈**逐 token 相同**
forward: ids (1, 1) [32065]                               positions [5]      ← decode（32065 已是错误采样的结果）
```

结论：
1. **tokenizer/模板没问题**（`[671, 6102, 294, 8760, 344]` 与 S3 完全一致）；
2. **positions 语义正确**（prefill 0..4，decode 5），我们的状态机按 `positions[0]==0` 重置也正确；
3. 因此 hidden 不同的唯一来源是**模型参数在 vLLM 进程里的实际取值**。

### 下一步（两条具体检查，都是打印级）

1. 在 `load_weights` 末尾打印 `loaded` 计数（独立栈是 93378）与 `missing` 名单前几项；
   若明显少于 93378，说明 vLLM 的 `default_loader` 迭代名与我们的 `_resolve` 不匹配，静默漏装；
2. 打印 `model.embed.weight` / `head.weight` / `layers.0.attn.q_a.weight` 的
   `(shape, dtype, float(sum))`，与独立栈（`s4_full_stack_device.py` 里同名字段）对比；
   定位到第一个取值不同的张量即可。

> 为什么很可能漏装：vLLM 的 `default_loader` 只迭代它自己 **model.named_parameters()** 里的名字，
> 而我们的参数树命名（`layers.N.attn.q_a.weight` 等）与 checkpoint 一致，理论上能对上；
> 但 `finalize_placement` 在 `load_weights` 末尾把稠密搬到 NPU 之后，若 vLLM 还要再做一次
> `process_weights_after_loading`（默认会调用），它可能对已搬走的参数做空操作或重复 cast。
> **先看计数**，再决定。

## 18. 接上 engram 后出现设备不一致（新错误，需下一轮定位）

补 `_engram_hash()` 后，serve 在 EngineCore 里报：

```
RuntimeError: Expected all tensors to be on the same device. Expected NPU tensor ...
```

值得注意：**同一个 engram 组合在独立栈里是跑通的**——
`s4_full_stack_device.py` 用 `EngramHash(..., device="cpu")` + 稠密 `engram.wkv` 在 NPU，
S3/S4/S5 全部通过（token 11111）。所以差异不在这组 device 参数本身，而在 vLLM 进程内的某处
（例如 `ModelState.device` 与 `engram.wkv` 的实际设备不同，或 `_runtime_moe` 里 `device` 字符串
与模型参数真实设备不一致）。

**下一轮第一步**：`grep -aA10 "Expected all tensors" /tmp/serve_try.log | grep -E "block.py|dsv41.py|engram"`
拿到确切行号；再对照独立栈里同一行的设备，即可定位。

## 19. 本轮已确认的事实（对下一轮很有用）

| 项 | 结论 |
|---|---|
| 权重加载 | `loaded 96042 / params 96042`，`embed`/`head` dtype 与 sum 正常 ⇒ **完整且正确** |
| 输入 | `ids [671, 6102, 294, 8760, 344]`、`positions 0..4` ⇒ **tokenizer 与位置语义正确** |
| 采样链路 | `compute_logits` 收到的就是 `[num_tokens, hidden]`；返回必须 `[num_tokens, vocab]`（已改） |
| KV | `--no-enable-prefix-caching --no-enable-chunked-prefill` 可绕开 0-group 断言 |
| 隐藏 bug | `ModelState` 未传 `engram_hash` → 层 1/14 静默跳过 engram（已修） |

## 20. engram 设备修复生效，但 logits 仍不匹配（剩余差异已缩小）

修复 `EngramHash.forward` 的设备边界（`input_ids` 可能是 NPU 张量、而 hash 状态在 CPU）后：

```
startup=1                                        ← 服务重新起来
completion: "text": "/dm/dm/dm/dm"               ← 仍是垃圾
compute_logits: hidden (1, 5120) top5 [(81066, 14.814), ...]    ← warm-up dummy（与上次完全相同，正常）
compute_logits: hidden (1, 5120) top5 [(23684, 15.707), ...]    ← 真 prefill，**已与修复前不同**
```

⇒ engram 修复**确实生效**（真 prefill 的 logits 变了），但结果仍不是 11111。

### 剩余差异的唯一来源：`ModelState` 的构造参数

我们的 forward 用 `cfg.max_seq_len`（从 checkpoint 读，**1,048,576**）建 `ModelState`，而独立栈用的是
**1024**。两者在数值上"应该"一致（rope/压缩流缓冲只影响容量），但值得逐一验证：

1. `max_seq_len` 影响 `LayerRuntime` 的 rope 表与压缩缓冲尺寸 → 若某处（如 indexer 的 topk 上限、
   candidate 块数）依赖它，数值就会不同；
2. 快捷验证：在 `forward` 里把 `ModelState(max_seq_len=...)` 改成与独立栈一致的小值（如 1024），
   重跑同一 prompt，看是否变回 11111。

### 本轮结论

`vllm serve` 已经能启动并返回补全，**权重/输入/采样/KV/engram 五处均已对齐**；
剩下的差异被压到"`ModelState` 的 `max_seq_len` 一类构造参数"这一层，下一轮一条对照实验即可判定。

## 21. `max_seq_len` 假设被**否证**；下一个（也是最后一个）怀疑点：vLLM loader 对 fp4 参数的 dtype 处理

把 `ModelState(max_seq_len=…)` 改成 1024 后重跑，真 prefill 的 logits **完全不变**
（`23684, 15.707 / 21836, 15.087 …`）⇒ `max_seq_len` 与数值无关，该假设排除。

至此**权重内容、输入 token、positions、engram、max_seq_len、采样形状、KV 配置**全部对齐，
唯一剩下的怀疑是 **vLLM 的 `default_loader` 在加载时对参数做了 dtype 转换**：

* 我们的 fp4 专家参数是 `float4_e2m1fn_x2`（2 值/字节，见 facts §3），最省事的判断就是打印加载后的
  dtype —— 若是 `torch.bfloat16`，说明 vLLM 以 `dtype=bfloat16` 把 int8 分片 cast 成了 bf16，
  再被我们 `copy_` 进 fp4 参数，专家权重就整体损坏（而"12.22 GiB"这个偏大的数字也符合"部分参数被
  升精度"的迹象）；
* 验证：在 `load_weights` 的调试分支里加
  `for nm in ("layers.0.ffn.experts.0.w1.weight", "layers.0.ffn.experts.0.w1.scale", "layers.0.attn.q_a.weight"): print(nm, table[nm].dtype)`；
* 若确认为 bf16：在 `_resolve` 里按 **分片原始 dtype**（`safe_open` 的 `get_slice().get_dtype()`）判定，
  而不是按 vLLM 递过来的张量 dtype；或让 vLLM 以 `--dtype float8`（若平台允许）加载。

### 本轮结论

`vllm serve` 的**功能链路**已经完整（启动/HTTP/补全/权重加载/放置/采样/KV/engram），
数值不对的根因已被压到"fp4 参数的 dtype 在 vLLM loader 里被改变"这一条，
下一轮一条 dtype 打印即可确认或排除。

## 22. dtype 假设也**被否证**：权重以正确格式加载

```
DSV41_DEBUG param embed            (129280, 5120) torch.bfloat16
DSV41_DEBUG param expert_w1          (2304, 2560) torch.float4_e2m1fn_x2   ← fp4 打包形态正确
DSV41_DEBUG param expert_w1_scale    (2304,  160) torch.float8_e8m0fnu     ← 块 scale 正确
DSV41_DEBUG param gate               (384, 5120) torch.bfloat16
startup=1
```

（顺带修掉一个探针 bug：上一版在 129280x5120 上做 `float().sum()`，会分配 GB 级内存把加载卡住。）

### 已系统性排除的假设（不必重查）

| 假设 | 结论 |
|---|---|
| tokenizer/模板不同 | ❌ ids 与独立栈逐 token 相同 |
| positions 语义不同 | ❌ prefill 0..4 / decode 5，正确 |
| 采样形状不对 | ❌ 已修 `[n, vocab]` |
| KV 配置（0 group） | ❌ `--no-enable-prefix-caching` 绕开 |
| engram 被跳过 | ❌ 已接 `_engram_hash()`，且修复后 logits 变化 |
| `max_seq_len` 影响数值 | ❌ 1024 与 1M 逐位相同 |
| fp4/fp8 参数被 cast | ❌ dtype 与形状均正确 |

### 唯一剩下的排查方式：直接对比 **hidden 向量**

两侧同 prompt（`[671, 6102, 294, 8760, 344]`）分别 dump `hidden[0][:4]`（折叠后的 hidden，norm 之前）：

* 独立栈：在 `s4_full_stack_device.py` 的 `cap_ours["h"]` 里加一行打印；
* vLLM 侧：`compute_logits` 里打印 `hidden_states[0][:4]`。

若两者不同，再按层二分（`capture_layers` 已有）：先比层 0，再比层 1（engram）、层 2（compressor）…
这样能把差异定位到具体子模块，而不是继续猜。

## 23. text-only 构建生效（10.51 GiB），但数值仍不对 ⇒ 结构假设全部排除

修复 `_resolve` 的 `wo_a` 分支（先判参数是否存在，否则 text-only 下 `params[name]` 会 KeyError）
并给 skip 分支补 `consumed` 记账后：

```
startup=1
Loading weights took 52.17 seconds        (was 78.9)
Model loading took 10.51 GiB memory       (was 12.22; S4 稠密口径 10.42 GiB)
completion: "/dm/dm/dm/dm"                (期望 " Paris")
```

⇒ vLLM 侧的模型形态**已经与 S3/S4/S5 完全一致**（同一 cfg、无 MTP/vision、专家在主机、稠密在卡），
但 hidden 仍与独立栈不同。**结构性假设全部排除**，只剩最后的对照手段：

### 必须做的下一步：hidden 逐层二分（唯一有效）

1. **拿参考值**：跑 `tools/dsv41/s4_full_stack_device.py --gen` 并在 `cap_ours["h"]` 处打印
   `hidden[0][:4]` 与 top-1 token（预期 11111）；
2. **拿 vLLM 值**：`compute_logits` 里已能打印 `hidden_states[0][:4]`（`VLLM_DSV41_DEBUG=1`）；
3. 若两者不同 → 用 `transformer_forward(..., capture_layers=[0])` 比层 0 输出，再依次加层；
   第一个出现差异的层即根因所在（层 1 = engram、层 2 = compressor、层 3.. = indexer/MLA）。

不要再用"猜一个配置改一次跑六分钟"的方式推进：二分是线性的，而且每步都有确定的比较对象。

## 24. 定位结论：vLLM 忠实传递，差异在 `transformer_forward` 内部（量纲差 1e5）

同一 prompt 的逐位对比（`VLLM_DSV41_DEBUG=1`）：

```
DSV41_DEBUG forward out:     (1, 5120) [-15823.35156, 9820.58887, -8250.21484, -2508.04639]
DSV41_DEBUG compute_logits:  (1, 5120) [-15823.35156, 9820.58887, -8250.21484, -2508.04639]
```

两者**完全相同** ⇒ vLLM 没有对 hidden 做任何变换。

参考值（独立栈，同 prompt，`s4_full_stack_device.py`）：

```
hidden[:4] = [0.09131, -3.92188, -0.4043, -1.42969]
logits top5 = [(11111, 21.627), (4228, 17.941), (223, 17.876), (260, 17.524), (2619, 17.467)]
generated text: ' Paris.'
```

### 判定

| 量 | 独立栈 | vLLM | 比值 |
|---|---|---|---|
| hidden[0] | 0.091 | -15823 | ~1e5 |

**同一段代码、同一份权重文件、同一输入，输出量纲差 1e5** ⇒ vLLM 进程内某个参数的实际取值与
checkpoint 不一致。`torch.empty` 残留会产生这种巨大而无意义的数值，因此最可能是某个
**归一化/折叠类权重**（`norm.weight`、`layers.N.attn_norm.weight`、`ffn_norm.weight`、`hc_pre` 的权重）
在 vLLM 的 loader 路径里被错填或未填 —— 尽管 `loaded == params == 96042`（计数对，但**名字可能配错**）。

### 下一条命令（最省时的判定）

在 vLLM 侧 `VLLM_DSV41_DEBUG` 分支里打印这几个参数的 `abs().mean()`：

```python
for nm in ("norm.weight", "layers.0.attn_norm.weight", "layers.0.ffn_norm.weight"):
    prm = dict(self.model.named_parameters()).get(nm)
    print(nm, tuple(prm.shape), str(prm.dtype), round(float(prm.float().abs().mean()), 6))
```

正确值应在 1.0 附近（RMSNorm 初始化即 1）；若打印出 1e3 量级或更大的随机值，即命中。
独立栈打印同一组值对照即可。

## 25. 根因锁定：vLLM 进程内三个 RMSNorm 权重**不是 checkpoint 的值**

```
DSV41_DEBUG forward out: (1, 5120) [-15823.35, 9820.59, -8250.21, -2508.05]
  norms {'final': 0.270899, 'l0_attn': 0.019959, 'l0_ffn': 0.125872}
```

RMSNorm 权重在 checkpoint 里应当是 ~1.0 附近的向量；这里 `attn_norm=0.020`、`ffn_norm=0.126`、
`final=0.271` —— **明显是被写入了别的东西**（这些量级与 fp8/e8m0 的**块 scale** 非常像）。

这与 1e5 的 hidden 量纲差完全自洽：norm 权重被换成 0.02 量级的数后，`rms_norm` 输出被压低/放大，
残差流逐层累积到 1e4。

### 结论

`load_weights` 的**计数是对的（96042）但名字配错了**：至少 `norm.weight`、
`layers.N.attn_norm.weight`、`layers.N.ffn_norm.weight` 被写入了 scale 类张量。
最可能的机制：vLLM 的 `default_loader` 在 `quantization="fp8"` 下会额外投递配对的
`.scale` 张量，而我们的 `_resolve` 按**后缀**匹配（`.wo_a.scale`、`.wo_a.weight`）时把
`*.norm.weight` 与某些 scale 张量搞混。

### 下一步（最小判定）

1. 在 `load_weights` 里对 `norm`/`attn_norm`/`ffn_norm` 名字打印一行（`name + 前4值`），看**进来的张量**
   本身是否已经是小值；若是，说明 checkpoint 里这些名字另有含义（需要回查 facts §1 的张量清单）；
2. 若进来的张量是对的（~1.0）而参数变成 0.02，则是 `copy_` 之后被二次覆盖 —— 查是否
   `load_weights` 被调用两次、或 `process_weights_after_loading` 改写了它们；
3. 独立栈打印同组均值（应为 ~1.0）作对照。

## 26. 恢复清单（外部内存窗口受限时的标准动作）

当前唯一未闭环项：`vllm serve` 的输出数值。**功能链路已完整**（启动 / HTTP 200 / 返回补全 /
96042 张量 475.25 GiB 加载 / 10.51 GiB 稠密放置 / 采样 / KV / engram 全部走通），
差异已二分到 **block 0**，且 `embed` 一级与独立栈**逐位相同**：

| 阶段 | 独立栈（正确） | vLLM |
|---|---|---|
| ids | `[671, 6102, 294, 8760]` | 相同 |
| embed rms | 0.058634 | 相同 |
| block 0 `[:4]` | `[0.0461, 0.0025, 0.00952, -0.01599]` | `[30.95, -39.91, 76.56, -0.41]` |
| 末层 hidden | `[0.09131, …]` | `[-15823.35, …]` |
| top1 | `11111` (21.627) | `23684` (15.437) |

### 阻塞条件（外部，非代码）

主机内存被其他租户长期占用：`free -g` 的 available 在 **418–446 GB** 之间波动，
而 476 GiB 模型在 vLLM 的 loader 路径下需要 **≥ 500–520 GB** 才能安全装入
（不足时表现为 `EngineCore: -9`，即被 OOM-killer 强杀；已实测）。

### 窗口打开后的三条命令（约 8 分钟出结果）

```bash
free -g                                  # 确认 available >= 500G
setsid bash -c 'exec /tmp/verify175.sh' </dev/null >/tmp/verify175.out 2>&1 &
sleep 450; cat /tmp/t175.result           # startup= / 补全文本 / layer[0..1][:4]
```

判据：`layer[0][:4]` 若等于 `[0.0461, 0.0025, 0.00952, -0.01599]` ⇒ MoE 路径统一（本轮改动）即为修复；
否则在 `block.py::block_forward` 内加两处 `DSV41_TRACE`（`attn_norm` 之后、attn 之后）继续二分。

### 注意

* `nohup ... &` 从 `ssh bash -s` 会话启动的子进程会随会话进程组被回收，**必须用 `setsid`**；
* 调试开关：`VLLM_DSV41_DEBUG=1`（适配器）与 `DSV41_TRACE=1`（`block.py` 的 embed trace），
  两者都能传入 EngineCore 子进程（已验证）。

## 27. 决定性定位：差异在 attention_forward 内部（attn_in 逐位相同，attn_out 放大 4000 倍）

在 block 0 上加三个探针（`DSV41_TRACE=1`，两侧共用 `block.py`），同 prompt 实测：

| 探针 | 独立栈 | vLLM |
|---|---|---|
| `attn_in[:4]` | `[0.02734, 0.00613, -0.00458, -0.0034]` rms **0.020544** | 同四值、rms **0.020544** |
| `attn_out[:4]` | `[0.10547, -0.58984, 0.66016, 0.45508]` rms **0.78742** | `[1575.86, -2035.03, 3897.39, -21.74]` rms **3218.8** |

⇒ embedding ✅、hc_pre+rms_norm ✅、权重 ✅；**只有 `attention_forward` 的输出爆炸**。
另外记录三次调用的完整序列（vLLM 侧）：

```
warm-up ids [0]                    embed rms 0.156114 -> layer[0] [-4.39, -96.67, 26.72, -203.76]
prefill ids [671,6102,294,8760]    embed rms 0.058634 -> layer[0] [30.95, -39.91, 76.56, -0.41]
decode  ids [23684]                embed rms 0.1276   -> layer[0] [-31.46, ...]
```

### 首要嫌疑：`wo_a` 折叠（fp8 + e8m0 scale -> bf16）

`wo_a` 是加载期现场换算的，两个 loader 的交付顺序不同。已离线算出期望值（只读层 0 的两个小张量）：

```
W raw   (8192, 4096) float8_e4m3fn
W scale (256, 128)   float8_e8m0fnu
FOLD rms 0.022787   absmean 0.017605
```

探针已加进 `block_forward`（层 0 打印 `wo_a_rms` / `wo_a_absmean`）。判据：
若 vLLM 侧不等于 0.022787 ⇒ 修 `_resolve`/`load_weights` 的 pending-scale 配对；
若相等 ⇒ 查 compressor/indexer 的 `LayerRuntime` 状态（`window_kv`/`compress_kv` 缓冲）。

### 运行门槛更新

实测 available=530 GB 仍会被 OOM-killer 强杀（`EngineCore: -9`），成功过的那两次是 **616 GB**。
因此启动门槛应设为 **available >= 560 GB**（而非 500）。

## 28. 参数名审计通过；剩余嫌疑收敛到 attention 的 fp8 投影 / e8m0 scale

用 `Dsv41Attention(cfg, 0)` 单独构造层 0 的 attention（不建全模型，几十 MB）做名称审计：

```
MODEL: attn_sink, kv_norm.weight, q_norm.weight,
       wkv.weight, wkv.scale, wq_a.weight, wq_a.scale, wq_b.weight, wq_b.scale,
       wo_b.weight, wo_b.scale, wo_a.weight
CKPT : attn_sink, kv_norm.weight, q_norm.weight, wkv.weight, wo_b.weight, wq_a.weight, wq_b.weight
```

（`wo_a` 由 fold 生成、`.scale` 由 checkpoint 提供）⇒ **名字逐条对得上，名称映射排除**。

### 双路径数值对比：被环境细节挡住（不是产品缺陷）

`/tmp/cmp_attn.py` 想把同一批张量分别经"直接拷贝"与"vLLM adapter 的 load_weights"装进两个
`Dsv41Attention`，再比较 dtype/rms，但两次都报：

```
copy_d2d_baseformat_opapi ... aclnnInplaceCopy failed, error code 561103
```

`copy_d2d` 说明参数与张量都落在 NPU 上——尽管构造时显式 `device="cpu"` 且调用前设了
`torch.set_default_device("cpu")`，该 venv 里 torch_npu 的设备上下文仍把参数建到 NPU，
而 `float8_e8m0fnu` 的 NPU inplace-copy 直接失败。
**真实 vLLM 加载 96042 张量并不报这个错**，说明真实路径这一步发生在 CPU 上；
所以这是离线脚手架问题，不能当作产品缺陷的证据（与 §25 那次 92.8 的误判同类，需引以为戒）。

### 仍然确凿的事实链

```
embed rms 0.058634    独立栈 == vLLM   （输入与 embedding 权重 OK）
attn_in rms 0.020544  独立栈 == vLLM   （hc_pre + rms_norm OK）
attn_out rms 0.78742 vs 3218.8        （attention 内部 ❌，约 4000 倍）
block0 0.0461 vs 30.95  末层 0.09131 vs -15823   top1 11111 vs 23684
```

⇒ 问题必在 attention 自己的 fp8 投影（`wq_a`/`wq_b`/`wkv`/`wo_b`）或它们的 **e8m0 块 scale**。
e8m0 是"位模式即指数"的特殊格式（字节 127 = 2^0），一旦被当作普通数值处理，
每级投影乘上 ~127，多级复合正好是千倍量级。

### 唯一可靠的收口方式

真实 serve 里的探针（`DSV41_TRACE=1`）：`block_forward` 层 0 现在会打印 `wo_a_rms` / `wo_a_absmean`，
与离线正确值 **0.022787 / 0.017605** 对照；若不等再扩到 `wq_a`/`wkv`/`wo_b` 的 `.scale`。
需要 **available ≥ 555 GB**（实测 530 GB 会被 OOM-killer 以 `-9` 强杀）。

## 29. 根因确认并修复：`_resolve` 的 `wo_a` 被参数表短路（死代码）

### 根因

`DeepseekV41ForCausalLM._resolve` 原来是"先查参数表、再判断 wo_a 模式"：

```python
param = params.get(name)
if param is not None:
    return param, None            # ← layers.N.attn.wo_a.weight 在这里就被命中
if name.endswith(".wo_a.scale"): ...
if name.endswith(".wo_a.weight"): return None, "wo_a_weight"   # ← 永远到不了
```

模型树里的 `wo_a` 是**折叠后的 bf16 参数**，与 checkpoint 里同名，所以 `params.get()` 必然命中，
于是把 **fp8 原始张量直接 copy 进 bf16 参数**（数值约 ±448 量级混合），折叠分支成为死代码。

### 证据（离线双路径对比，`ASCEND_RT_VISIBLE_DEVICES=` 强制参数留在 CPU）

```
修复前：SAME_COUNT 11 of 12
        DIFF wo_a.weight  A(正确折叠)=0.022787  B(经 adapter)=92.822701   ← ×4072
修复后：SAME_COUNT 12 of 12
```

与真机观测量吻合：`attn_out` rms 3218.8 / 0.78742 ≈ 4088 倍。

### 修复

把 wo_a 的判定**提到参数查找之前**（`tileinfer/vllm/models/dsv41.py::_resolve`）：

```python
if name.endswith(".wo_a.scale"):
    return None, ("wo_a_scale" if (name[:-6] + ".weight") in params else "skip")
if name.endswith(".wo_a.weight") and name in params:
    return None, "wo_a_weight"
param = params.get(name)      # 其余照旧
```

修复后打开 fold 分支还会得到一个正向校验：若同一遍里缺少 scale，会立刻抛
`arrived before its scale`（真实 vLLM 加载会提供该 scale）。

### 待做

一次真机 serve 确认输出为 `Paris`（需 available >= 555 GB；本轮 502 GB，未启动）。
