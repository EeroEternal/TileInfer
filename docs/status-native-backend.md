# 原生 vLLM 后端交接状态（status-native-backend）

> **读者**：在 Ascend 机器上接手本项目、且没有读过任何对话记录的人。
> **用途**：告诉你「哪些已经建成、凭什么证据相信它、哪些还没建成、上机第一件事做什么、哪些坑已经踩过」。
> **约定**：本文所有行号默认是 `E:\project\TileInfer\runtime\vllm-api\` 下**本地文件**的行号（转录自
> vLLM `v0.23.0`），例外只有两类，且都会显式标注：**B 级**行号指向 sglang vendored 副本（§1 定义），
> **本仓库文件**的行号则写成 `tileinfer/...`。所有计数都是本轮**实际跑出来的**。凡是没验证的都写「未核实」，不写推测。
> **测量环境**：Windows 开发机，`runtime/pylibs` 里的 torch 2.14.1+cpu / pytest 9.1.1；
> **没有** vLLM、**没有** torch_npu、**没有** tilelang、**没有** numpy；CPU 套件用「注入 fake `vllm` 包」
> 的方式验证接口契约。**本仓库的代码从未在 Ascend 硬件上运行过一行**，也从未在真实 vLLM 进程里加载过。

---

## 0. 一句话状态

**已建成**：平台层（`platform.py`，含本轮核实的两条启动路径：`worker_cls` 选择与模型架构接受，见 §7.7）、
`NPUWorker` 的四个 hook（`worker.py:825` / `:1510` / `:1689` / `:1796`）、attention 后端四件套
（`attn/backend.py`、`metadata.py`、`impl.py`、`kv_cache.py`）、插件入口与 patch 注册表
（`plugin.py`、`compat/patches.py`）、图捕获判定与桥接（`graphs.py`）、GDN 全套 torch 算子
（六步 chunk 链 + `causal_conv1d` 两入口 + gating + recurrent + `post_conv`）、MoE 三个算子、
引擎无关 serving 层（`buckets.py` / `planner.py` / `step.py` / `counters.py`）。

**三个扩展点已成形（加模型/加算子族/加更快的实现 = 加文件，不改核心）**：模型事实集中在
`tileinfer/vllm/profiles/`（`ModelProfile`：`archs` / `head_size` / `group_min` / 页策略 / KV 形状 /
hybrid / `op_family` / 拒绝清单；今天只有 `qwen3_6.py`）；vLLM 算子表集中在
`tileinfer/vllm/ops/families/`（`required_names()` + `bind()`；今天只有 `gdn_moe.py`）；同一个 ABI 的
第二个实现放进 `tileinfer/vllm/ops/candidates.py` 的 `CandidateSet`（baseline 恒在；候选必须
`enabled_by_default` **且** 记录的设备实测 `measured_ratio >= 1.0` 才可能被选中，
`TILEINFER_OPS_CANDIDATES=off` 只能关不能开）。因此 `platform.py` / `attn/backend.py` / `worker.py` /
`ops/install.py` 里不再有模型常量或算子表，它们只执行注册表。加东西的模板与「哪些代码不能动」见
[`extending.md`](extending.md)。

**混合 KV 已打开（不再是「缺口」）**：`support_hybrid_kv_cache()` 返回 **True**；10 层 full-attention +
30 层 GDN 的 spec 由模型给出，平台只贡献能力位与页几何 —— `block_size=2048`、`mamba_block_size=2048`、
`mamba_page_size_padded = 2048 × 2(K/V) × 2(kv_heads) × 256(head_dim) × 2(bf16) = 4 MiB`
（`platform.py:1144` / `:1319` / `:1398`）。这三个数是 **Qwen3.6 profile 的声明**
（`profiles/qwen3_6.py` 的 `BlockSizePolicy(hybrid_align=2048)` 与 `HybridKind.page_size_padded`），
`platform.py` 只负责校验与写入；换模型时改的是 profile，不是这三处。KV 池是 **block-first**
`(num_blocks, 2, block_size, heads, dim)`，物理顺序 `get_kv_cache_stride_order() == (1, 0, 2, 3, 4)`
= `[2][blocks][page][heads][dim]`（`attn/backend.py:174-246`），两半各自连续、子页视图零拷贝。

**已接线**：decode plan 预烧在 `compile_or_warm_up_model` 内、`capture_model()` **之前**
（`worker.py:1796-1876`，调用点 `:1876`）；`KVBlockZeroer` 是**故意跳过**的（§3.7），不是待办。

**运维面已存在**：`scripts/serve_qwen36.py`（一条命令起服，另有 `--plan-only` / `--doctor`，
`main` 在 `:802`）与 `tileinfer doctor`（`tileinfer/vllm/doctor.py:719`，console script 在
`pyproject.toml:41`）。

**已实现、未上机**：worker 四个 hook、MoE 三个算子、GDN decode 的 state 更新算子（`fused_recurrent_gated_delta_rule_packed_decode` /
`fused_sigmoid_gating_delta_rule_update`，由 `ops/install.py` 绑定到本仓库 torch 端口；vLLM 的 mamba SSU 后端对 GDN 层不参与）在真实 vLLM 进程里能否
绑定/跑通未在硬件上确认；图捕获未验证因而平台**强制 `enforce_eager=True`**。
**从未在硬件上跑过**：以上全部只在 CPU + fake vLLM + fp64 参考上验证过，`torch_npu` 侧的每一个符号
（graph 原语、`get_device_properties`、`MemorySnapshot` 等价物、`dispatch_key="NPU"`）都还是未核实状态。

---

## 1. 证据等级表（本文的核心）

四级定义：

* **A 硬证据** —— 语义/签名从本地 vLLM 0.23 源码逐行转录（`runtime/vllm-api/`），行号可跳转核对。
* **B 同源修订** —— 函数签名与 dtype 契约来自 A 级源（`fla/ops/chunk.py` 的实参表），但 **kernel 体内的
  数学**来自 sglang 0.5.10 vendored 的同一份上游 FLA（`SG-*`），因为 vLLM 版那四个文件本地没有。
  B 级的行号指向 `sglang/srt/layers/attention/fla/`，**不是** `runtime/vllm-api/`。
* **C 重建** —— 任何来源都没有该算子的实现，只能从调用点反推重建。
* **D 推断/未绑定** —— 没有任何本地证据，行为是选出来的/容错双实现的，必须在硬件上回答。

| # | 子系统 | 文件 | 等级 | 证据（文件:行） | CPU 验证 |
|---|---|---|---|---|---|
| 1 | `AttentionBackend` / `AttentionImpl` / `CommonAttentionMetadata` | `tileinfer/vllm/attn/backend.py`、`impl.py`、`metadata.py` | **A** | `vllm__v1__attention__backend.py:55`（ABC）、`:72-96`（4 个 abstractmethod）、`:58-66`、`:98-349`（非抽象钩子）、`:359-500`（CAM 字段）、`:506-520`（`AttentionCGSupport`）、`:589-607`（`build`）、`:770-794`（`__init__`）、`:796-809`（`forward`）；`vllm__v1__attention__backends__utils.py:44`（`PAD_SLOT_ID=-1`）、`:436-576`（decode/prefill 切分）；`registry.py:103-144` | `tests/test_attn_contract.py` **59 passed** |
| 2 | `Platform` ABC 与平台选择 | `tileinfer/vllm/platform.py` | **A** | `vllm__platforms__interface.py`（1057 行）：ABC 共 **78** 个方法/属性（75 个公开），本文件实现 **10**、`must override` **6**、`unverified` **3**、`default is fine` **56**；`vllm__platforms____init__.py:217-222`（非 `None` 返回值即类路径）、`:231-235`（两个 OOT 插件是硬错误）、`:237`（胜出者被**第二次**调用）、`:262-281`（`resolve_obj_by_qualname` + 无参实例化）。**本轮新增核实**：`worker_cls` 机制（`parallel.py:248`、`worker_base.py:249-253`、`config/vllm.py:1335`、`cpu.py:153-154`）与模型架构接受（`interface.py:708-718`、`registry.py:945`），详见 §7.7 | `tests/test_platform_contract.py` **122 passed** |
| 3 | 两个 entry point | `tileinfer/vllm/plugin.py` + `pyproject.toml` | **A** | `pyproject.toml` 的 `[project.entry-points."vllm.platform_plugins"] tileinfer = "tileinfer.vllm.platform:register"` 与 `[project.entry-points."vllm.general_plugins"] tileinfer = "tileinfer.vllm.plugin:install"`；调用点见 `vllm__platforms____init__.py:217-281` | `tests/test_ops_install.py:841-854` 钉住 `install_ops()` 必须是最后一步；**无专用测试文件** |
| 4 | `CustomOp` 分派阶梯与「必须双写」 | `tileinfer/vllm/ops/install.py` | **A** | `vllm__model_executor__custom_op.py:174-207`（rocm→hip / cpu / tpu / xpu / **out_of_tree→`forward_oot`** / else cuda）、`:169-172`（`forward_oot`→`forward_native`）、`:138-144`（`forward_native` 抛）；`qwen_gdn_linear_attn.py:290-311`（`ChunkGatedDeltaRule.__init__` **重钉** `_forward_method`）、`:24-49`（按名 import 五个 FLA 算子）、`:1344/1365/1378/1428/1457/1484/1542/1684`（当模块全局调用） | `tests/test_ops_install.py` **19 passed** |
| 5 | GDN chunk 链的**调用顺序与 dtype 改写** | `tileinfer/vllm/ops/fla/chunk.py` | **A** | `vllm__model_executor__layers__fla__ops__chunk.py:37-86`：`g=chunk_local_cumsum`(:37) → `A=chunk_scaled_dot_kkt_fwd(...,output_dtype=fp32)`(:41-48) → `A=solve_tril(...,output_dtype=k.dtype)`(:49-51) → `w,u=recompute_w_u_fwd(k,v,beta,A,g_cumsum=g)`(:52-60) → `h,v_new,final_state=chunk_gated_delta_rule_fwd_h(...)`(:61-71) → `o=chunk_fwd_o(q,k,**v=v_new**,h,g,scale,…)`(:72-82)；`:75` 的 `v=v_new`；`:83-86` 的 `SUPPRESS_LEVEL` 分支；`:129-133` `core_attn_out` 只能推理期原地写 | `tests/test_gdn_chunk_full.py` **19 passed** |
| 6 | `causal_conv1d` 两个入口（**两套 token 轴**） | `tileinfer/vllm/ops/fla/causal_conv1d.py` | **A** | `vllm__model_executor__layers__mamba__ops__causal_conv1d.py`（1239 行）：`:18`（`x_ptr # (dim, cu_seqlen)`）、`:563`（`is_channel_last = (stride(0)==1) & (stride(1)>1)`）、`:620`（`assert is_channel_last`）、`:1085-1090`（update 接受 `[batch,dim]`/`[batch,dim,seqlen]`/`[num_tokens,dim]`）、`:1171-1172`（`stride_x_token, stride_x_dim`）；调用点 `qwen_gdn_linear_attn.py:1365`（prefill，传 `mixed_qkv.T`）、`:1378`/`:1674`（decode） | `tests/test_gdn_causal_conv1d.py` **32 passed** |
| 7 | `solve_tril` / `l2norm_fwd` / `chunk_local_cumsum` 语义 | `tileinfer/vllm/ops/fla/solve_tril.py`、`ops/gdn.py` | **A** | `fla_solve_tril.py:513-530`（docstring：inverse of `I + A`，`A.triu()==0`）、`:64-66`（`m_A` 严格下三角 / `m_I` 单位阵）、`:82`（`b_A = -tl.where(m_A,b_A,0)`）、`:89`（`b_A += m_I`）、`:539`（`zeros_like`）；`fla_l2norm.py` 的 `l2norm_fwd_kernel1`（`b_rstd = 1/sqrt(b_var + eps)`，eps 加在**平方和**上）；`fla_cumsum.py:243-252`（`chunk_local_cumsum` 上游签名：`chunk_indices`/`head_first`/`output_dtype`）、`:27-71`（chunk 边界重启的 kernel） | `tests/test_gdn_solve_tril.py` **30** + `tests/test_gdn_ops.py` **12** passed |
| 8 | decode 路径：gating 与 recurrent 两个入口 | `tileinfer/vllm/ops/fla/gating.py`、`recurrent.py` | **A** | `fla_fused_sigmoid_gating.py`（279 行）：`:129-133`（`x=a+dt_bias` → softplus → `g=-exp(A_log)*softplus`）、`:136`（`beta=sigmoid(b)`）、`:141`（q 缩放、k 不缩放）、`:144`（**先**衰减）、`:148-151`（对已衰减状态做 delta、**后**读状态出 `o`）、`:189-190`（`beta=1.0`/`threshold=20.0` 是 **softplus** 参数）；`fla_fused_recurrent.py`（619 行）：`:122-149`（序列形逐 token 递推）、`:339-478`（packed decode，grid `(NV, B*HV)` 在 `:449`，无 token 循环）、`:516-619`（序列形 launcher） | `tests/test_gdn_gating_delta_rule.py` **30** + `tests/test_gdn_recurrent.py` **45** passed |
| 9 | 图捕获判定（两条 CUDA 语句 + 桥接） | `tileinfer/vllm/graphs.py` | **A**（vLLM 侧）/ **D**（torch_npu 侧） | `vllm__compilation__cuda_graph.py`（361 行）：`:283`（`torch.cuda.CUDAGraph()`）、`:313-317`（`with torch.cuda.graph(...)`）、`:200`/`:303-306`（`get_global_graph_pool`/`set_graph_pool_id`）、`:233-254`（forward-context 分派）；结论「只有 2 处语句碰 CUDA，其余设备无关」= `graphs.CUDA_CALL_SITES == [283, 313]` | `tests/test_platform_contract.py` 内的 graph 段（含 `CaptureState` 的 7 个 violation code 与 5 条 `UNVERIFIED_SYMBOLS`） |
| 10 | KV 写与 per-step 元数据重建 | `tileinfer/vllm/attn/kv_cache.py`、`serving/step.py` | **A**（语义）/ 自定契约 | `vllm__v1__attention__backend.py:464-469`（`seq_lens - query_lens` ⇒ `seq_lens` **已含**本步 token，正是 `refresh_step` 需要的语义）、`:471-500`（`unpadded`）、`utils.py:44`（`PAD_SLOT_ID=-1`）；dense→ragged 的具体实现是本仓库自己的契约（trash slot 不变量） | `tests/test_kv_cache_write.py` **15** + `tests/test_serving_step.py` **7** passed |
| 11 | **四个 chunk 算子的 kernel 体数学** | `ops/fla/kkt.py`、`wu.py`、`delta_h.py`、`chunk_o.py` | **B** | 签名/轴序/dtype 来自 `fla/ops/chunk.py`（A 级，见 #5）；**数学**来自 sglang vendored FLA：`SG-chunk_scaled_dot_kkt.py:53-86`、`SG-wy_fast.py:55-108`、`SG-chunk_delta_h.py:123-251`、`SG-chunk_o.py:76-123`。`runtime/vllm-api/` 里**没有** `chunk_scaled_dot_kkt.py` / `wy_fast.py` / `chunk_delta_h.py` / `chunk_o.py` 四个文件（36 个文件里只有 `fla__ops__chunk.py` 与 `fla_cumsum/l2norm/solve_tril/fused_recurrent/fused_sigmoid_gating`），**vLLM 版 kernel 体本轮未核实** | `tests/test_gdn_kkt_wu.py` **35**、`tests/test_gdn_delta_h_chunk_o.py` **37**、`tests/test_gdn_wu_reference.py` **8** passed |
| 12 | `fused_post_conv_prep`（prefill 的 conv 后拆分/归一/门控） | `ops/fla/post_conv.py` | **C**（记录在案）→ 实为 **A**（见 §7.3） | 原记录：vLLM 与 sglang 两边都没有该文件，**从调用点反推重建**；调用点 `qwen_gdn_linear_attn.py:1428-1439`（prefill，`apply_l2norm=True`/`output_g_exp=False`）、`:1113-1124`（warmup）、`:27-31`（按名 import）；打包布局反推自 `rearrange_mixed_qkv` `:821-840`、`conv_dim` `:466`。本轮实测：源文件其实**已在** `runtime/vllm-api/vllm__model_executor__layers__fla__ops__fused_gdn_prefill_post_conv.py`（7,935 B，`def` 在 `:152`）。**若 GDN prefill 数值不对，第一个重查 `beta` 的容器 dtype（端口 bf16 vs 上游 fp32，见 §7.3），第二个重查 `apply_l2norm=False` / `output_g_exp=True` 这两个无上游调用点的分支** | `tests/test_gdn_post_conv.py` **17 passed**（与自身契约对拍，**不是**与上游逐位对拍） |
| 13 | MoE 三个算子 | `tileinfer/vllm/ops/moe.py` + `ops/families/gdn_moe.py`（表；`install.py` 只执行） | **D**（记录在案）→ 实为 **A**（见 §7.2） | 原记录：本地无 `fused_moe` 模块，无法判定 vLLM 调 `torch.ops._C.*` 还是 `_custom_ops` 包装；family 表因此给每个算子一个**候选名列表**（`targets`），绑不上就记 `unbound` 并继续（不抛、不静默）。本轮实测三条路径全部可定位（见 §7.2） | `tests/test_moe_ops.py` **36 passed**（CPU oracle，含 256 专家） |
| 14 | `g` / `beta` 的容器 dtype | `ops/fla/post_conv.py:189-194`（N3/N4） | **D**（记录在案）→ 已核实为 **fp32**（见 §7.3） | 原记录：`g` 的 fp32 是「两条独立惯例推出」的，`beta` 在 vLLM 是 `b.dtype`、在 sglang 是 fp32；本轮实测源 `:202-203` 两者**都** `torch.float32` | 由 #12 的 17 个用例间接覆盖 |
| 15 | worker 四个 hook（已实现） | `tileinfer/vllm/worker.py` | **D**（实现层；硬件未验证） | 为什么不能继承：`vllm__v1__worker__gpu_worker.py:249`（`init_device`，`:251` 判 `"cuda"`、`:285` 建 `torch.device(f"cuda:{local_rank}")`、`:319-320` else 抛）、`:371`（`determine_available_memory`，走 `torch.accelerator.memory_stats` `:412`）、`:562`（`initialize_from_config`，CuMem 池 `:577`）、`:591`（`compile_or_warm_up_model`，`:630` `capture_model`、`:626` Triton warmup）。四个 hook 已按这些调用点实现：`worker.py:825`（`init_device`）、`:1510`（`determine_available_memory`）、`:1689`（`initialize_from_config`）、`:1796`（`compile_or_warm_up_model`），每个都带 `HARDWARE:` 标记与首个诊断命令 | `tests/test_platform_contract.py:2025-2034` 断言「不是旧的 raise 骨架」；行为断言在 `tests/test_worker_contract.py`（签名 / dtype / 内存会计 / KV 路径 / 捕获门） |
| 16 | `dispatch_key = "NPU"` | `platform.py:221`、`:891`、`UNVERIFIED_NOTES["dispatch_key"]`（`:365`） | **D，仍未核实** | 原文：「'NPU' is the torch_npu PrivateUse1 dispatch-key spelling; **no local file reads the attribute**, so it cannot be confirmed offline.」本机无 torch_npu 源码/二进制，无法证伪 | 无（`tests/test_platform_contract.py` 只钉住字符串值） |
| 17 | 图桥依赖的 5 个 torch_npu / vLLM 符号 | `graphs.py:140-164`（`UNVERIFIED_SYMBOLS`） | **D，仍未核实** | 5 条：`vllm.utils.torch_utils.current_stream`（`cuda_graph.py:27,316`）、`...pynccl_allocator.set_graph_pool_id`（`:304,306`）、`...monitor.validate_cudagraph_capturing_enabled`（`:277`）、`...offloader.get_offloader`（`:310,324,359`）、`torch.npu.CUDAGraph / torch.npu.graph / torch.npu.graph_pool_handle`（`interface.py:887-915` 走 `__getattr__`）。`capture_path_ready()` 只能检查 torch 侧的那几条 | `tests/test_platform_contract.py:1018` 断言「恰好 5 条」，并断言出现这些符号名 ≠ 失败 |
| 18 | 混合 KV：能力位 + 页几何（10 attention + 30 GDN） | `platform.py:1144` / `:1319` / `:1398`（值是 `profiles/qwen3_6.py` 的声明） | **A** | ABC 默认 False（`interface.py:945-950`）；False 会让 vLLM 0.23 关掉 hybrid KV manager（v0.23.0 `config/vllm.py:1453-1455`）并拒绝 attention+mamba spec 混编（v0.23.0 `v1/core/kv_cache_utils.py:1421-1428`；这两个文件**不在**本地子集，只作因果引用）；spec 由模型给出（`gpu_worker.py:547-548` → `gpu_model_runner.py:7440-7470`）；10/30 来自 `runtime/test-model-qwen36/config.json` 的 40 层 `layer_types`（每 4 层 1 个 `full_attention`）；mamba 状态页 = 4 MiB，由 `align_mamba_state_pages` 写入 `mamba_block_size` / `mamba_page_size_padded` | `tests/test_platform_contract.py` hybrid 段 + `tests/test_model_profiles.py` |
| 19 | KV 池布局：block-first + 声明物理顺序 | `attn/backend.py:174-246` | **A** | `get_kv_cache_shape` = `(num_blocks, 2, block_size, heads, dim)`（`:174-211`，值来自 profile 的 `kv_shape`）；`get_kv_cache_stride_order` = `(1, 0, 2, 3, 4)`（`:213-246`）；runner 按 stride order 分配、再 permute 回逻辑形（`gpu_model_runner.py:7093-7110`），`block_dim == 0` 时跳过 hybrid 重写（`:7190-7191`） | `tests/test_attn_contract.py`、`tests/test_kv_cache_write.py`、`tests/test_model_profiles.py` |
| 20 | decode plan 预烧接线 | `worker.py:1796-1876` | **A** | 顺序照 `gpu_worker.py:591-750`；`_dummy_run` 之后、`capture_model()` **之前**调用 `_prewarm_decode_plans()`（`worker.py:1876`，定义在 `:1351`）；bucket 集由 `max_num_seqs` / `max_num_batched_tokens` / `max_model_len` / `block_size` 经 `serving/planner` 推出，plan 由 impl 自己的缓存持有（`attn/impl.py:745`） | `tests/test_platform_contract.py`、`tests/test_boot_rehearsal.py` |
| 21 | 两个既存静默 bug（已修） | `serving/buckets.py:219-259`、`attn/backend.py:177-191` / `buckets.py:262-305` | **A** | ① `capacity_view` 重建视图必须携带 `flat.storage_offset()`，否则融合池的 V 半会别名回 K 半（`buckets.py:228-231`）：内核把 K 当 V 再读一遍，**全程无报错**；② `block_dim == 1` 时 vLLM 的 hybrid 重写把同一 shape 改成按块 K/V 交错（`gpu_model_runner.py:7196-7199`），`pool_sub_view` 没有平坦的 `[pages,128,heads,dim]` 视图，只能复制；平台因此声明 `block_dim == 0` + stride order（`backend.py:182-191`、`buckets.py:269-288`） | `tests/test_serving_buckets.py`、`tests/test_kv_cache_write.py` |
| 22 | 起服脚本与 `tileinfer doctor` | `scripts/serve_qwen36.py`、`tileinfer/vllm/doctor.py` | **A** | console script 在 `pyproject.toml:35-41`（`tileinfer = tileinfer.vllm.doctor:main`）；`serve_qwen36.py` 校验 `config.json`、推出预烧计划、设 env 后 exec `vllm serve --attention-backend CUSTOM`（`main` `:802`）；`doctor.py` 把 `hardware-checklist.md` 步骤 1-5 变成 PASS/FAIL/SKIP（`main` `:719`） | `tests/test_doctor.py` |
| 23 | **worker 选择机制（SUSPECT 1）** | `platform.py:_select_worker_class`（`:1510`） | **A** | `ParallelConfig.worker_cls: str = "auto"`（`vllm__config__parallel.py:248-250`）；唯一消费者 `WorkerWrapperBase.init_worker` 用 `resolve_obj_by_qualname` 解析（`vllm__v1__worker__worker_base.py:249-253`），非字符串直接拒绝（`:254-259`）；hook 在 `VllmConfig.__post_init__` 里被调用（`vllm__config__vllm.py:1335`），早于 executor 构造 wrapper（`vllm__v1__executor__uniproc_executor.py:48`）；in-tree 平台在 `check_and_update_config` 里替换 `"auto"`（`vllm__platforms__cpu.py:153-154`）。**0.23 没有 `Platform.get_worker_cls`**（`interface.py:105-1052` 无此名） | `tests/test_platform_contract.py` 的 worker_cls 源契约用例 |
| 24 | **模型架构接受（SUSPECT 2）** | `platform.py` 不 override `verify_model_arch` | **A** | ABC 体是 `pass`（`interface.py:708-718`）；调用点在 import 模型类之前（`registry.py:945`，`_try_load_model_cls` `:939-950`）；接受由 `_VLLM_MODELS` 表决定（`:693-704`），`Qwen3_5ForConditionalGeneration` / `Qwen3_5MoeForConditionalGeneration` 映射到 `qwen3_5`（`:569-573`）；裸 `...ForCausalLM` 由 `_normalize_arch`（`:1134-1158`）+ `model.py:1912-1929` 归一；`qwen3_5.py` **不自注册**（只有 class 定义 `:529` / `:533` / `:554` / `:771`），所以没有 import 期副作用可依赖 | 同上（registry 源契约用例） |
| 25 | **三个扩展点（model profile / op family / candidate）** | `profiles/base.py`、`profiles/qwen3_6.py`、`ops/families/{__init__,gdn_moe}.py`、`ops/candidates.py` | **A**（值从重构前的常量与 `if` 逐项转录） | profile：archs / head 128·256 / `group_min=2` / 内核页 128 + hybrid 页 2048 / block-first KV 形状 + stride order / GDN `mamba_types` 与 state update / 13 条拒绝项（两套文案）；family：12 个 vLLM 名字 + `targets`/`aliases`（双写）/ `abi_ok=False` 的 `chunk_local_cumsum`；candidate：门顺序 = `TILEINFER_OPS_CANDIDATES` 熔断 → `enabled_by_default` → `measured_ratio >= 1.0` → predicate，baseline 恒在（0.86x 的 MoE grouped-GEMM 记录见 `docs/status.md` E19/H7） | `tests/test_model_profiles.py`（**8**）、`tests/test_op_candidates.py`（**25**）、`tests/test_ops_install.py`（**19**）、`tests/test_doctor.py`（**40**，含候选行） |

**这张表的时效性**：第 12–14 行的「C/D」是**代码与旧文档里的记录**；上一轮实测（§7）已把它们的源码前提
推翻——`runtime/vllm-api/` 现在是 **36 个文件**（不是旧 docstring 写的 17/24 个）。本轮为核实两条启动路径
新增 12 个：`config/{vllm,model,parallel,model_arch}.py`、`platforms/cpu.py`、`plugins/__init__.py`、
`v1/executor/{abstract,uniproc_executor}.py`、`v1/engine/core.py`、`v1/worker/worker_base.py`、
`model_executor/models/{registry,__init__}.py`。**上机第一件事仍不是重新抓源码，而是按 §7 判读**；真正的
B 级只剩第 11 行的四个 kernel 体。

---

## 2. 模块清单（40 个文件：`tileinfer/serving/` 5 个 + `tileinfer/vllm/` 35 个）

> **本轮计数口径**：`tileinfer/vllm/` 从 29 涨到 **35**，多出的 6 个全部属于三个扩展点
> （`profiles/` 3 个、`ops/families/` 2 个、`ops/candidates.py` 1 个）；`serving/` 不变。
> 这三个点怎么用、哪些代码不能动，见 [`extending.md`](extending.md)。

> **`ops` 与 `kernels` 不是一回事（读本节前先记住）**：`tileinfer/vllm/ops/**` 是 **vLLM 算子 ABI 层**
> —— 每个模块给出与上游同名的 **torch** 实现，correctness-first，由 `ops/install.py` 绑定到 vLLM 的调用名，
> 在 CPU 上就能与 fp64 参考对拍；`tileinfer/kernels/**` 是 **TileLang 内核层** —— 设备编译
> （`torch_npu` + TileLang Ascend 工具链）的 perf 实现，模块级 `import tilelang` 就会失败。两层刻意解耦：
> ops 可以先没有对应内核（用 torch 顶住），kernels 也可以暂时没有 vLLM 调用点。看到 `ops/fla/chunk.py`
> 不要去里面找 TileLang；看到 `kernels/gdn/` 也不要指望它实现 vLLM ABI。`docs/plan-vllm-native-backend.md`
> §W2 的目录树把两者分列，正是这个意思。

### 2.1 `tileinfer/serving/`（引擎无关层，**禁止** import `vllm` / `vllm_ascend` / `torch_npu`）

| 文件 | 一行说明 | 测试文件（本轮计数） | 状态 |
|---|---|---|---|
| `serving/__init__.py` | 包导出：bucket/页算术、`PlanCache`/`prewarm`、`StepBuffers`/`refresh_step`、`Counters`；并声明依赖规则 | 被 `tests/test_serving_buckets.py`、`test_serving_step.py`、`test_plan_cache.py` 间接导入 | 完成 |
| `serving/buckets.py` | `KERNEL_PAGE_SIZE=128` 的子页算术：`bucket_for`/`buckets_for`（batch 向上取桶）、`expand_page_table`（`page*k+i`）、`pool_sub_view`（零拷贝子页视图）、`last_page_len`/`declared_pool_cap` | `tests/test_serving_buckets.py`（**35**） | 完成 |
| `serving/counters.py` | 进程级线程安全计数器：每次变更走锁，`snapshot()` 返回 dict 副本供进程外探针序列化 | `tests/test_serving_buckets.py:167` 有 `Counters` 用例；**无专用测试文件**（覆盖薄） | 完成 |
| `serving/planner.py` | plan 缓存与预烧：`plan_key`（用**声明**池容量而非精确池，避免页数漂移击穿缓存）、`PlanCache` 单飞（`_building` 集合）、`prewarm`、后台编译兜底 | `tests/test_plan_cache.py`（**11**） | 完成 |
| `serving/step.py` | 每步 dense 页表 → ragged 三元组的**设备端**重建（`kv_indptr`/`kv_indices`/`kv_last_page_len`），带 trash-slot 不变量；全定形、原地、无 host 同步 ⇒ 可入图 | `tests/test_serving_step.py`（**7**） | 完成 |

### 2.2 `tileinfer/vllm/`（唯一允许 import vLLM 的包）

| 文件 | 一行说明 | 测试文件（本轮计数） | 状态 |
|---|---|---|---|
| `vllm/__init__.py` | 包说明 + `__all__ = []`；**导入本包不得有副作用、也不 import vLLM** | 无 | 完成（纯文档） |
| `vllm/platform.py` | `TileInferPlatform`：ABC 75 个公开成员的清单（**10 实现 / 6 must-override / 3 unverified / 56 默认够用**）；`check_and_update_config` 的固定顺序是**先解析并激活模型 profile**（`profile_for_config` → `set_active_profile`），再按该 profile 的 `unsupported_features` 拒绝、按 `block_size_policy` 校验页（Qwen3.6 = 2048 + state 页对齐）、按 `"auto"` 哨兵写 `worker_cls`；`HYBRID_BLOCK_SIZE` / `SELECTOR_REJECT_*` 只是**启动 profile 的镜像**（`BOOT_PROFILE`，给日志与测试读），不是第二个真值；`_force_eager_until_capture_lands`、6 条 `UNVERIFIED_NOTES` | `tests/test_platform_contract.py`（**122**，含 graphs/worker/arch 段）+ `tests/test_model_profiles.py`（profile 驱动的页/拒绝段） | CPU 绿；**未上机** |
| `vllm/plugin.py` | `install()` 四步固定顺序：patch 注册表 → attention 后端注册（`AttentionBackendEnum.CUSTOM`）→ 图桥（仅当 `capture_path_ready`）→ `install_ops()`（这一步现在还会解析 profile 的 `op_family`）；末尾一行 INFO banner；`status()` 供 doctor 打印 | `tests/test_ops_install.py:841-854`（顺序 + `plugin._install_ops()` 不抛）；**无专用测试文件** | CPU 绿（部分） |
| `vllm/worker.py` | `NPUWorker(Worker)` **已实现、硬件未验证**：`WORKER_CLASS_PATH`、`HARDWARE_CHECKLIST`（6 条，`:208`）、四个 hook 按本地 `gpu_worker.py` / `gpu_model_runner.py` 的调用点实现（`worker.py:825` / `:1510` / `:1689` / `:1796`）并附「上机要核实什么」；mamba 路径判定与 `block_dim` 读 `active_profile()`（`mamba_state_path` `:299`）；`TileInferModelRunner` override `_init_device_properties`，并**故意拒绝** `KVBlockZeroer`（`_init_kv_zero_meta`，`:743`，调用点 `:1785`）；`compile_or_warm_up_model`（`:1796`）在捕获前调用 `_prewarm_decode_plans`（`:1351`，调用点 `:1876`） | `tests/test_platform_contract.py` + `tests/test_worker_contract.py`（行为断言）+ `tests/test_model_profiles.py`（mamba 门用 profile 的 hybrid kind） | **已实现（CPU 契约验证）；未上机** |
| `vllm/doctor.py` | `tileinfer doctor`：把 `hardware-checklist.md` 步骤 1-5 变成 PASS/FAIL/SKIP + 修复提示，`--json` / `--help`；`ops.binding` 行同时打印每个 op 的**候选门状态**（`candidates: <op>[<name>=<status>]`，含「为什么被关」），候选表是本仓库数据，所以没有 vLLM 时该行仍带这段（`_candidate_segment`，不改退出码）；console script 在 `pyproject.toml:41` | `tests/test_doctor.py`（**40**） | CPU 绿 |
| `vllm/graphs.py` | 图捕获判定（结论：branch 2「可打补丁」，只有 2 处 CUDA 语句）+ `wrap()` + `CaptureState`/7 个 violation code + 5 条 `UNVERIFIED_SYMBOLS` | `tests/test_platform_contract.py` 的 graph 段 | CPU 绿；torch_npu 侧未核实 |
| `vllm/profiles/__init__.py` | 模型 profile 包：导入即注册发货 profile（今天只有 `qwen3_6`）；导出 `PROFILES` / `profile_for` / `active_profile` / `set_active_profile` 等。新增 profile 要在这里加**一行 import**（放在 `qwen3_6` 之后：`default_profile()` = 第一个注册的） | 被 `tests/test_model_profiles.py` 直接使用 | 完成 |
| `vllm/profiles/base.py` | **扩展点 A**：`ModelProfile`（frozen）及其值对象 `BlockSizePolicy` / `KvShape` / `HybridKind` / `UnsupportedFeature`；注册表（`register_profile` / `unregister_profile` / `known_profiles`）、解析（`profile_for`：按 archs 匹配 + hints 收窄，未知架构抛并列出已知 profile）与进程级激活（`active_profile` / `set_active_profile`）。**不含**内核页、dtype 集合、worker 机制 | `tests/test_model_profiles.py`（**8**，含 dummy profile 驱动平台/backend/worker/install 的 seam 用例） | 完成 |
| `vllm/profiles/qwen3_6.py` | Qwen3.6 的 profile：4 个 `Qwen3_5*` arch、head 128/256、`group_min=2`、内核页 128 + hybrid 2048、block-first KV 形状 `(blocks,2,page,heads,dim)` + stride order `(1,0,2,3,4)`、GDN hybrid（`GDN_ATTN` / SSU `MAMBA1/2` / 4 MiB state 页 / packed-decode state update）、`op_family="gdn-moe"`、13 条拒绝项（8 个 selector/plain_selector + 1 个 `attn_type` + 4 个 config；`use_mla` 在 selector 与 config 各一条，各带两套文案 + 证据） | 同上 | 完成 |
| `vllm/attn/__init__.py` | 空导出（仅说明） | 无 | 完成 |
| `vllm/attn/backend.py` | `TileInferAttentionBackend`：**形状/顺序/块策略/head 宽/拒绝清单全部读 `active_profile()`**——融合 KV 形状来自 profile 的 `kv_shape`（Qwen3.6 = block-first `(num_blocks, 2, block_size, kv_heads, head_size)`，`block_dim == 0`）、物理顺序来自 `kv_shape.stride_order`（`(1,0,2,3,4)`）、块策略是 `MultipleOf(kernel_page_size)`（Qwen3.6 = 128）、`validate_configuration` 的拒绝项来自 `unsupported_features`（两套文案）、`check_head_layout`（`:376`，GQA `group_min` 校验，因为 `validate_configuration` 没有 head 数参数）、`register_with_vllm()`（不在 import 时注册） | `tests/test_attn_contract.py`（**59**）+ `tests/test_model_profiles.py`（dummy profile 驱动形状/顺序/head/拒绝） | CPU 绿 |
| `vllm/attn/impl.py` | 一层 attention 的 forward：`write_kv_cache_fused` → `pool_sub_view` → `refresh_step` → plan 查找 → `BatchAttention.run` → `copy_` 回；5 个降级计数器（`prefill_reference` / `plan_not_ready` / `plan_failed` / `kernel_backend_unavailable` / `kernel_path_unavailable`），无 `except: pass` | `tests/test_attn_contract.py`（`test_attn_contract.py:294` 从本模块 import 并断言签名/拒绝策略）；**无专用测试文件** | CPU 绿（reference 路径；kernel 路径未上机） |
| `vllm/attn/metadata.py` | `TileInferAttentionMetadata` + builder：**引用不拷贝**引擎张量（断言 identity）、三个整数 + bucket、decode/prefill 只看 `max_query_len`、声明 `UNIFORM_SINGLE_TOKEN_DECODE` | `tests/test_attn_contract.py`（`:301`/`:422` 从本模块 import）；**无专用测试文件**（`tests/test_metadata.py` 测的是 `tileinfer.metadata`，不是本文件） | CPU 绿 |
| `vllm/attn/kv_cache.py` | NHD paged 池的 KV 写入（`write_kv_cache` / `write_kv_cache_fused` / `flat_pool` / `check_slot_mapping`）；`slot_mapping` 要求 int64、拒绝在捕获区内做 cast | `tests/test_kv_cache_write.py`（**15**） | 完成 |
| `vllm/compat/__init__.py` | 说明：`patches` 注册表 + `audit` 静态扫描（`audit` 模块**尚未落地**，只有文档提及） | 无 | 部分（audit 未实现） |
| `vllm/compat/patches.py` | `Patch` 记录（why / since / remove_when）+ `install_all`（目标消失只 skip+log，不抛）+ `InstallReport` | `tests/test_patch_registry.py` | 完成 |
| `vllm/ops/__init__.py` | 导出 `gdn`/`moe` 的 **7** 个公共名（`l2norm_fwd`、`chunk_local_cumsum`、`topk_softmax`、`moe_align_block_size`、`moe_align_block_size_block_first`、`fused_experts`、`silu_and_mul`）；故意**不**导入 `install` | 间接 | 完成 |
| `vllm/ops/gdn.py` | 两个无状态叶子：`l2norm_fwd`（`x/sqrt(sum(x²)+eps)`，eps 加**平方和**）、`chunk_local_cumsum`（chunk 内前缀和 + varlen 重启，token-major `[T,...]`） | `tests/test_gdn_ops.py`（**12**） | 完成 |
| `vllm/ops/moe.py` | `topk_softmax` / `moe_align_block_size`（主签名 `(topk_ids, num_experts, block_size, …)` + 另一种顺序的别名）/ `fused_experts`（含共享专家的 TileInfer 扩展）/ `silu_and_mul` | `tests/test_moe_ops.py`（**36**） | 完成（CPU oracle） |
| `vllm/ops/install.py` | **机制，不再持有算子表**：`install_ops()` 解析当前 profile 的 `op_family`（`_profile_family_names` → `family_for`），逐行 plan（`_plan` 五态：switch 关 / `abi_ok=False` / 无实现 / 目标解析失败 / planned）+ `bound/unbound/disabled` 三态报告 + 每个算子**双写**（定义模块 + `aliases` 的消费方命名空间）；`_installed_impl` 在有 `CandidateSet` 时绑 `dispatched()`（`impl` 仍是 baseline）；`candidate_report()` 供 doctor/探针打印每个 op 的候选门状态 | `tests/test_ops_install.py`（**19**）+ `tests/test_model_profiles.py:374-392`（follows the profile） | CPU 绿（fake vllm） |
| `vllm/ops/families/__init__.py` | **扩展点 B 的注册表**：`FAMILIES` / `register_family(name, module)`（校验 `required_names`+`bind` 可调用）/ `family_for(name)`（先查表，查不到就 lazy import `tileinfer.vllm.ops.families.<name 的 - 换成 _>`；模块有 `NAME` 就按它注册）。文件底部显式注册发货的 `gdn_moe` | 被 `tests/test_model_profiles.py` 的 dummy family 用例直接驱动 | 完成 |
| `vllm/ops/families/gdn_moe.py` | **扩展点 B 的实例**：`gdn-moe` 族的 vLLM 接管表（12 个名字，`REQUIRED_NAMES` 与表序一致），每行带 `family`（开关族 `gdn`/`moe`/`dense`）、`targets`/`aliases`、`why`/`remove_when`/`upstream_ref`；`chunk_local_cumsum` 以 `abi_ok=False` 只报告；每行经 `_with_candidates` 挂一个 `CandidateSet`（baseline = torch 端口），并为 `fused_experts` 登记一个**默认关**的等价 oracle 候选 | `tests/test_ops_install.py` + `tests/test_op_candidates.py`（**25**：family 名字表、每行都有 set、dispatcher 与 baseline 逐字节一致、0.86x 记录不 eligible） | CPU 绿 |
| `vllm/ops/candidates.py` | **扩展点 C 的机制**：`Candidate`（impl + predicate + `enabled_by_default` + `measured_ratio`）/ `CallSpec`（调用实参作为数据，predicate 只能读 shape/dtype/连续性）/ `CandidateSet`（baseline 恒在、`add` 顺序即优先级、`select` 零分配、`dispatched()` 薄封装、`describe()` 带原因）；门顺序 = `TILEINFER_OPS_CANDIDATES` 熔断 → flag → `measured_ratio >= 1.0` → predicate；env **只能关**，拼错 fail-safe 成 off | `tests/test_op_candidates.py`（**25**） | 完成 |
| `vllm/ops/switch.py` | `TILEINFER_OPS` 族开关：默认 `gdn,moe` 开、`dense` 关；`ops_enabled` 遇未知名**抛**（遗留契约），`enabled_families` 是容忍版。**注意 `OpBinding.family` 填的是这三个开关族名，不是 `ModelProfile.op_family` 的模块名**（填错 → 该行永远 `disabled`） | **有直接测试**：`tests/test_ops_install.py`（`:48` import 本模块；`test_switch_defaults_are_the_families_the_model_needs` / `test_switch_honours_the_environment` / `test_switch_raises_on_a_typo_but_install_does_not` / `test_install_skips_the_disabled_family`）。但**安装器不用** `enabled_families`：`install.py:57` 只 import `ops_enabled`/`ENV_OPS`/`OPS_ALL`/`OPS_DEFAULT_ON`，并在 `_family_on`（`:143-157`）里自己 try/except 兜底 —— `enabled_families` 只有测试断言、**无生产调用点**（见 §7.6） | CPU 绿 |
| `vllm/ops/fla/__init__.py` | FLA 公共面：15 个名字（5 个 chunk 算子 + 组合链 + conv1d 两入口 + gating + recurrent 两入口 + `prepare_chunk_indices`） | 间接 | 完成 |
| `vllm/ops/fla/chunk.py` | 六步链的组合与重绑（`A`、`v` 两个名字中途改绑）、`FLA_CHUNK_SIZE=64`、`assert_strict_lower_chunks` | `tests/test_gdn_chunk_full.py`（**19**） | 完成 |
| `vllm/ops/fla/kkt.py` | `chunk_scaled_dot_kkt_fwd`：严格下三角转移矩阵，`beta` 行缩放 + `safe_exp(g_i-g_j)`（**B 级数学**） | `tests/test_gdn_kkt_wu.py`（**35**） | 完成 |
| `vllm/ops/fla/solve_tril.py` | `(I+A)^{-1}` 的 16→32→64 分块求逆 + `prepare_chunk_indices`（A 级语义） | `tests/test_gdn_solve_tril.py`（**30**） | 完成 |
| `vllm/ops/fla/wu.py` | `recompute_w_u_fwd`（WY 表示）：`u=A_inv@(v*beta)`、`w=A_inv@(k*beta*exp(g))`，两个 dot 前先降回 bf16（**B**） | `tests/test_gdn_wu_reference.py`（**8**）+ `test_gdn_kkt_wu.py` | 完成 |
| `vllm/ops/fla/delta_h.py` | `chunk_gated_delta_rule_fwd_h`：每 chunk 进块状态 `h`、`v_new`（**未加门**）、`final_state`；`chunk_offsets` 只在 varlen 用（**B**） | `tests/test_gdn_delta_h_chunk_o.py`（**37**） | 完成 |
| `vllm/ops/fla/chunk_o.py` | `chunk_fwd_o`：块内 `>=` 掩码（含对角）+ 跨块状态项，`scale` 同时乘两项（**B**） | 同上（**37**） | 完成 |
| `vllm/ops/fla/causal_conv1d.py` | `causal_conv1d_fn`（prefill，`[dim, cu_seqlen]` channel-last）与 `causal_conv1d_update`（decode，token-major）——**两个函数、两套轴**，深度可分离因果卷积 + 状态缓存行 | `tests/test_gdn_causal_conv1d.py`（**32**） | 完成 |
| `vllm/ops/fla/gating.py` | `fused_sigmoid_gating_delta_rule_update`：softplus(β=1,threshold=20) → `g=-exp(A_log)*softplus`、`beta=sigmoid(b)`；**先衰减后 delta、先更新后出 o** | `tests/test_gdn_gating_delta_rule.py`（**30**） | 完成 |
| `vllm/ops/fla/recurrent.py` | 两个入口：`fused_recurrent_gated_delta_rule`（序列形）与 `..._packed_decode`（解码形，`q\|k\|v` 打包、门在核内由 `a`/`b`/`A_log`/`dt_bias` 导出） | `tests/test_gdn_recurrent.py`（**45**） | 完成 |
| `vllm/ops/fla/post_conv.py` | `fused_post_conv_prep`：conv 后按 q\|k\|v 拆分、q/k 逐 head L2 归一、逐 value head 门控与 sigmoid β；返回 `(q,k,v,g,beta)` | `tests/test_gdn_post_conv.py`（**17**） | 绿，但**只与自己重建的契约对拍**；见 §7.3 的 `beta` dtype 差异 |

> 另有 1 个 `test_gdn_*` 文件是**遗留 vLLM-Ascend 时代**的用例、不测上面任何模块：
> `tests/test_gdn_gating.py`(6)。同一批里的 `test_gdn_meta_memo.py`(6)、`test_gdn_proj_fusion.py`(9)、
> `test_gdn_reorder.py`(11)、`test_gdn_small_fusion.py`(11) 已随 legacy 迁移到 `legacy/tests/`，
> **不在活套件里**（§5.1b 的旧表仍列着它们，读的时候按这一句打折）。

---

## 3. 已知缺口（按阻塞力排序）

### 3.1 `gpu_model_runner.py` —— 源码已取回，worker 已实现（**硬件未验证**）

* **它当初为什么是 blocker**：引擎要的四个 worker hook（`init_device` / `determine_available_memory` /
  `initialize_from_config` / `compile_or_warm_up_model`）**全部落在 model runner 上**，而 runner 类
  （`GPUModelRunner`）由 `init_device` 构造（`vllm__v1__worker__gpu_worker.py:327-341`）。
* **谁需要它**：`tileinfer/vllm/worker.py` 的四个 hook（已实现：`worker.py:825` / `:1510` / `:1689` /
  `:1796`）；`tileinfer/vllm/platform.py` 的 `_select_worker_class` 写入的 `worker_cls` 正是 `NPUWorker`。
  当时 `docs/hardware-checklist.md` §4 与 `worker.py` 的 docstring 都把它列为「先取回文件」（两处现已更新）。
* **记录的理由**：「没有这个文件，步骤 6 之后只能靠猜签名；每个签名都从 worker 的调用点反推，第一个不匹配
  会在权重加载完之后才以 `AttributeError` 出现」。
* **本轮订正**：`runtime/vllm-api/vllm__v1__worker__gpu_model_runner.py` **已经存在**（334,736 B；
  `class GPUModelRunner` 在 `:418`、`_dummy_run` `:5599`、`profile_run` `:6168`、`capture_model` `:6525`、
  `initialize_kv_cache` `:7284`，其中 `:7301` 调 `initialize_mamba_ssu_backend`）。四个 hook 也已按这份
  源码实现（见 §3.4），所以缺的既不是源码也不是实现，而是**硬件验证**：torch_npu 的内存 API 拼写、40 组
  hybrid 池的实际分配、图捕获门都还没在 NPU 上观测过。详见 §7.1。

### 3.2 三个 MoE 名字（记录为「未绑定」→ 名字已定，缺的是真机确认）

* 记录（当时的期望报告，`bound` 7 个；`unbound` = `chunk_local_cumsum`(ABI) + `topk_softmax` /
  `moe_align_block_size` / `fused_experts`（「vLLM name not verified offline」）；`disabled` = `silu_and_mul`）。
  **这条记录的位置已经不存在**：算子表从 `ops/install.py` 搬到了 `ops/families/gdn_moe.py`
  （`REQUIRED_NAMES` 在 `:63-76`，每行的 `targets` 就在表里），`install.py` 只负责 plan。
  「期望报告」现在的权威版本是 `tests/test_ops_install.py` 的 `EXPECTED_BOUND` / `EXPECTED_UNBOUND` /
  `EXPECTED_DISABLED` 三张精确列表。
* **为什么当时无法定**：本地没有 `fused_moe` 模块，也不确定 vLLM 是直接调 `torch.ops._C.<op>`、还是走
  `vllm/_custom_ops.py` 的包装、还是 `vllm/model_executor/layers/fused_moe/` 里的函数。
* **本轮订正**：三条候选路径现在全部在本地可定位（`vllm._custom_ops.topk_softmax` 等，见 §7.2）。
  `families/gdn_moe.py` 的 `targets` **刚好命中**，所以真正的缺口是「在真 vLLM 进程里跑一次 `install_ops()`
  并确认报告里这三个是 `bound` 而不是 `unbound`」——这是上机后 5 分钟内能做完的事。

### 3.3 `chunk_local_cumsum` —— **故意不绑定**（ABI 不兼容）

`ops/families/gdn_moe.py:313-332`（`OpBinding(op="chunk_local_cumsum", abi_ok=False, …)`）：本仓库的端口是
token-major（`[T, ...]`，参数 `(g, chunk_size, reverse, cu_seqlens)`），
上游是 `(g, chunk_size, reverse, cu_seqlens, chunk_indices, head_first, output_dtype)` 并支持 3-D/4-D 布局
（`runtime/vllm-api/fla_cumsum.py:243-252`）。**不是 drop-in**，绑上去会污染整条链；而我们自己的 chunk 链
内部调用的是自己的 cumsum，没有任何调用点需要替换这个符号。它以 `abi_ok=False` 列出，让这个「缺席」出现在
日志与 `tileinfer doctor` 里，而不是变成无人解释的空洞。**不要**为了让报告「全绿」而删掉这一行。

### 3.4 worker：四个 hook 已实现，等待硬件验证

四个 hook 已实现（`worker.py:825` `init_device`、`:1510` `determine_available_memory`、`:1689`
`initialize_from_config`、`:1796` `compile_or_warm_up_model`），不再 `raise`；`HARDWARE_CHECKLIST`
（`worker.py:208`）把上机要确认的 6 件事写成了数据：`torch.npu.set_device` 是否接受 `torch.device`、
`dist_backend`/HCCL 拼写、`mem_get_info`/`memory_stats` 两组内存 API、图捕获仍被 `capture_path_ready`
关掉且 runner 自己还有两处 `torch.cuda.mem_get_info`（`:6547`/`:6565`）、AI-core 数（`num_sms`）、
10 层 NHD 池 + 30 层 mamba state 的池分配。**CPU 契约测试只证明签名与分支，不证明其中任何一行在 NPU 上跑过。**

### 3.5 `enforce_eager` 在图捕获验证前被强制为 `True`

`platform.py:1579`（`_force_eager_until_capture_lands`）：`graphs.capture_path_ready()` 为 False 时，
平台把 `model_config.enforce_eager`
改写成 `True` 并**只喊一次**警告。含义：**今天 `capture_calls == 0` 是决策，不是故障**。只有 §1 第 17 行那
5 个符号在硬件上核实、桥接成功之后，才允许翻转；翻转前不要用「没进图」当成性能问题来查。

### 3.6 混合 KV 已打开，KV 池是 block-first（这条曾经的「缺口」已关闭）

* **能力位**：`support_hybrid_kv_cache()` → True（`platform.py:1144`）。这不是性能开关：False 会让
  vLLM 0.23 关掉 hybrid KV manager，然后拒绝 attention+mamba spec 混编（§1 第 18 行的两条上游行号）。
* **10/30**：模型 40 层 `layer_types` 每 4 层 1 个 `full_attention` ⇒ **10 full-attention + 30 GDN**
  （`runtime/test-model-qwen36/config.json`）。平台不枚举层，spec 由模型给出
  （`gpu_worker.py:547-548` → `gpu_model_runner.py:7440-7470`）。
* **页几何**：`validate_block_size`（`platform.py:1319`）按 profile 的 `block_size_policy` 把 Qwen3.6 的
  `block_size` 固定为 2048，并把
  `user_specified_block_size` 置 True（否则 vLLM 之后的 `update_block_size_for_backend` 阶段会把它降回
  128）；`align_mamba_state_pages`（`:1398`）写 `mamba_block_size=2048` 与
  `mamba_page_size_padded=4 MiB`。必须相等的原因：padded `MambaSpec` 的 page 不随 `block_size` 缩放，
  `unify_kv_cache_spec_page_size` 无法把两者对齐。
* **池布局**：`(num_blocks, 2, block_size, heads, dim)` + stride order `(1, 0, 2, 3, 4)` ⇒ 物理
  `[2][blocks][page][heads][dim]`；`kv_cache[:, 0]` / `[:, 1]` 是两段连续的 `[blocks, page, heads, dim]`，
  所以 `fused_halves` / `pool_sub_view` / 写 KV 都是零拷贝，`block_dim == 0` 也让 vLLM 的 hybrid 重写
  直接 `continue`（`gpu_model_runner.py:7190-7191`）。
* **`KVBlockZeroer` 故意跳过**：`needs_kv_cache_zeroing` 是 `has_mamba_layers`
  （v0.23.0 `v1/kv_cache_interface.py:873-879`），但 zeroer 用 `stride(block_dim)` 推每块字节数，在这个
  物理顺序下只有半页；`TileInferModelRunner._init_kv_zero_meta`（`worker.py:743`，调用点 `:1785`）
  因此**拒绝**它并记一条带原因的日志。代价是回收页可能残留旧数据 —— 这是设备跑要观察的残留项，不是启动错误。

### 3.7 两个既存静默 bug（已修，读代码时不要「修回去」）

1. **`capacity_view` 丢了 `storage_offset`**（`serving/buckets.py:219-259`）。融合池的 V 半在同一块
   storage 的非零偏移处；重建容量视图时用 `set_(storage, 0, ...)` 会把 V 别名回 K —— 内核把 K 当 V
   再算一遍，形状正确、数值全错、**没有任何异常**。修法是 `set_(..., flat.storage_offset(), ...)`
   （`:228-231` 的注释就是这条）。
2. **hybrid interleaved rewrite**（`attn/backend.py:177-191`、`serving/buckets.py:262-305`）。
   `block_dim == 1` 时 `_update_hybrid_attention_mamba_layout`（`gpu_model_runner.py:7167-7199`）只改
   dims 0/1 的 stride，把池改成按块 K/V 交错；此时一页由三个下标 `(block, kv, sub)` 寻址，没有平坦的
   `[pages, 128, heads, dim]` 视图，`pool_sub_view` 只能复制（所以它现在对非连续池直接拒绝）。修法是
   **声明 block-first 形状 + stride order**，让重写根本不发生。

### 3.8 预烧已接进 worker 生命周期（不是靠外部脚本）

`compile_or_warm_up_model`（`worker.py:1796`）在 eager `_dummy_run` 之后、`capture_model()` **之前**调用
`_prewarm_decode_plans()`（`:1876`）；这是唯一同时满足两半契约的窗口：KV 池已在
`initialize_from_config` 绑定，而图还没捕获。bucket 集来自 `max_num_seqs` / `max_num_batched_tokens` /
`max_model_len` / `block_size` 经 `serving/planner` 推导，plan 由 `attn/impl.py:745` 的缓存持有，所以捕获
区内只做查找。调用幂等；没有设备时是一条带原因的 skip。`scripts/prewarm-tileinfer-kernels.py` 仍在，用于
离线/登录节点预算，不是这条路径的替代。

### 3.9 运维入口：一条命令起服 + 一个自检命令

`scripts/serve_qwen36.py` 校验模型 `config.json`、推出预烧计划（`--plan-only` 可在无卡节点打印）、设好
env 后 exec `vllm serve --attention-backend CUSTOM`（`main` `:802`）。`tileinfer doctor` 把
`hardware-checklist.md` 步骤 1-5 变成 PASS/FAIL/SKIP，`--json` 供脚本消费（`tileinfer/vllm/doctor.py:719`，
console script 在 `pyproject.toml:41`）。两者都只是入口：真正的判据仍在 §5 与硬件清单。

---

## 4. 上机入口

**只认一份文档：[`docs/hardware-checklist.md`](hardware-checklist.md)。**

那份清单按「后一步依赖前一步结论」排了 11 步 + 首次失败分类表 + 每步「今天能不能过」的表格：
1 环境 → 2 entry point → 3 CPU 套件 → 4 取回 runner → 5 平台被选中/block_size=2048 → 6 权重+KV 池 →
7 首个 decode step 对 logprobs 基线 → 8 GDN 30 层 state 有限 → 9 256 专家路由对 CPU oracle →
10 eager vs 图捕获 → 11 TTFT/TPOT。

> **start there, do not skip steps.**
>
> 从第 4 步开始时注意：该步的产物（`runtime/vllm-api/vllm__v1__worker__gpu_model_runner.py`）**已经在仓库里**
> （见 §7.1），所以第 4 步的拷贝命令应当直接通过；`worker.py` 的四个 hook 也已按它实现（见 §3.4），
> 所以这一步真正要产出的是「在 NPU 上跑通 `init_device`」，而不是「文件存在」。

---

## 5. 测试证据（本轮实测，不是引用）

### 5.1 本轮命令与结果

任务指定的命令（只跑平台/启动三件套，避免环境噪声）：

```powershell
$env:PYTHONPATH="E:\project\TileInfer\runtime\pylibs"; $env:PYTHONDONTWRITEBYTECODE="1"
python -m pytest tests/test_platform_contract.py tests/test_boot_rehearsal.py tests/test_worker_contract.py -q -p no:cacheprovider -p no:warnings
```

实测（profile / family / candidate 三个扩展点落地后重跑）：

```text
194 passed
```

逐文件收集数（`--collect-only` 实测）：

| 文件 | 收集 | 内容 |
|---|---|---|
| `test_platform_contract.py` | **122** | ABC 清单、selector 策略、`check_and_update_config`（含 `worker_cls` 源契约与架构接受）、graphs 段、worker 段 |
| `test_boot_rehearsal.py` | **28** | 从 entry point 到 `compile_or_warm_up_model` 的整条启动序列（本重构在启动链上补了 profile 解析/激活的用例，25 → 28） |
| `test_worker_contract.py` | **44** | worker 四个 hook 的签名/dtype/内存会计/KV 路径/捕获门 + `worker_cls` 交接字段 |

守门四件套（`hardware-checklist.md` §3.2；按它列出的顺序重跑）：

```text
tests/test_no_vllm_ascend.py tests/test_layer_deps.py tests/test_platform_contract.py tests/test_attn_contract.py
226 passed        # 2 + 43 + 122 + 59
```

> `test_no_vllm_ascend.py` 现在是 **2** 个汇总 ratchet 用例（整树扫描 + 新层免检），不再是逐文件参数化，
> 所以「39」那个数字（旧文档的 222/263 口径）已经作废。引用任何总数前先重跑，不要抄本节的历史值。

### 5.1b 上一轮的全量数字（**本轮未重跑**，只作对照）

```text
21 failed, 835 passed, 18 skipped, 116 deselected, 8 errors in 21.50s
```

即 **882 个用例被收集并执行**（21+835+18+8），另有 116 个 `npu` 标记的用例被 `-m "not npu"` 取消选择。
**注意**：`test_platform_contract.py` 当时是 83，本轮加入 worker_cls / 架构 / ops-vs-kernels 契约后是
**122**；`test_attn_contract.py` 当时 57，现在是 **59**。所以任何「222」「558」「570」这类旧总数都不
再成立，引用前先重跑。其余文件上一轮的收集数（仅供定位，未重跑）：

| 文件 | 收集 | 文件 | 收集 |
|---|---|---|---|
| `test_ops_install.py` | 18 | `test_kv_cache_write.py` | 15 |
| `test_serving_buckets.py` | 35 | `test_serving_step.py` | 7 |
| `test_plan_cache.py` | 11 | `test_moe_ops.py` | 36 |
| `test_gdn_ops.py` | 12 | `test_gdn_solve_tril.py` | 30 |
| `test_gdn_causal_conv1d.py` | 32 | `test_gdn_gating_delta_rule.py` | 30 |
| `test_gdn_recurrent.py` | 45 | `test_gdn_kkt_wu.py` | 35 |
| `test_gdn_delta_h_chunk_o.py` | 37 | `test_gdn_chunk_full.py` | 19 |
| `test_gdn_post_conv.py` | 17 | `test_gdn_wu_reference.py` | 8 |
| `test_gdn_gating.py` | 6 | `test_gdn_meta_memo.py` | 6 |
| `test_gdn_proj_fusion.py` | 9 | `test_gdn_reorder.py` | 11 |
| `test_gdn_small_fusion.py` | 11 | | |

### 5.2 那 9 个「既存收集错误」（`--ignore` 之前）

不加任何 `--ignore`、只加 `-m "not npu"` 时：

```text
898/1014 tests collected (116 deselected), 9 errors in 2.02s
ERROR tests/test_fused_sample.py / test_fused_sample_integration.py / test_greedy_fast_sampler.py /
      test_moe_align_packer.py / test_moe_prenz_pack.py / test_moe_slot_layout.py /
      test_moe_weight_prefetch.py / test_prep_fastpath.py / test_sw_pipeline_switch.py
```

**为什么不是信号**：其中 **8 个**在模块级 `import tilelang`（`ModuleNotFoundError: No module named
'tilelang'`，CPU box 没有 Ascend 工具链），第 **9 个** `test_prep_fastpath.py` 在模块级 `import numpy`
（本机 `numpy` 缺失）。它们都在 import 阶段就死，与本次接线无关；`docs/notes/dev-env-windows.md`
「Pre-existing collection failures」记的是同一批文件。判读口径：**这 9 个以外的错误才是新的**。

### 5.3 剩下那 21 failed / 8 errors —— 逐条归因（也不是信号，但**必须**知道）

| 数量 | 文件 | 原因 | 归因 |
|---|---|---|---|
| 16 | `test_prewarm_specs`(2)、`test_vllm_ascend_ktp_policy`(1)、`test_vllm_ascend_prefill_block_q`(1)、`test_vllm_ascend_prefill_ctx_shunt`(3)、`test_vllm_ascend_prefill_split_ctx`(9) | `ModuleNotFoundError: No module named 'tilelang'`（都在 `tileinfer/kernels/attention/paged_decode_ascend950.py:34`） | 遗留 vLLM-Ascend 路径 + 内核工具链缺失 |
| 3 | `test_dense_nn` | 注入的 fake `vllm` 太薄：`ImportError: cannot import name 'VllmConfig' from 'vllm.config'`、`No module named 'vllm.model_executor'`、`AssertionError: vllm linear classes unavailable` | 需要**真** vLLM |
| 2 | `test_shared_expert_fold` | `ModuleNotFoundError: No module named 'vllm_ascend'` | 遗留依赖，已在迁移目标之外 |
| 2 + 1 + 5 | errors：`test_lb_dump_hook`(2)、`test_vllm_ascend_gdn_wu_dispatch`(1)、`test_vllm_ascend_prefill_step_guard`(5) | `PermissionError`（沙箱拒绝写 `pytest-of-Dell` 临时目录）/ `No module named 'vllm'` / `tilelang` | 环境 |

**关键结论**：这 21+8 个**没有一个落在 `tileinfer/serving/` 或 `tileinfer/vllm/` 的目标模块上**；上一轮
23 个目标测试文件 570 个用例 0 失败。上机后若这些还红，**先按上表对环境再判一次**，不要当成内核回归。
（本轮只重跑了 §5.1 的三件套 + 守门四件套 + §5.4 的扩展点四件套；这三组的 0 失败是本轮实测。）

### 5.4 三个扩展点的契约（四件套，本轮实测）

```powershell
$env:PYTHONPATH="E:\project\TileInfer\runtime\pylibs"; $env:PYTHONDONTWRITEBYTECODE="1"
python -m pytest tests/test_doctor.py tests/test_model_profiles.py tests/test_op_candidates.py tests/test_ops_install.py -q -p no:cacheprovider -p no:warnings
```

实测：**92 passed**（`test_doctor` **40** / `test_model_profiles` **8** / `test_op_candidates` **25** /
`test_ops_install` **19**）。这四件套是「加模型 / 加算子族 / 加更快的实现」时的最小回归面：

* `test_model_profiles.py` —— profile 注册表、平台/backend/worker 真的执行 profile 的值、以及
  **dummy profile 不改核心就驱动四者**（seam 本身）；`ops.install` 跟着 profile 的 `op_family` 走。
* `test_op_candidates.py` —— 候选门（熔断 / flag / 实测比值 / predicate）、`select` 不碰张量值、
  dispatcher 逐字节等于 baseline、`describe()` 带原因。
* `test_ops_install.py` —— `EXPECTED_BOUND` / `EXPECTED_UNBOUND` / `EXPECTED_DISABLED` 的**精确**列表
  （上游改名或漏移植会直接红），以及 install 仍是最后一步。
* `test_doctor.py` —— 含「候选门状态出现在 `ops.binding` 行、且不改变退出码」的用例。

`python -m tileinfer.vllm.doctor` 在本机（无 vLLM）仍 **exit 1**（hard check：`torch_npu` / `cann_env` /
`tilelang` / `tilelang_cache`），`ops.binding` 行是 SKIP，但**带**候选段：

```text
[SKIP] ops.binding  skipped: vLLM is not importable, so there are no vLLM operators to take over | candidates: fused_experts[moe_reference_oracle=disabled: off by default]
```

加扩展的步骤与「不能改什么」见 [`extending.md`](extending.md)。

---

## 6. 重复劳动警告（这些代价已经付过一次，别再付）

1. **`solve_tril` 求的是 `(I + A)^{-1}`，不是 `(I - A)^{-1}`。**
   `docs/notes/vllm-api-0.23.md:103` 那句 “`(I - A)`” 是**错的**。正确证据：`fla_solve_tril.py:513-530`
   的 docstring、`:82` 的 `b_A = -tl.where(m_A, b_A, 0)`、`:89` 的 `b_A += m_I`，
   以及 `tileinfer/vllm/ops/fla/solve_tril.py:34-45`。差一个符号的结果「形状完全正确、数值全错」。
2. **`causal_conv1d` 的两个入口 token 轴不同，不能互换比较。**
   `causal_conv1d_fn`（prefill）= `[dim, cu_seqlen]` 且 channel 轴最内
   （`causal_conv1d.py` 本地副本 `:18`/`:563`/`:620`）；`causal_conv1d_update`（decode）= token-major
   `[num_tokens, dim]`（`:1085-1090`、`:1171-1172`）。decode 的 `[T,dim]` 必须转置后才能和 prefill 的
   `[dim,T]` 比。踩过的具体症状：`conv_finite=0` / 只有第一个 token 对。
3. **`v_new` 存的是「未加门」的值。**
   `delta_h` 里 `safe_exp(g_last - g)` 只乘那份**用于更新状态**的副本，`v_new` 先落盘再乘门；门由下游
   `chunk_o` 以 `exp(g_i - g_j)` 施加（`fla/ops/chunk.py:75` 把 `v_new` 传给 `chunk_o` 的 `v` 形参）。
   在 `v_new` 上再乘一次 = **双重衰减**，整条链 O(1) 失败。已有数值证据：未加门版 fp64 合链误差
   3.5e-16（`tests/test_gdn_chunk_full.py`）、而加门版 `v_new` 相对差 7.3e-1、`o` 相对差 5.4e-1。
4. **掩码必须 `torch.where` 选择，并且要配对 `safe_exp`。**
   `safe_exp(x) = exp(where(x <= 0, x, -inf))` 是**单边截断**：正的指数参数一律变 0，不是 `exp(x)`。
   分工规则（六个位置已核实）：`exp` 只用于「绝对的 `g`」（必 ≤0）；`safe_exp` 只用于「`g` 的差」（可正）。
   **写成 `b_A * mask` 会得到 `inf * 0 = nan`**，NaN 会顺着 `solve_tril → w/u → h` 扩散，而且**只在 `g`
   衰减足够大时**才复现——小规模单测全绿，上真模型才炸。
5. **`accept_output_buffer` 在 vLLM 0.23 里不存在。**
   `grep -n accept_output_buffer runtime/vllm-api/*` 无匹配；`AttentionBackend` ABC（`backend.py:55-349`）
   没有这个名字。`docs/plan-vllm-native-backend.md` §W4 把它列为接口是错的。`tileinfer/vllm/attn/backend.py`
   保留的同名钩子只是为了向前兼容，**不得**描述成 0.23 的 API。
6. **同一个链路里有两个不同的三角，`>` 与 `>=` 不是笔误。**
   `kkt` 造 `A` 用**严格**下三角 `o_t[:,None] > o_t[None,:]`（SG-kkt:82）；`chunk_o` 的块内注意力用
   **含对角** `o_i[:,None] >= o_i[None,:]`（SG-chunk_o:109）。`chunk_o` 写成 `>` 会系统性丢掉
   `scale·v_i`；而 `kkt` 写成 `>=` **永远不会被自己的单测发现**（`solve_tril` 只读严格下三角，对角被静默
   丢弃），只在别人消费 `A` 时才暴露。
7. **两个名字碰撞**：`chunk_fwd_o` 的 `v` 实参拿到的是 `v_new`（不是原始 `v`）；`chunk_gated_delta_rule_fwd_h`
   的 `u` 形参在 kernel 里**叫 `v`**。两处对调都会跑出「形状完全正确的错误答案」。
8. **`BT` 恒等于 64，绝不随 `T` 收缩。**
   `qwen_gdn_linear_attn.py:1085-1087`：「All kernels … now use a fixed `BT = chunk_size` (64)」。
   sglang 副本的 `chunk_o` 会 `min(chunk_size, max(16, next_pow2(T)))`，**照抄它在短序列上会改变数值**。
9. **`h[i_t]` 是「进块状态」**（store 在循环体最前、状态更新在最后），用更新后的 `h` 做 `chunk_o` 会让
   每个 chunk 错位一格；`b_g_last` 取 `min((i_t+1)*BT, T)-1`，**尾块不是 `g[BT-1]`**。
10. **三处 bf16 截断是契约不是实现细节**：`A_inv` 是 `k.dtype`（`chunk.py:50`）、`wy_fast` 两个 dot 前先降回
    bf16、`h` 每个 chunk 边界落盘截断一次。全程 fp32 的实现**更准但不逐位一致**；要 bit-comparable 必须
    复现这三处。
11. **`l2norm` 的 `eps` 加在平方和上**：`x / sqrt(sum(x²) + eps)`，不是 `x / max(||x||, eps)`。两者只在
    近零行上分道扬镳，而那正是门控早期会产生的行——「看起来合理」的错。
12. **不要给 pytest 传第二个 `-q`。** `pyproject.toml:56` 已有 `addopts = "-q"`，再传 `-q` 会让 pytest 进入
    更高静默级别、**吞掉最后的汇总行**，你只能从进度条数点号。本次就踩了一次：想要计数就**不要**手写 `-q`。
13. **`chunk_indices` 只在 varlen 分支被读**；`IS_VARLEN` 只看 `cu_seqlens`。dense 情况下传了也不生效，
    「传了 `cu_seqlens` 但内部仍按 `cdiv(T,BT)` 循环」会多算或漏算。

---

## 7. 与仓库现有文档不一致之处（本轮实测订正，**上机前先读这节**）

> 这一节不是「文档写错了要批评」，而是「如果你只信旧 docstring，会白做一轮工作」。以下每条都可复现。

### 7.1 `gpu_model_runner.py` 已经不缺

* 旧记录（当时的 `worker.py` docstring 与 `HARDWARE_CHECKLIST` 第 1 条，两处现已改写）：
  `docs/hardware-checklist.md` §4、`docs/notes/vllm-api-0.23.md:131-135` 都说它
  「不在 `runtime/vllm-api/`，未取回」。
* 实测：`runtime/vllm-api/` 共 **36 个文件**（旧 docstring 写 17，上一轮写 24），其中
  `vllm__v1__worker__gpu_model_runner.py` = **334,736 B**，
  `class GPUModelRunner` `:418`、`_dummy_run` `:5599`、`profile_run` `:6168`、`capture_model` `:6525`、
  `initialize_kv_cache` `:7284`（其 `:7301` 调 `initialize_mamba_ssu_backend`）。同时
  `vllm__v1__worker__gpu_worker.py`（已验证 50,256 B，行号在 `worker.py` 里引用的
  `:249/:371/:562/:591` 等位置可对上）。
* **本轮新增的 12 个**只服务于两条启动路径的核实：`config/{vllm,model,parallel,model_arch}.py`、
  `platforms/cpu.py`、`plugins/__init__.py`、`v1/executor/{abstract,uniproc_executor}.py`、
  `v1/engine/core.py`、`v1/worker/worker_base.py`、`model_executor/models/{registry,__init__}.py`。
  `platforms/interface.py`、`platforms/__init__.py`、`gpu_worker.py` 三个旧文件被重新下载覆盖，字节数不变
  （即与上一轮同一份 v0.23.0）。
* **结论**：blocker #1 的**源码**障碍已消失，`worker.py` 的四个 hook（`:825` / `:1510` / `:1689` /
  `:1796`）也已按它实现；剩下的是**硬件验证**，不是写实现。`hardware-checklist` §4 的拷贝命令会直接成功，
  不要因为它「已经成功」就以为整步做完了——这一步的产物现在是「`init_device` 在 NPU 上跑通」。
* **未核实**：这些文件是否与 `v0.23.0` 的发布物逐字节一致（本轮只按 raw.githubusercontent 的 tag 抓取，
  没有 sha256 对照）。

### 7.2 MoE 三个名字已经能定位，而且 `moe_align_block_size` 的参数顺序**推论是对的**

| 算子 | 上游可调用名 | 底层 | 落地行号 |
|---|---|---|---|
| `topk_softmax` | `vllm._custom_ops.topk_softmax(topk_weights, topk_ids, token_expert_indices, gating_output, renormalize=False, e_score_correction_bias=None)` | `torch.ops._moe_C.topk_softmax` | `vllm___custom_ops.py:2494` / `:2502` |
| `moe_align_block_size` | `vllm._custom_ops.moe_align_block_size(topk_ids, **num_experts, block_size**, sorted_token_ids, experts_ids, num_tokens_post_pad, expert_map=None)` | `torch.ops._moe_C.moe_align_block_size` | `vllm___custom_ops.py:2346` / `:2355` |
| `fused_experts` | `vllm.model_executor.layers.fused_moe.fused_moe.fused_experts(...)` | `torch.ops.vllm.fused_experts` | `vllm__model_executor__layers__fused_moe__fused_moe.py:1474` / `:1490` |

* `ops/families/gdn_moe.py` 的 `targets` **恰好**就是这三条（`vllm._custom_ops.topk_softmax`、
  `vllm._custom_ops.moe_align_block_size`、`vllm.model_executor.layers.fused_moe.fused_moe.fused_experts`；
  表从 `install.py` 搬来这里后行号变了，按 `op=` 找），所以它们不是「名字未知」，
  而是「在无 vLLM 的 CPU 环境里必然 `unbound`」。
* 附带收获：`ops/moe.py` 里被标 `[推断]` 的「`num_experts` 在 `block_size` 之前」**与上游一致**
  （`_custom_ops.py:2346-2350`），`moe_align_block_size_block_first` 这个别名在真 vLLM 上不会被用到
  （保留无害）。
* **仍然未核实**：`vllm.model_executor.layers.fused_moe.moe_align_block_size`（`fused_moe.py:25-26` 从
  同名子模块 import）内部是否直接转调 `_custom_ops`——该子模块**不在**本地 36 个文件里；以及
  `fused_experts` 的实际 kwargs（量化/LoRA 分支）是否与 `ops/moe.py` 的签名兼容。

### 7.3 `post_conv` 的 C 级前提已不成立，且 `beta` 的容器 dtype 有一处真实差异

* 旧记录（`ops/fla/post_conv.py:12-27`、`:183-201`）：vLLM 与 sglang 两侧**都没有**该文件，所以是 C 级重建；
  `g`/`beta` 的容器 dtype、`apply_l2norm=False`、`output_g_exp=True` 三个分支标 `[推断]`。
* 实测：`runtime/vllm-api/vllm__model_executor__layers__fla__ops__fused_gdn_prefill_post_conv.py`（7,935 B）
  **存在**，`def fused_post_conv_prep` 在 `:152`，返回 `return q, k, v, g, beta`（`:206` / `:248`），
  与端口的 5 元组契约一致；`g` 与 `beta` 都分配为 **`torch.float32`**（`:202-203`），
  且 `beta_ptr` 注释为 `[L, HV] float32`（`:33`）。
* **由此产生一处需要上机优先复核的差异**：端口在 `ops/fla/post_conv.py:404` 把 `beta` 落回 `b.dtype`
  （生产是 bf16），而**上游是 fp32 容器**。`g` 侧端口是 fp32（`_sigmoid_gating_decay` 返回 working dtype
  = fp32，见 `gating.py:148-152`、`:187`），与上游一致。
  ⇒ **如果 GDN prefill 的数值不对，第一个要看的就是 `beta` 这一处 bf16 截断**（它只影响
  `kkt`/`wy_fast` 的 `beta` 行缩放，量级 1/256，不会崩，只会漂）。
* 仍属 C 级、必须回填的：`apply_l2norm=False` 与 `output_g_exp=True` 两个分支**仍然没有上游调用点**
  （两个生产调用点都传 `True` / `False`），端口对这两个分支的实现是读 flag 名字推的。

### 7.4 文档内部的计数归属有错（不影响结论，但会误导核对）

`docs/hardware-checklist.md` §3.2 曾写「`222 passed`（57 + 43 + 39 + 83）」并按
`test_no_vllm_ascend / test_layer_deps / test_platform_contract / test_attn_contract` 的顺序对应，
其中逐文件归属是错位的；后来同步成 `263`（39 + 43 + 122 + 59）。**这两个数现在都作废**：
本轮实测 `test_no_vllm_ascend.py` 只有 **2** 个汇总 ratchet 用例，守门四件套 = **2 + 43 + 122 + 59 =
226 passed**（§5.1；`test_platform_contract.py` 122、`test_attn_contract.py` 59 保持不变）。
引用任何总数前先重跑 §5.1 的命令，并留意 `hardware-checklist.md` §3.2 是否已跟上 226。

### 7.5 「CPU 套件全绿」的措辞与实测不符

`docs/hardware-checklist.md` §3 标题与期望输出是「全绿（含 9 个既存的收集错误）」。实测在给定 10 个
`--ignore` 后仍会得到 **21 failed / 8 errors**（原因见 §5.3，全部是 tilelang / 真 vLLM / `vllm_ascend` /
沙箱写权限）。**23 个目标测试文件上一轮是 0 失败的**，所以接线本身的结论不变，但「全绿」这个词会让人在第一次
上手时误判环境坏了。建议把期望输出改成实测那一行（本轮已按 §5.1 重跑三件套 + 守门四件套，并按 §5.4 重跑
扩展点四件套；三组的 0 失败都是本轮实测）。

### 7.6 `switch.enabled_families` 没有**生产**调用点

`ops/switch.py:19-24` 的 docstring 写「:func:`enabled_families` is the tolerant form the installer uses
instead」，但全仓库搜 `enabled_families` 只有：本模块的定义（`:94`）与自身 docstring、
`ops/candidates.py:76` 的一句注释，以及 `tests/test_ops_install.py:718` 的一条断言——
**安装器没有用它**。真正的容忍逻辑是 `ops/install.py:143-157` 的 `_family_on` 自己 try/except +
回退 `OPS_DEFAULT_ON` 实现的。
**不是 bug（行为正确），但下一代读者会以为改 `enabled_families` 就能改安装行为。** 上机时若非要动族开关，
改的是 `install.py` 的 `_family_on`，或者直接把 `enabled_families` 接上（那样要同时改
`tests/test_ops_install.py:718` 的断言口径）。

### 7.7 两条启动路径假设已核实（SUSPECT 1 / 2）

SUSPECT 1 以前标着 `[unverified]`，SUSPECT 2 则是「没有调用点证据的假设」；本轮把 vLLM 0.23 的对应源码
抓进 `runtime/vllm-api/`，逐行读完之后的结论是：**SUSPECT 1 的字段名是对的、机制需要改严；SUSPECT 2
什么都不用实现。** 行号都是本地文件的行号。

**SUSPECT 1：`parallel_config.worker_cls`（字段真实，但不是唯一写入者）**

* 字段存在：`vllm__config__parallel.py:248` —— `worker_cls: str = "auto"`（docstring `:249-250` 明确说
  `"auto"` 表示「由平台决定」）。
* 唯一消费者：`vllm__v1__worker__worker_base.py:249-253`，`WorkerWrapperBase.init_worker` 里
  `resolve_obj_by_qualname(parallel_config.worker_cls)`；非字符串在 `:254-259` 直接 `ValueError`
  （"passing worker_cls is no longer supported"）。wrapper 由 executor 构造
  （`vllm__v1__executor__uniproc_executor.py:48`，单卡走 `uni`）。
* 写入时机：`VllmConfig.__post_init__` 调 `current_platform.check_and_update_config(self)`
  （`vllm__config__vllm.py:1335`），早于 executor/worker 创建，所以在这里写**会被看到**。
* in-tree 先例：`vllm__platforms__cpu.py:153-154`（CUDA 平台用同一分支，未收录本地）——
  `if parallel_config.worker_cls == "auto": parallel_config.worker_cls = "<自己的类>"`。
* **不是** `Platform.get_worker_cls`：0.23 的 `Platform` ABC（`vllm__platforms__interface.py:105-1052`）
  没有这个方法；也不是 entry point：`vllm__plugins____init__.py:14-22` 只定义
  `general_plugins` / `io_processor_plugins` / `platform_plugins` / `stat_logger_plugins`，没有
  worker 组——`platform_plugins` 选的是平台，不是 worker。
* 因此 `_select_worker_class` 改成：`"auto"`（或手工 config 的 `None`/`""`）→ 写
  `tileinfer.vllm.worker.NPUWorker`；已经是这个值 → 保持（幂等）；**其他任何值 → 抛**
  （原来是无条件覆盖：会悄悄盖掉操作员的 `--worker-cls`，而不覆盖又会让 CUDA worker 在 NPU 上死）。
  测试新增了源契约用例（钉住 `parallel.py:248` / `worker_base.py:250,252` / `cpu.py:153-154`，并断言
  interface 里没有 `get_worker_cls`）与三种取值分支。

**SUSPECT 2：模型架构接受（不需要 hook，也不要加 allow-list）**

* `Platform.verify_model_arch` 的 ABC 体就是 `pass`，原文「By default all models are considered
  supported」（`vllm__platforms__interface.py:708-718`）。
* 它在 import 模型类**之前**被调用：`vllm__model_executor__models__registry.py:945`
  （`_try_load_model_cls`，`:939-950`）。OOT 平台在这里只能**否决**，不能白名单。
* 真正的接受判据是注册表：`_VLLM_MODELS`（`:693-704`）里
  `Qwen3_5ForConditionalGeneration` / `Qwen3_5MoeForConditionalGeneration` → `qwen3_5`
  （`:569-573`）；裸 `Qwen3_5ForCausalLM` 由 `_normalize_arch`（`:1134-1158`）按
  `model.py:1912-1929` 的后缀默认值改写成已注册的 `...ForConditionalGeneration`。
* 本地 `qwen3_5.py` **不自注册**：文件里只有 class 定义（`:529` / `:533` / `:554` / `:771`），
  没有任何 `ModelRegistry.register_model` 调用，也没有 try/except 包裹注册——注册是中央懒加载表的职责
  （`registry.py:933-935` 才 import 模块）。所以「注册是否无条件」这个问题不成立：它根本不注册。
* 结论：**不改 `verify_model_arch`**。若加了 override，它反而会拒绝所有没列出的 arch。平台的贡献仍是
  0 行——这与 `UNIMPLEMENTED_ABC_METHODS["verify_model_arch"]` 的 `default is fine` 一致；测试新增了
  「未 override + 对三个 Qwen 拼写不抛」与「注册表/归一化/不自注册」的源契约用例。

**仍然未核实的两点**（不要当成已解决）：① 本地 checkpoint 用的是
`Qwen3_5MoeForConditionalGeneration`；如果换成一个 `architectures` 只写裸 `Qwen3_5ForCausalLM` 的
文本 checkpoint，归一化是否总命中已注册项需要按 `_normalize_arch` 的实际输入再跑一次（本轮只做了源码级
推演）。② `--worker-cls` 与平台冲突时抛 `ValueError` 是**本仓库的选择**（比 in-tree 平台更严）；
vLLM 0.23 本身不校验显式值，只要求它是字符串。

---

## 8. 交接时该做的三件事（按顺序）

1. 在 Ascend 机器上重跑 §5.1 的两组命令，把计数与本文对齐：三件套预期 **194 passed**，守门四件套预期
   **226 passed**，扩展点四件套（§5.4）预期 **92 passed**；全量套件本轮未重跑（§5.1b 的 882/570 是上一轮
   数字），失败归因仍按 §5.3。**不要**为了凑旧数字去删测试：变少就是契约被删了。
2. 按 §7 复核两条启动路径的**硬件侧**结论：`worker_cls` 的写入已在源码层定案（§7.7），设备上只需确认
   `init_device` 之前日志里的 `worker=tileinfer.vllm.worker.NPUWorker`；`verify_model_arch` 没有 override，
   设备上若出现「architecture not supported」那是注册表问题，不是平台 hook 问题。同时把
   `docs/notes/` 里被推翻的前提划掉（尤其 `vllm-api-0.23.md:131-135` 与 `post_conv.py` 的
   「no source anywhere」）。
3. 打开 `docs/hardware-checklist.md`，从第 1 步开始；第 4 步的产物已在仓库里，直接进第 5 步。第 5 步现在
   能直接看到混合 KV 的 10/30 组与 4 MiB 的 mamba 页，不要再把它当成「hybrid 未打开」去查。
   第 11 步之后若有人要「顺手加个 TP」，先读 [`extending.md`](extending.md) §6：并行是**明确未扩展**的区域，
   profile 与 `plan_key` 都没有 `tp` 维度，算子是整张量假设。
