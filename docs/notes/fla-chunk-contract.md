# FLA chunk 链路移植契约（vLLM 0.23 GDN 的四个待移植算子）

本文只记录**从源码读出来的**东西，不写实现。目的：四个算子由四个人分别实现，但它们的
张量约定是相互耦合的（同一个 `g`、同一个 `A`、同一个 `h` 被多方消费），所以先把约定钉死，
避免四个人各自发明一套轴序。

阅读约定：

* `chunk.py` = `runtime/vllm-api/vllm__model_executor__layers__fla__ops__chunk.py`（**真·vLLM 0.23**）
* `fla_solve_tril.py` / `fla_cumsum.py` / `fla_l2norm.py` = `runtime/vllm-api/fla_*.py`（**真·vLLM 0.23**）
* `SG-<name>.py` = `E:\project\python_test\.venv\Lib\site-packages\sglang\srt\layers\attention\fla\<name>.py`
  （sglang 0.5.10 vendored 的同一份上游 FLA 代码，**不是** vLLM 0.23，见下节）
* `tileinfer` = 本仓库已移植的 torch 版本，是本契约的既有对端

标签含义：

* **已核实（行号）**：该行号所在的文件我实际读过，行号可直接跳过去对照。
* **推断（依据 X）**：源文件不可得，只能由调用点/兄弟副本反推，依据写在括号里。

---

## 0. 先说来源问题（必读，否则下面的行号会被误用）

**四个待移植文件的 vLLM 0.23 原文，本轮没有拿到。** 本机无出网能力，逐条试过：

| 通道 | 结果 |
|---|---|
| `python urllib` → `raw.githubusercontent.com` | `RemoteDisconnected` ×3、`_ssl.c:993 handshake timed out` ×1 |
| `web_fetch` 工具 | `TypeError: fetch failed` |
| `curl.exe` → raw.githubusercontent / cdn.jsdelivr / fastly.jsdelivr / gcore.jsdelivr / ghproxy.net / raw.gitmirror | `http_code=000`（全灭） |
| `curl.exe` → pypi.org / tuna / aliyun 镜像 | `http_code=000`（全灭） |
| DNS / TCP | `raw.githubusercontent.com` 解析正常，**TCP 443 可连**，TLS 握手失败 |

所以本契约的**证据分两级**，每一条都标了级别：

1. **调用链、参数名、dtype 流向、张量复用** —— 全部来自真·vLLM 0.23 的 `chunk.py`，
   它是这四个算子唯一的调用者，`chunk.py:41-82` 的每个关键字实参就是签名的一部分。**这一级是最硬的。**
2. **kernel 体内的数学（A 怎么算、掩码是 `>` 还是 `>=`、`safe_exp` 的钳位方向、h 的轴序）**
   —— 来自 SG-* 副本。它是同一份上游 FLA（`fla-org/flash-linear-attention`）的姊妹版本，
   文件头都写着 `Adapted from .../fla/ops/common/...`。

**SG-\* 与 vLLM 0.23 确实不是同一版**，这点必须钉住，因为直接影响移植：

我手上同时有 vLLM 版和 SG 版的三个同名文件，逐个比对：`solve_tril` 558 行 vs 464 行、
`cumsum` 282 vs 292、`l2norm` 151 vs 150，哈希全不相同 —— vLLM 版多出 TMA
（`make_tensor_descriptor` / `USE_TMA`）、`input_guard`、`do_not_specialize` 等。

**已知的、必须按 vLLM 而不是按 SG 来的差异（都在下面各节复述）：**

| # | SG 副本 | vLLM 0.23（以 `chunk.py` 实参为准） |
|---|---|---|
| 1 | `chunk_scaled_dot_kkt_fwd(k, beta, g_cumsum, cu_seqlens, chunk_size, output_dtype)`（SG-kkt:89-96），**没有** `chunk_indices` | 门控形参叫 **`g`**（`chunk.py:44` 写 `g=g`），且**接受 `chunk_indices`**（`chunk.py:46`） |
| 2 | `chunk_gated_delta_rule_fwd_h(...)` 返回 **2** 元 `(h, v_new)`（SG-delta_h:284/340），有 `gk` / `initial_state_indices` / `save_new_value`，**没有** `output_final_state` / `chunk_indices` / `chunk_offsets` | 返回 **3** 元 `(h, v_new, final_state)`（`chunk.py:61`），形参含 `output_final_state` / `chunk_offsets`（`chunk.py:67,70`），不再传 `initial_state_indices` |
| 3 | `chunk_fwd_o(...)` 收 `chunk_size=64`（SG-chunk_o:134），**没有** `chunk_indices` / `core_attn_out` | 收 `chunk_indices` 和 `core_attn_out`（`chunk.py:80-81`），**没有** `chunk_size` |
| 4 | `chunk_fwd_o` 里 `BT = min(chunk_size, max(16, next_pow2(T)))`（SG-chunk_o:138），**BT 会随 T 缩** | **BT 恒等于 64**，见 `qwen_gdn_linear_attn.py:1085-1087`：“All kernels including `chunk_fwd_kernel_o` now use a fixed `BT = chunk_size` (64).” |
| 5 | `solve_tril` 无 TMA 分支 | 有 `USE_TMA`（`fla_solve_tril.py:46,72-81,90-100`） |

> **结论**：照抄 SG 副本会踩第 1/2/3/4 条。凡是写“函数签名”的地方，都以 `chunk.py` 的实参为准；
> SG 副本只用来读 **kernel 体内的数学与轴序**，因为那部分在 FLA 各版本间是稳定的
> （同样的 `beta` 行缩放、同样的严格下三角、同样的 `h[V,K]` 布局）。

`recompute_w_u_fwd` 是个例外：它的形参名在两边**完全一致**，可以放心引用 SG 版行号。

---

## 1. 调用链（`chunk_gated_delta_rule` 内部到底按什么顺序跑）

入口两层：`chunk_gated_delta_rule`（`chunk.py:137`）→ `ChunkGatedDeltaRuleFunction.forward`
（`chunk.py:93`）→ `chunk_gated_delta_rule_fwd`（`chunk.py:23`）。真正的链在最后这个函数里，
**`chunk.py:37-86`，共 6 步**：

| 步 | 行号 | 动作 | 产出 |
|---|---|---|---|
| 0 | `chunk.py:37-39` | `g = chunk_local_cumsum(g, chunk_size=FLA_CHUNK_SIZE, ...)`（**已移植**） | `g` 变成 **chunk 内累加 log-decay，fp32** |
| 1 | `chunk.py:41-48` | `A = chunk_scaled_dot_kkt_fwd(k, beta, g, ..., output_dtype=torch.float32)` | `A` **fp32** `[B,T,H,BT]`，块内严格下三角 |
| 2 | `chunk.py:49-51` | `A = solve_tril(A, ..., output_dtype=k.dtype)`（**已移植**） | 同名 `A` 被**改绑**为 `(I+A)^{-1}`，**bf16** |
| 3 | `chunk.py:52-60` | `w, u = recompute_w_u_fwd(k, v, beta, A, g_cumsum=g, ...)` | `w` `[B,T,H,K]`、`u` `[B,T,H,V]` |
| 4 | `chunk.py:61-71` | `h, v_new, final_state = chunk_gated_delta_rule_fwd_h(k, w, u, g, initial_state, output_final_state, ..., chunk_offsets=...)` | `h` 每 chunk 的入块状态、`v_new`、`final_state` |
| 5 | `chunk.py:72-82` | `o = chunk_fwd_o(q, k, v=v_new, h=h, g, scale, ...)` | `o` `[B,T,H,V]` |

所以**四个待移植算子的实际顺序**是：

```
chunk_local_cumsum (已有)
      → chunk_scaled_dot_kkt_fwd      [待移植 1]
      → solve_tril                    (已有，夹在 1 和 2 之间)
      → recompute_w_u_fwd (wy_fast)   [待移植 2]
      → chunk_gated_delta_rule_fwd_h  [待移植 3]
      → chunk_fwd_o                   [待移植 4]
```

**已核实（`chunk.py:37-82`）**：顺序、每个调用的关键字实参、以及 `output_dtype` 的两次改写。

### 1.1 中间张量的生产/消费矩阵（谁被复用了几次）

这张表是本文档最重要的部分——四个人分工时，任何一个人改了自家输出的 dtype 或轴序，
都会静默污染下游。

| 张量 | 生产于 | 消费于 | 复用次数 |
|---|---|---|---|
| `g`（fp32，chunk 内 cumsum） | `chunk.py:37` | kkt `:44`、wy_fast `:57`、delta_h `:65`、chunk_o `:76` | **4** |
| `A_kkt`（fp32） | kkt | 仅 `solve_tril:50` | 1 |
| `A_inv`（**bf16**，`k.dtype`） | `solve_tril` | 仅 wy_fast（作为 `A`） | 1 |
| `k` | 入参 | kkt `:42`、wy_fast `:53`、delta_h `:62`、chunk_o `:74` | **4** |
| `beta` | 入参 | kkt `:43`、wy_fast `:55` | 2（**delta_h 和 chunk_o 都不碰 beta**） |
| `v`（原始） | 入参 | **仅** wy_fast `:54` | 1 |
| `q` | 入参 | **仅** chunk_o `:73` | 1 |
| `w` | wy_fast | delta_h `:63` | 1 |
| `u` | wy_fast | delta_h `:64`（kernel 里这个形参**叫 `v`**，SG-delta_h:314 `v=u`） | 1 |
| `v_new` | delta_h | chunk_o 的 **`v` 形参** `:75` | 1 |
| `h` | delta_h | chunk_o `:76` | 1 |
| `initial_state` | 入参 | 仅 delta_h `:66` | 1 |

**已核实（`chunk.py:42-81` 每个关键字实参）。**

两处最阴的名字碰撞：

* **`chunk_fwd_o` 的 `v` 拿到的是 `v_new`，不是原始 `v`**（`chunk.py:75`：`v=v_new`）。
* **`chunk_gated_delta_rule_fwd_h` 的 `u` 形参在 kernel 里叫 `v`**（SG-delta_h:314 `v=u`，
  而 kernel 形参名是 `v`，SG-delta_h:35）。看 kernel 时以为在读 `v`，其实读的是 `u`。

### 1.2 返回值与 `SUPPRESS_LEVEL`

**已核实（`chunk.py:83-86`）**：`SUPPRESS_LEVEL < 3` 时 `chunk_gated_delta_rule_fwd` 返回
`(g, o, A, final_state, None, None, None)`；`>= 3` 时返回 `(g, o, A, final_state, w, h, v_new)`。
这只影响调试可见性，不影响数学。`SUPPRESS_LEVEL` 的默认值在 `fla/ops/utils.py`，**该文件未下载，
本轮无法确定**（`chunk.py:19` 只证明它从 `.utils` import）。

### 1.3 外层的两处副作用

* **`q`/`k` 可能被就地 l2norm**：`chunk.py:109-111`，`use_qk_l2norm_in_kernel=True` 时
  `q = l2norm_fwd(q); k = l2norm_fwd(k)` —— 但这是**重新绑定**，不改写调用者的张量。
  生产 prefill 路径传 `False`（`qwen_gdn_linear_attn.py:1529`）。
* **原地写 `core_attn_out`**：`chunk.py:129-133` 断言 `core_attn_out` 复用**只允许推理**
  （`assert not torch.is_grad_enabled()`）且 `q.dtype == o.dtype`。最终返回
  `o.to(q.dtype), final_state`（`chunk.py:134`）。**已核实。**

---

## 2. 四个算子：函数名、完整签名、每个张量的形状/轴序/dtype

### 2.0 通用轴序（四个算子一致，先钉死这一条）

**已核实（`chunk.py:155-176` docstring + `fla_solve_tril.py:519` + SG 四个 kernel 的
`make_block_ptr` 步长）：**

* **token 轴在 dim 1，不是 dim 0**。所有张量是 `[B, T, H, ...]`，**不是** `[T, H, ...]`，
  也**不是** `[B, H, T, ...]`。
* `head_first=False` 是唯一被这四个算子使用的布局。`fla_cumsum.py:55-64` 显示 FLA 确实支持
  `HEAD_FIRST` 分支（`s + bos*H + i_h*T`），但 `chunk.py:37-39, 52-60, 61-71, 72-82`
  **一次都没传 `head_first`**，所以走默认 `False`，即 `[B,T,H,...]`。**已核实。**
* **打包（packed）约定**：`cu_seqlens` 非空时 `B` **必须等于 1**，所有序列沿 `T` 首尾相接。
  `chunk.py:217-222` 显式 `raise ValueError`（“The batch size is expected to be 1 ... Please
  flatten variable-length inputs”）。**已核实。**
* **GQA**：`k`/`q` 的头数记 `HK`（kernel 里叫 `Hg`），`v`/`beta`/`o`/`w`/`h` 的头数记 `HV`
  （kernel 里叫 `H`）。**`H` 永远取自 `beta.shape[-1]` / `v.shape[-2]`，不是 `k` 的**。
  **已核实（SG-kkt:120-122、SG-wy_fast:120-122、SG-delta_h:285-286、SG-chunk_o:136-137）。**
  k 头 → v 头的映射是 `i_h // (H // Hg)`（SG-kkt:63、SG-wy_fast:90、SG-delta_h:85、
  SG-chunk_o:70-71）。**已核实。**
* **`BT = 64` 恒定**（见 §3.4）。

### 2.1 算子 1：`chunk_scaled_dot_kkt_fwd`

**签名（形参名以 `chunk.py:41-48` 的实参为准；`chunk_size` 的存在取自 SG-kkt:94，推断）：**

```python
# chunk_scaled_dot_kkt.py
def chunk_scaled_dot_kkt_fwd(
    k: torch.Tensor,                            # [B, T, Hg, K]
    beta: torch.Tensor,                         # [B, T, H]
    g: torch.Tensor | None = None,              # [B, T, H]   ← vLLM 叫 g；SG 叫 g_cumsum
    cu_seqlens: torch.Tensor | None = None,     # [N+1]
    chunk_indices: torch.Tensor | None = None,  # [NT, 2]     ← vLLM 有，SG 没有
    chunk_size: int = 64,
    output_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:                              # [B, T, H, BT]
```

* 形参 `k` / `beta` / `g` / `cu_seqlens` / `chunk_indices` / `output_dtype` —— **已核实
  （`chunk.py:41-48`）**：`k=k, beta=beta, g=g, cu_seqlens=cu_seqlens,
  chunk_indices=chunk_indices, output_dtype=torch.float32`。
* 门控形参**名为 `g`**：**已核实（`chunk.py:44` 关键字 `g=g`）**。注意此时实参已经是
  `chunk_local_cumsum` 的输出，即**kernel 里的 `g_cumsum` 语义、形参名 `g`**。这是最容易看错的
  一处（SG 副本函数名与形参都还叫 `g_cumsum`，SG-kkt:92）。
* `chunk_size: int = 64` 与 `output_dtype` 默认 `torch.float32`：**已核实（SG-kkt:94-95）**；
  vLLM 侧是否仍有该形参——**推断**（`chunk.py` 未传，但默认值可覆盖，不传即 64）。

**张量契约：**

| 名字 | 形状 | dtype | 来源 |
|---|---|---|---|
| `k` | `[B, T, Hg, K]` | `k.dtype`（生产环境 bf16） | 已核实 SG-kkt:120 |
| `beta` | `[B, T, H]`（**3 维**） | `beta.dtype` | 已核实 SG-kkt:122 + `chunk.py:216` 断言 `len(beta.shape)==3` |
| `g` | `[B, T, H]` | **fp32** | 已核实 `fla_cumsum.py:250` 默认 `output_dtype=torch.float`；`chunk.py:37` 未覆盖 |
| `A`（返回） | `[B, T, H, BT]` | `output_dtype` = **fp32** | 已核实 SG-kkt:128 + `chunk.py:47` |

**关键：`A` 的 `H` 是 `beta.shape[-1]`，不是 `k.shape[2]`（SG-kkt:122 `H = beta.shape[-1]`，
`A = torch.empty(B, T, H, BT)` 在 SG-kkt:128）。**已核实。**

**kernel 数学（SG-kkt:53-86，全部已核实行号）：**

```
b_beta = load(beta)                                   # SG-kkt:58   [BT]
b_A    = dot(b_k, trans(b_k))                         # SG-kkt:71   [BT,BT] fp32
b_g_diff = b_g[:, None] - b_g[None, :]                # SG-kkt:78   g_i - g_j
b_A    = b_A * safe_exp(b_g_diff)                     # SG-kkt:79
b_A   *= b_beta[:, None]                              # SG-kkt:81   行缩放 beta_i
b_A    = tl.where(o_t[:,None] > o_t[None,:], b_A, 0)  # SG-kkt:82   严格下三角
store to A[(bos*H + i_h)*BT ...] shape (T,BT) stride (BT*H,1)   # SG-kkt:83-86
```

* **`A` 的内存布局**：把 `[B,T,H,BT]` 看成每个 `(b,h)` 一个 `(T, BT)` 矩阵，**行 = token，
  列 = chunk 内的相对下标**，行步长 `H*BT`、列步长 `1`（SG-kkt:84）。**已核实。**
  这和 `fla_solve_tril.py:68-75` 的 `(T, BT), (H*BT, 1)` 完全一致，是 `A` 能直接喂给
  `solve_tril` 的前提。
* `A` 用 `torch.empty` 分配（SG-kkt:128），**不是 `zeros`** → 序列尾部不足一个 chunk 的行是
  **未初始化垃圾**。`solve_tril` 从 `torch.zeros_like` 出发且 store 带 `boundary_check`，
  所以不读这些行（`fla_solve_tril.py:539, 94-98`）。**已核实。**

### 2.2 算子 2：`recompute_w_u_fwd`（wy_fast）

**签名（vLLM 与 SG 形参名一致，可直接引用）：**

```python
# wy_fast.py
def recompute_w_u_fwd(
    k: torch.Tensor,                            # [B, T, Hg, K]
    v: torch.Tensor,                            # [B, T, H,  V]
    beta: torch.Tensor,                         # [B, T, H]
    g_cumsum: torch.Tensor,                     # [B, T, H]   ← 这里叫 g_cumsum（不是 g）
    A: torch.Tensor,                            # [B, T, H, BT]  = solve_tril 的输出
    cu_seqlens: torch.Tensor | None,
    chunk_indices: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:         # (w, u)
```

* 形参名 `k/v/beta/A/g_cumsum/cu_seqlens/chunk_indices` —— **已核实（`chunk.py:52-60`）**，
  注意这里是 **`g_cumsum=g`**（`chunk.py:57`），与算子 1 的 `g=g` 不同名，**同一个张量两个名字**。
* 形参顺序 SG 是 `(k, v, beta, g_cumsum, A, ...)`（SG-wy_fast:111-118），vLLM 的调用是
  `A=A` 在 `g_cumsum=g` 之前；因为**全部用关键字传递**，顺序不构成契约。
  **已核实（`chunk.py:52-60` 全为关键字）**。

**张量契约：**

| 名字 | 形状 | dtype | 来源 |
|---|---|---|---|
| `k` | `[B, T, Hg, K]` | `k.dtype` | 已核实 SG-wy_fast:120 |
| `v` | `[B, T, H, V]` | `v.dtype` | 已核实 SG-wy_fast:120 |
| `beta` | `[B, T, H]` | `beta.dtype` | 已核实 SG-wy_fast:121 |
| `g_cumsum` | `[B, T, H]` | fp32 | 已核实（同 §2.1） |
| `A` | `[B, T, H, BT]` | **`k.dtype`（bf16）** | 已核实（`chunk.py:50` `output_dtype=k.dtype`） |
| `u`（返回） | `[B, T, H, V]` | **`v.dtype`** | 已核实 SG-wy_fast:129 `torch.empty_like(v)` |
| `w`（返回） | `[B, T, H, K]` | **`k.dtype`** | 已核实 SG-wy_fast:130 `k.new_empty(B,T,H,K)` |

**注意 `w` 的头数是 `H`（v 头数），宽度是 `K`；`u` 与 `v` 同形。**已核实（SG-wy_fast:129-130）。

**kernel 数学（SG-wy_fast:55-108，全部已核实）：**

```
b_g = tl.exp(load(g_cumsum))                                  # SG-wy_fast:64  ← 普通 exp
b_vb = (b_v * b_beta[:, None]).to(b_v.dtype)                  # SG-wy_fast:84  ← 先降回 bf16
b_u  = tl.dot(b_A, b_vb, allow_tf32=False)                    # SG-wy_fast:85
b_kb = (b_k * b_beta[:, None] * b_g[:, None]).to(b_k.dtype)   # SG-wy_fast:106 ← 再降回 bf16
b_w  = tl.dot(b_A, b_kb)                                      # SG-wy_fast:107
```

即（WY 表示）：

```
u = A_inv @ (v * beta)                 # beta 按行缩放，逐 head
w = A_inv @ (k * beta * exp(g))        # exp(g) 逐 head 标量
```

* **`beta` 只按行乘（`[:, None]`），不乘列、不元素级**。**已核实（SG-wy_fast:84,106）。**
* **`b_w` 的 dot 没有 `allow_tf32=False`，而 `b_u` 有**（SG-wy_fast:85 vs 107）—— 这暗示
  `w` 允许 tf32、`u` 要求 ieee。**已核实（两行原文对照）**。torch 侧无 tf32 开关时按 ieee 做更准，
  但不与 Triton 逐位一致（见 §5.3）。

### 2.3 算子 3：`chunk_gated_delta_rule_fwd_h`

**签名（vLLM 0.23 形参由 `chunk.py:61-71` 实参确定；返回 3 元由 `chunk.py:61` 的
三元解包 + `chunk.py:86` 确定）：**

```python
# chunk_delta_h.py
def chunk_gated_delta_rule_fwd_h(
    k: torch.Tensor,                            # [B, T, Hg, K]
    w: torch.Tensor,                            # [B, T, H,  K]
    u: torch.Tensor,                            # [B, T, H,  V]   ← kernel 里叫 v
    g: torch.Tensor,                            # [B, T, H]  fp32
    initial_state: torch.Tensor,                # [N, H, V, K]
    output_final_state: bool,
    cu_seqlens: torch.Tensor | None = None,
    chunk_indices: torch.Tensor | None = None,  # 收下但 kernel 不用（见下）
    chunk_offsets: torch.Tensor | None = None,  # [N+1]
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:   # (h, v_new, final_state)
```

* 形参名 `k/w/u/g/initial_state/output_final_state/cu_seqlens/chunk_indices/chunk_offsets`
  —— **已核实（`chunk.py:61-71`，全部关键字）**。
* 返回 **3** 元 —— **已核实（`chunk.py:61` `h, v_new, final_state = ...`；`chunk.py:86`）**。
  **这不是 SG 版本（SG 返回 2 元，SG-delta_h:340）。**
* vLLM 0.23 **不再有 `initial_state_indices`**：`chunk.py:61-71` 没有传；生产侧改为在调用前
  用 gather 造一份 `initial_state`（`qwen_gdn_linear_attn.py:1513-1514`
  `initial_state = ssm_state[prefill_state_indices]`）。**已核实。**
* `chunk_indices` 被接收但 kernel 不用：varlen 分支只读 `cu_seqlens` 和 `chunk_offsets`
  （SG-delta_h:61-67）。**已核实（SG-delta_h:42-44 形参表里根本没有 `chunk_indices`）**。

**张量契约：**

| 名字 | 形状 | dtype | 来源 |
|---|---|---|---|
| `k` | `[B, T, Hg, K]` | `k.dtype` | 已核实 SG-delta_h:85,285 |
| `w` | `[B, T, H, K]` | `k.dtype` | 同上 |
| `u` | `[B, T, H, V]` | `v.dtype` | 已核实 SG-delta_h:84,285 |
| `g` | `[B, T, H]` | fp32 | 已核实（§2.1） |
| `initial_state` | **`[N, H, V, K]`** | 状态 dtype | 已核实 `chunk.py:169` docstring；`qwen_gdn_linear_attn.py:1130-1137` 造的是 `[1, HV, V, K]` |
| `h`（返回） | **`[B, NT, H, V, K]`** | **`k.dtype`（bf16！）** | 已核实 SG-delta_h:305 `k.new_empty(B, NT, H, V, K)` |
| `v_new`（返回） | `[B, T, H, V]` | `v.dtype` | 已核实 SG-delta_h:307 `torch.empty_like(u)` |
| `final_state` | `[N, H, V, K]` | 同 `initial_state` | 推断（`chunk.py:61` 返回；形状承 `chunk.py:169`） |

* **`h` 的轴序是 `[..., V, K]`，不是 `[..., K, V]`**。**已核实（SG-delta_h:104 的
  `make_block_ptr(h0, (V, K), (K, 1), ...)` 与 SG-delta_h:305 的 `new_empty(B,NT,H,V,K)`）。**
* **`h` 在内存里按“全局 chunk 序号”扁平存放**：`h += ((boh * H + i_h) * V * K)`
  （SG-delta_h:83），varlen 时 `boh = chunk_offsets[i_n]`（SG-delta_h:67），dense 时
  `boh = i_n * NT`（SG-delta_h:71）。`chunk_o` 用同一个公式反查（SG-chunk_o:74
  `h += (i_tg * H + i_h) * V * K`，其中 `i_tg = i_b*NT + i_t`）。**已核实。**
  这就是 `chunk_offsets` 存在的唯一理由。

**kernel 数学（SG-delta_h:123-251，全部已核实）：**

```
for i_t in range(NT):
    store h[i_t] = b_h1                        # SG-delta_h:127  ← 先存，后更新
    b_v = dot(b_w, trans(b_h1))                # SG-delta_h:148
    b_v = load(u) - b_v                        # SG-delta_h:170  ← u - w @ h^T
    if USE_G:
        b_v = b_v * safe_exp(b_g_last - b_g)[:, None]   # SG-delta_h:185
        b_h1 = b_h1 * exp(b_g_last)                     # SG-delta_h:186-187
    b_v = b_v.to(k.dtype)                      # SG-delta_h:227
    b_h1 += trans(dot(b_k, b_v))               # SG-delta_h:233
```

要点：

* **`h[i_t]` 是 chunk `i_t` 的“进块状态”（更新前）**，因为 store 在循环体最前面
  （SG-delta_h:127），而状态更新在最后（SG-delta_h:233）。**已核实。**
* `b_g_last` 取的是**该 chunk 内最后一个有效 token** 的 `g`：
  `last_idx = min((i_t+1)*BT, T) - 1`（SG-delta_h:178-180）。**尾块不满时不是 `g[BT-1]`**。
  **已核实。**
* 递推式：`S_{c+1} = S_c·exp(g_last) + k_c^T @ [ (u_c - w_c S_c^T) * exp(g_last - g_t) ]`。

### 2.4 算子 4：`chunk_fwd_o`

**签名（形参由 `chunk.py:72-82` 实参确定）：**

```python
# chunk_o.py
def chunk_fwd_o(
    q: torch.Tensor,                            # [B, T, Hg, K]
    k: torch.Tensor,                            # [B, T, Hg, K]
    v: torch.Tensor,                            # [B, T, H,  V]   ← 实参是 v_new
    h: torch.Tensor,                            # [B, NT, H, V, K]
    g: torch.Tensor,                            # [B, T, H]  fp32
    scale: float,
    cu_seqlens: torch.Tensor | None = None,
    chunk_indices: torch.Tensor | None = None,
    core_attn_out: torch.Tensor | None = None,
) -> torch.Tensor:                              # [B, T, H, V]
```

* 形参名 `q/k/v/h/g/scale/cu_seqlens/chunk_indices/core_attn_out` —— **已核实
  （`chunk.py:72-82`，全部关键字）**。
* **没有 `chunk_size`**：SG 版有（SG-chunk_o:134），vLLM 换成了 `chunk_indices`。
  **已核实（`chunk.py:72-82` 无 `chunk_size` 实参，有 `chunk_indices`）。**
* `core_attn_out` —— **已核实（`chunk.py:81`）**；语义是“预分配输出缓冲、原地写”，
  只在推理期允许（`chunk.py:129-133`）。**具体写整块还是写前缀，vLLM 本体未拿到，无法确定**（§7）。

**张量契约：**

| 名字 | 形状 | dtype | 来源 |
|---|---|---|---|
| `q` | `[B, T, Hg, K]` | `q.dtype` | 已核实 SG-chunk_o:136 |
| `k` | `[B, T, Hg, K]` | `k.dtype` | 同上 |
| `v` | `[B, T, H, V]` | `v.dtype` | 已核实 SG-chunk_o:136-137（`H = v.shape[-2]`） |
| `h` | `[B, NT, H, V, K]` | `k.dtype` | 已核实 SG-chunk_o:74 + SG-delta_h:305 |
| `g` | `[B, T, H]` | fp32 | 已核实（§2.1） |
| `scale` | Python `float` | — | 已核实 SG-chunk_o:39（kernel 标量参数） |
| `o`（返回） | `[B, T, H, V]` | **`v.dtype`** | 已核实 SG-chunk_o:146 `torch.zeros_like(v)` |

**kernel 数学（SG-chunk_o:76-123，全部已核实）：**

```
b_o  = dot(b_q, trans(b_h))                  # SG-chunk_o:97   [BT, BV]
b_A  = dot(b_q, b_k)                          # SG-chunk_o:99   [BT, BT]
if USE_G:
    b_o = b_o * exp(b_g)[:, None]             # SG-chunk_o:105
    b_A = b_A * safe_exp(b_g[:,None]-b_g[None,:])  # SG-chunk_o:106
m_A  = o_i[:,None] >= o_i[None,:]             # SG-chunk_o:109  ← 含对角线
b_A  = tl.where(m_A, b_A, 0)                  # SG-chunk_o:110
b_o  = b_o * scale + tl.dot(b_A.to(b_v.dtype), b_v) * scale   # SG-chunk_o:122
```

* **`scale` 同时乘两项**（`b_o` 和块内项），见 SG-chunk_o:122。**已核实。**
* 默认 `scale` 用 **`k.shape[-1]`**（不是 `q`）：`chunk.py:228-229`
  `scale = k.shape[-1] ** -0.5`。**已核实。**
* **`h` 被强转成 `b_v.dtype` 再做 dot**（SG-chunk_o:122 的 `.to(b_v.dtype)`；SG-chunk_o:94 加载
  `b_h` 后 SG-chunk_o:97 的 `tl.trans(b_h)` 未转、随 dot 提升）。**已核实（行号见上）。**

---

## 3. `A` / `g` / `beta` / `BT` 的定义，以及 `solve_tril` 怎么插进来

### 3.1 `A`（转移矩阵 / transition matrix）

逐 chunk、逐 head、块内严格下三角：

```
A[i, j] = beta_i * exp(g_i - g_j) * (k_i · k_j)     若 i > j（同一 chunk 内）
A[i, j] = 0                                          否则
```

* 三项因子与顺序：`k_i·k_j`（SG-kkt:71）→ `* exp(g_i-g_j)`（SG-kkt:79）→ `* beta_i`
  （SG-kkt:81）。**已核实。**
* **严格下三角**：`tl.where(o_t[:,None] > o_t[None,:], b_A, 0)`（SG-kkt:82），是 `>` 不是 `>=`，
  所以**对角线恒为 0**。**已核实。** 与 `fla_solve_tril.py:515` 的 host 断言
  “A should be strictly lower triangular, i.e., `A.triu() == 0`” 一致。**已核实。**
* 由 §2.1 的 `A` 布局：行 = token、列 = chunk 内相对下标，所以“上/下三角”是在
  `(T, BT)` 的**块内**意义上说的，跨 chunk 不成立（每个 chunk 在自己的 `BT` 列里各成一个三角）。

### 3.2 `g`（cumulative log-decay）

* **语义**：log 空间的 forget gate（`chunk.py:162` 原文 “(forget) Gating tensor (**in log
  space!**) of shape `[B, T, H]`”）。**已核实。**
* **定义**：`g = chunk_local_cumsum(g_raw, chunk_size=BT)`，即**每个 chunk 内重新起算**的前缀和；
  `cu_seqlens` 非空时在序列边界也重新起算（`fla_cumsum.py:27-71` 的 kernel：`b_o = tl.cumsum(b_s, 0)`，
  `REVERSE=False`；本仓库已移植版的 docstring“The sum restarts at every chunk boundary”
  （`tileinfer/vllm/ops/gdn.py:67-68`）也是同一结论）。
  **已核实（`fla_cumsum.py:67`；`chunk.py:37-39`）。**
* **dtype 是 fp32**：`chunk_local_cumsum` 的 `output_dtype` 默认 `torch.float`
  （`fla_cumsum.py:250`），`chunk.py:37-39` 没有覆盖它。**已核实。**
  即 `g` 输入是 bf16，`g` 输出是 fp32，**四个消费者拿到的都是 fp32 的 `g`**。
* **数值范围**：`g` 是负数且 chunk 内单调不增（log-decay 的累积），所以 `g_i ≤ 0`。
  这一条决定了 §5.1 的 `exp`/`safe_exp` 分工。

### 3.3 `beta`

* 形状 `[B, T, H]`，**3 维**，`chunk.py:216` 有显式断言 `len(beta.shape) == 3`。**已核实。**
* **只在两个地方消费**：kkt（行缩放，SG-kkt:81）和 wy_fast（行缩放，SG-wy_fast:84,106）。
  **`chunk_delta_h` 和 `chunk_fwd_o` 完全不碰 `beta`**。**已核实（`chunk.py:61-71`、`chunk.py:72-82`
  的实参表里都没有 `beta`）。**
* **“masked/multiplied”的确切含义**：是**行方向的乘法缩放**，`b_A *= b_beta[:, None]`
  （SG-kkt:81），即 `A[i,j] ← beta_i · A[i,j]`，**乘的是 query 侧下标 `i`，不是 key 侧 `j`**，
  也**不是**加性 mask、**不是**与三角 mask 融合。三角 mask 是**后一步单独做的**
  （SG-kkt:82），用的是 `tl.where` 选择，**不是乘 0/1**。**已核实。**
* 在 wy_fast 里 `beta` 同样只按行乘：`v * beta`（SG-wy_fast:84）、`k * beta`（SG-wy_fast:106）。

### 3.4 chunk size `BT`

* `BT = FLA_CHUNK_SIZE`（`chunk.py:38`），且 **`FLA_CHUNK_SIZE == 64`**。
  * **已核实（间接但很硬）**：`qwen_gdn_linear_attn.py:1085-1087`：“All kernels including
    `chunk_fwd_kernel_o` now use a fixed ``BT = chunk_size`` (64). A single warmup pass with
    T = 64 is sufficient”。同文件 `:1107` `T = FLA_CHUNK_SIZE` 且 `:1086` 说这个 T 就是 64。
  * **已核实（SG 侧旁证）**：`SG-chunk_delta_h.py:19 CHUNK_SIZE = 64`；
    `SG-kkt:94 chunk_size: int = 64`；`SG-chunk_o:134 chunk_size: int = 64`。
  * **推断（依据缺失文件）**：`FLA_CHUNK_SIZE` 的字面定义在 `fla/ops/utils.py`，该文件本轮**没拿到**，
    所以“字面等于 64”这一句本身是推断；但上面两条已经把 64 钉死。
* `BT` 必须是 16/32/64 之一：`fla_solve_tril.py:531` `assert A.shape[-1] in [16, 32, 64]`。
  **已核实。**
* **vLLM 0.23 固定 `BT = 64`，不随 `T` 收缩。** 这一点与 SG 副本不同（SG-chunk_o:138 会
  `min(chunk_size, max(16, next_pow2(T)))`）。**已核实（`qwen_gdn_linear_attn.py:1085-1087`）。**
  所以 `T < 64` 时也必须按 64 分块并零填充，**照抄 SG 会在短序列上改变数值**。

### 3.5 `solve_tril` 如何接入（返回 `(I+A)^{-1}`）

* 语义：`out = (I + A_kkt)^{-1}`，在**每个 `BT×BT` 块内**、逐 head 求逆；输入 `A_kkt` 的
  严格下三角被保留、对角与上三角被丢弃，输出带单位对角。**已核实（`fla_solve_tril.py:515-530`
  docstring、`:65-66` 的 `m_A`/`m_I`、`:82` `b_A = -tl.where(m_A, b_A, 0)`、`:89` `b_A += m_I`）。**
* **两次 dtype 改写是契约的核心**：
  1. `A_kkt` 由 kkt 产出，`output_dtype=torch.float32`（`chunk.py:47`）→ **fp32**。
  2. `solve_tril(..., output_dtype=k.dtype)`（`chunk.py:50`）→ **`A_inv` 是 bf16**。
  3. 这个 bf16 的 `A_inv` 就是 wy_fast 的 `A`（`chunk.py:56` `A=A`），直接进两个 dot。
  **已核实（三处行号）。**
* **块间无耦合**：每个 chunk 独立求逆，`BT` 个列全属本 chunk（`fla_solve_tril.py:68-75`），
  跨 chunk 不传播。**已核实。**
* **尾部零行语义**：`Ai = torch.zeros_like(A, ...)`（`fla_solve_tril.py:539`）且 store 带
  `boundary_check`，所以不满一个 chunk 的尾部行是 **0，不是单位阵**。**已核实。**
  这意味着 `u = A_inv @ (v*beta)` 在这些行上得到 0 —— 而下游 `delta_h`/`chunk_o` 也只写有效行，
  所以 0 不会再被读到。**推断（依据两条 store 的 `boundary_check`：SG-delta_h:127-176、
  SG-chunk_o:123）。**
* 与数学式的关系：`(I+A)^{-1}` 而非 `(I-A)^{-1}`。`vllm-api-0.23.md:103` 那句 “`(I - A)`”
  是**错的**，`fla_solve_tril.py:82,89` 与 `tileinfer/vllm/ops/fla/solve_tril.py:44-47` 都是 `(I+A)`。
  **已核实。**

---

## 4. varlen（`cu_seqlens`）与 `chunk_indices` 的共享约定

### 4.1 `cu_seqlens`

* `[N+1]`，累计 token 偏移，`cu_seqlens[0] == 0`、`cu_seqlens[-1] == T`。**已核实
  （`chunk.py:174-176` docstring “Cumulative sequence lengths of shape `[N+1]`”；示例
  `chunk.py:204` `[0, 2048, 4096, 6144, 8192]`）。**
* dtype **int32**：`chunk.py:204` 示例 `dtype=torch.int32`；`qwen_gdn_linear_attn.py:1138`
  `torch.tensor([0, T], device=..., dtype=torch.int32)`。kernel 侧再 `.to(tl.int32)`
  （SG-kkt:47、SG-chunk_o:59）。**已核实。**
* 非空时 `B` 必须为 1，且 `initial_state.shape[0] == len(cu_seqlens) - 1`，两条都有显式
  `raise`（`chunk.py:217-227`）。**已核实。**

### 4.2 `chunk_indices`

* `[NT, 2]`，每行 `(i_n, i_t)`：`i_n` = 序列下标（`cu_seqlens` 的下标），`i_t` = **序列内**的
  chunk 下标（不是全局 chunk 下标）。**已核实（SG-index.py:16-26 `prepare_chunk_indices`；
  四个 kernel 的读取方式一致：`chunk_indices + i_t*2` 取 `i_n`、`+1` 取 `i_t`，
  SG-kkt:44-46、SG-wy_fast:46-48、SG-chunk_o:56-58；`solve_tril` 同 `fla_solve_tril.py:53-56`）。**
* 生成规则：对每个序列取 `cdiv(len, BT)` 个 `(i_n, i_t)`，**sequence-major、`i_t` 递增**
  （SG-index.py:20-26）。**已核实。**
* **可选性与派生**：`chunk_indices is None and cu_seqlens is not None` 时内部派生
  （SG-kkt:124-126、SG-wy_fast:124-126、SG-chunk_o:139-141、`fla_solve_tril.py:535-537`）。
  即**上层可以不传，传了就必须与 `cu_seqlens`+`BT` 自洽**。**已核实。**
* **`chunk_indices` 只在 varlen 分支被读**：`IS_VARLEN` 由 `cu_seqlens is not None` **单独**决定
  （`fla_solve_tril.py:28` 的 `triton.heuristics`；各 kernel 启动时传
  `IS_VARLEN=cu_seqlens is not None`，SG-kkt:142、SG-wy_fast:149、SG-chunk_o:170）。
  所以 dense 情况下传了 `chunk_indices` 也不生效。**已核实。**

### 4.3 `chunk_offsets`

* `[N+1]`，**chunk 计数的累计和**（不是 token 数），`prepare_chunk_offsets`
  = `cat([0], cdiv(lens, BT)).cumsum(-1)`（SG-index.py:29-35）。**已核实。**
* **只有 `chunk_delta_h` 用得到**：它把 `h` 按全局 chunk 序号扁平存放，varlen 时
  `boh = chunk_offsets[i_n]`（SG-delta_h:67）。dense 时 `boh = i_n * NT`（SG-delta_h:71）。
  **已核实。**
* vLLM 0.23 把它**提升为 `chunk_gated_delta_rule_fwd_h` 的显式形参**（`chunk.py:70`），
  而 SG 版在函数内部自己算（SG-delta_h:298-302）。**这是签名差异，已核实（`chunk.py:70` vs
  SG-delta_h:289-302）。**

### 4.4 必填 / 可选 一览

| 算子 | `cu_seqlens` | `chunk_indices` | `chunk_offsets` | 其他 varlen 相关 |
|---|---|---|---|---|
| `chunk_scaled_dot_kkt_fwd` | 可选 | 可选（vLLM 有该形参；不传则内部派生） | 不用 | `IS_VARLEN` 只看 `cu_seqlens` |
| `recompute_w_u_fwd` | 可选 | 可选（同上） | 不用 | 同上 |
| `chunk_gated_delta_rule_fwd_h` | 可选 | 可选但**kernel 不读** | **varlen 时必须**（vLLM 由调用方传） | 靠 `chunk_offsets` 定位 `h` |
| `chunk_fwd_o` | 可选 | 可选（同上） | 不用 | `i_tg = i_b*NT + i_t` |

**已核实（`chunk.py:41-82` 的实参 + 上表各 kernel 行号）。**
注意 `cu_seqlens=None` 时 `NT = triton.cdiv(T, BT)`，非空时 `NT = len(chunk_indices)`
（`fla_solve_tril.py:537`、SG-kkt:127、SG-chunk_o:142）—— **grid 大小本身就不一样**，
所以“传了 cu_seqlens 但内部还按 cdiv(T,BT) 循环”会多算或漏算。

---

## 5. 最容易静默出错的地方

### 5.1 `safe_exp` 的非对称钳位，以及 `exp` / `safe_exp` 的分工

```python
# SG-op.py:25-27
@triton.jit
def safe_exp(x):
    return exp(tl.where(x <= 0, x, float("-inf")))
```

**已核实（SG-op.py:25-27）。** 语义：**任何正的指数参数被替换成 `-inf`，结果精确为 0.0**，
而不是 `exp(x)`。这是个“单边截断”，不是普通的 `exp`。

四个算子里 `exp` 与 `safe_exp` 的分工**严格按“绝对值 vs 差值”**：

| 位置 | 用法 | 参数 | 已核实 |
|---|---|---|---|
| kkt | **`safe_exp`** | `g_i - g_j`（可正可负） | SG-kkt:79 |
| wy_fast | **`exp`** | `g_i`（≤0） | SG-wy_fast:64 |
| delta_h | **`safe_exp`** | `g_last - g_i`（≤0，但尾块/乱序仍需保护） | SG-delta_h:185 |
| delta_h | **`exp`** | `g_last`（≤0） | SG-delta_h:186 |
| chunk_o | **`exp`** | `g_i`（≤0） | SG-chunk_o:105 |
| chunk_o | **`safe_exp`** | `g_i - g_j`（可正可负） | SG-chunk_o:106 |

**结论（可直接当规则用）：`exp` 只用在“绝对的 `g`”（必然 ≤0）；`safe_exp` 只用在“`g` 的差”
（可正）。**已核实（上表六行）。torch 侧若用 `torch.exp` 代替 `safe_exp`，在 `i<j` 区域会算出
`exp(正数)`，`g` 的 chunk 内累积衰减可达数百度（`g_i` 量级 ~ `BT × |logsigmoid|`），
fp32 下 `exp(>88)` 直接 `inf`。

**配套陷阱（更隐蔽）：掩码必须用 `torch.where` 选择，不能用“乘 0/1 mask”。**
`tl.where(mask, b_A, 0)` 是**丢弃**（`inf` 被丢掉，结果 0），而 `b_A * mask` 会得到
`inf * 0 = nan`。Triton 原文六处掩码全是 `tl.where`（SG-kkt:82、SG-chunk_o:110），
**已核实**。torch 侧若写成乘法，NaN 会顺着 `solve_tril` → `w/u` → `h` 一路扩散到整个输出，
而且**只在 `g` 的衰减足够大时**才复现——小规模单测全绿，上真模型才炸。

### 5.2 两个不同的三角掩码：`>` vs `>=`

| 位置 | 掩码 | 对角线 | 已核实 |
|---|---|---|---|
| kkt（造 `A`） | `o_t[:,None] > o_t[None,:]` | **去掉**（严格下三角） | SG-kkt:82 |
| chunk_o（块内注意力） | `o_i[:,None] >= o_i[None,:]` | **保留**（token 注意自己） | SG-chunk_o:109 |

**已核实（两行原文）。** 同一个链路里两个“三角”不是同一个三角。踩错后的表现很不对称：

* chunk_o 用了 `>`：**丢掉对角自项**，`o` 系统性偏小（每个 token 少了 `scale·v_i` 这一项），
  误差量级 O(1)，但输出仍然“看起来合理”。
* kkt 用了 `>=`：`A` 的对角线上多出 `beta_i·(k_i·k_i)`。它**不会报错、也不会影响结果**，
  因为 `solve_tril` 的 `b_A = -tl.where(m_A, b_A, 0)`（`fla_solve_tril.py:82`）只读严格下三角，
  对角被静默丢弃（`tileinfer/vllm/ops/fla/solve_tril.py:133-139` 明确记为 “Deliberate deviation”）。
  **所以这个 bug 在自己的单测里永远看不见，只在别人消费 `A` 时才暴露。**

### 5.3 bf16 的来回降精度（三处，都是“契约”不是“实现细节”）

1. **`A_inv` 是 bf16**：`solve_tril(..., output_dtype=k.dtype)`（`chunk.py:50`）。**已核实。**
2. **wy_fast 的两个 dot 之前先降回 bf16**：
   `(b_v * b_beta[:,None]).to(b_v.dtype)`（SG-wy_fast:84）、
   `(b_k * b_beta[:,None] * b_g[:,None]).to(b_k.dtype)`（SG-wy_fast:106）。**已核实。**
   即 `beta`/`exp(g)` 的缩放**在 fp32 里做、然后截断成 bf16**，再进 dot。
3. **`h` 是 bf16**：`h = k.new_empty(...)`（SG-delta_h:305），每次 store 都
   `.to(p_h1.dtype.element_ty)`（SG-delta_h:127）。**已核实。** 也就是**每个 chunk 边界上状态都被
   舍入到 bf16 一次**，64 token 一个回合，长序列会累积可见漂移。

一个全程 fp32 的 torch 实现**更准但不逐位一致**。要“bit-comparable”必须复现这三处截断。
**推断（依据：Triton 的 `.to(dtype)` 语义 + `new_empty`/`empty_like` 的 dtype 继承）。**

### 5.4 其他静默陷阱

* **`chunk_fwd_o` 的 `v` 实参是 `v_new`**（`chunk.py:75`），`delta_h` 的 `u` 在 kernel 里叫 `v`
  （SG-delta_h:314）。两处对调都会“跑出一个形状完全正确的错误答案”。**已核实。**
* **`h[i_t]` 是进块状态**，store 在循环体开头（SG-delta_h:127），更新在结尾（SG-delta_h:233）。
  用“更新后的 h”做 `chunk_o` 会让每个 chunk 错位一格。**已核实。**
* **`b_g_last` 用 `min((i_t+1)*BT, T)-1`**（SG-delta_h:178），尾块不是 `g[BT-1]`。**已核实。**
* **`A` 是 `torch.empty`**（SG-kkt:128），尾部是垃圾；`o` 是 `torch.zeros_like`
  （SG-chunk_o:146），尾部是 0。**已核实。** 一个 `empty`、一个 `zeros`，别互换。
* **`w` 的头数是 `H`（v 头数）而宽度是 `K`**（SG-wy_fast:130），GQA 下 `HK ≠ HV`，
  写成 `[B,T,Hg,K]` 会在 `delta_h` 的 `dot(b_w, ...)` 处形状对不上（但那是对不上，
  更好的一种情况——真正的静默错误见 §5.2）。
* **`scale` 默认取 `k.shape[-1]`**（`chunk.py:228-229`），不是 `q`、也不是 `v`。**已核实。**
* **`contiguous()` 要求**：`@input_guard` 把**每个** tensor 实参先 `.contiguous()`
  （SG-utils.py:145-179；`fla_solve_tril.py:506` 也挂着）。kernel 用常量算地址
  （`H*BT`、`1`），**非连续输入会被静默误读**。torch 版按逻辑下标走，**不会报错也不会误读**，
  于是“Triton 对、torch 也对、但喂非连续张量时两者不一致”。要一致就得显式
  `.contiguous()`，或至少断言。**已核实（SG-utils.py:152-158 原文）。**
* **`FLA_USE_FAST_OPS`**：`exp`/`exp2`/`log` 会被换成 `fast_expf` 等近似实现
  （SG-op.py:13-22）。**已核实。** 所以“逐位一致”本身依赖于这个环境变量为 `0`（默认）。
* **`.exp()` 而不是 `.exp2()`**：这四个算子**全都没有用 `exp2`**（SG-op.py:15 只定义，
  四个文件里 `exp2` 零引用，见 §2 各 kernel 的引用行）。**已核实（四文件 grep：只出现
  `exp` / `safe_exp`）。** 用 `exp2` 重写（`exp2(x * log2e)`）会引入 ~1 ulp 差异，
  在 bf16 下不可见、在 fp32 对照下可见。

---

## 6. 逐算子移植清单（要复现什么 / 最便宜的 oracle）

前置：本仓库已有 fp64 参照 `tileinfer/testing/gdn_reference.py`，覆盖前两个算子，
**直接用它当 oracle，不要新写**。

### 6.1 `chunk_scaled_dot_kkt_fwd`

必须复现：

1. `A[i,j] = beta_i · exp(g_i - g_j) · (k_i·k_j)`，`i>j` 块内；对角与上三角**精确 0**。
2. 门控用 **`safe_exp`**（正指数 → 0），掩码用 `torch.where`（§5.1）。
3. `beta` 行缩放（`beta[:, None]`，查 query 侧下标）。
4. 输出 `[B, T, H, BT]`、`output_dtype`（vLLM 传 fp32）、`H = beta.shape[-1]`。
5. k 的 GQA 映射 `k[..., i_h // (H//Hg), :]`。
6. 尾部不足 `BT` 的行：**不写**（允许是垃圾），别去 `nan_to_num` 或补 1。

最便宜 oracle：`gdn_reference.kkt_reference(k, beta, g_cumsum, cu_seqlens, chunk_size=64,
dtype=torch.float64)`（`gdn_reference.py:23-52`）。它已按
`beta_i·exp(g_i-g_j)·(k_i·k_j)` 且 `tril(-1)` 严格下三角实现（`gdn_reference.py:46-50`）。
最小用例：`B=1, T=64, HV=2, HK=1, K=8`，`cu_seqlens=[0,64]`；断言 `A.triu()==0`；
bf16 输入比对用 `rtol≈2e-2`，fp32 输入可收到 `1e-5`。
再用 `cu_seqlens=[0,37,64]` 覆盖尾块。

### 6.2 `recompute_w_u_fwd`

必须复现：

1. `u = A_inv @ (v * beta[:,None])`，`w = A_inv @ (k * beta[:,None] * exp(g)[:,None])`。
2. **`A_inv` 是解 `(I+A)` 得到的**，且上游传进来的已经是 bf16（`chunk.py:50`）——不要自己再升 fp32。
3. 两个 dot 之前**先降回 `v.dtype` / `k.dtype`**（§5.3 第 2 条）。
4. 返回 `w: [B,T,H,K]` `k.dtype`、`u: [B,T,H,V]` `v.dtype`。
5. `exp` 用普通 `exp`（参数 ≤0），**不要**在这里用 `safe_exp`。

最便宜 oracle：`gdn_reference.chunk_wu_reference(...)`（`gdn_reference.py:55-106`），
它同时返回 `(w, u, a_inv)`，**能把 `solve_tril` 也一起钉住**（`gdn_reference.py:94` 用
`torch.linalg.inv(I+A)` 独立求逆）。最小用例同上；额外断言
`w/o` 的 head 维是 `HV` 而不是 `HK`。

### 6.3 `chunk_gated_delta_rule_fwd_h`

必须复现：

1. `h[c]` = 第 `c` 个 chunk 的**进块状态**，布局 `[B, NT, H, V, K]`（**`V` 在 `K` 前面**）。
2. `v_new = u - w @ h[c]^T`，**存的是未加门（ungated）的值**：`safe_exp(g_last - g)` 只乘那份用于
   更新状态的副本（SG-delta_h:172-176 先把 `v_new` 存盘，SG-delta_h:185 才乘门），`g_last` 取
   `min((i_t+1)*BT, T)-1`。门由下游 `chunk_o` 自己以 `exp(g_i - g_j)` 施加（§2.4），这里再乘一次
   就是**双重衰减**，整条链会以 O(1) 失败。数值证据：未加门版本在 fp64 下与逐 token delta-rule
   参照的合链误差为 **3.5e-16**（`o`；`v_new` 2.1e-16，`tests/test_gdn_chunk_full.py`；另见
   `tests/test_gdn_delta_h_chunk_o.py` 的 4e-16），且从链自身的 `u`/`w`/`h` 重建
   `u - w @ h[c]^T` 时与落盘的 `v_new` **逐位相同（0.0）**；而改成加门版本后，`v_new` 与正确值的
   相对差为 **7.3e-1**、再喂给 `chunk_o` 合出的 `o` 相对差 **5.4e-1**，**每一个**接口测试都失败。
3. 状态递推 `S ← S·exp(g_last) + k^T @ v_new`，`h` 每次落盘截断到 `k.dtype`。
4. varlen 时 `h` 的全局块号用 `chunk_offsets[i_n] + i_t`；dense 时 `i_b*NT + i_t`。
5. `final_state: [N,H,V,K]`；**不要**原地写 `initial_state`，除非契约里明确要求
   （见 §5.4 与 §7 第 5 条）。

最便宜 oracle：**没有现成的**，但也不必写 Triton——用逐 token 的 delta-rule 递推在 fp64 里跑
（`S_t = exp(g_raw_t)·S_{t-1} + k_t (v_t - w_t S_{t-1})^T`），然后比对
“第 `c·BT` 个 token 之前的状态”。最小用例：`T=128`（2 个 chunk）、`BT=64`、
`HK=HV=1, K=8, V=8`，`initial_state=0` 与 `≠0` 各一遍（覆盖 `USE_INITIAL_STATE`）。
别忘了 `cu_seqlens=[0,37,128]` 这一档——它同时命中“尾块 `g_last`”和“`chunk_offsets` 偏移”。

### 6.4 `chunk_fwd_o`

必须复现：

1. `b_A` 的三角是 **`>=`（含对角）**，`b_o` 与 `b_A` 都乘 `exp(g)` / `safe_exp(g_i-g_j)`。
2. **`scale` 同时乘两项**（`SG-chunk_o:122`），且默认值取 `k.shape[-1] ** -0.5`。
3. `h` 按全局 chunk 号取（与 §6.3 第 4 条同一公式），并在 dot 前降到 `v.dtype`。
4. 输出 `[B,T,H,V]`、`v.dtype`、尾部为 0（`zeros_like`）。
5. `core_attn_out` 若非空则原地写（只在推理期）。

最便宜 oracle：**单 chunk 闭式**。取 `T ≤ 64`、`NT=1`、`initial_state=0`，则
`h[0]=0`，于是

```
o_i = scale · Σ_{j≤i} (q_i·k_j) · exp(g_i - g_j) · v_j
```

20 行 fp64 就能算，**能同时验证 `>=` 掩码、`safe_exp`、`scale` 双乘三件事**。
`T=64` 与 `T=1` 各跑一遍；再加一个 `T=65`（两个 chunk）验证 `h` 的跨块项。

---

## 7. 本轮无法从源头确定的事项（明确列出，不要猜）

1. **四个待移植文件的 vLLM 0.23 原文没拿到**（§0）。所有 kernel 体内数学来自 SG-* 姊妹版本，
   行号指向的是 `E:\project\python_test\.venv\Lib\site-packages\sglang\...` 而非 `runtime/vllm-api/`。
2. **`FLA_CHUNK_SIZE` 的字面定义**在 `vllm/.../fla/ops/utils.py`，未下载。值 64 已由
   `qwen_gdn_linear_attn.py:1085-1087` 与 SG 三处默认值交叉钉死，但“定义在哪一行”未知。
3. **`SUPPRESS_LEVEL` 默认值**（`chunk.py:19,83`）。只影响 `chunk_gated_delta_rule_fwd`
   的返回元组长度，不影响数学。
4. **vLLM 的 kkt 是否仍保留 `chunk_size` 形参**（`chunk.py:41-48` 没传）。SG 有（默认 64）。
5. **`output_final_state=False` 时 delta_h 的行为**：返回 `None` 还是仍写 `final_state`，
   以及它是否**原地改写 `initial_state`**。SG 版本是原地写（`SG-delta_h:96-100,254-271`
   `INPLACE_UPDATE=True`）；vLLM 的签名多出 `output_final_state` 且返回独立 `final_state`
   （`chunk.py:61,86`），**推断**vLLM 改为独立分配，但未见原文。这一条会决定
   `ssm_state[prefill_state_indices] = last_recurrent_state.to(...)`
   （`qwen_gdn_linear_attn.py:1532`）是否需要防御性拷贝。
6. **`core_attn_out` 的写入范围**：写满整个 buffer 还是只写 `o.numel()` 前缀。
   `chunk.py:129-133` 只给了两条断言（仅推理、dtype 相同）。可参考 flashinfer 分支的处理
   `co_flat[: o_flat.numel()].copy_(o_flat)`（`qwen_gdn_linear_attn.py:339-342`），
   但那是另一个后端，**不能直接当 vLLM FLA 的契约**。
7. **vLLM 的 `chunk_o` 是否仍按 `next_power_of_2(T)` 收缩 `BT`**。`qwen_gdn_linear_attn.py:1085-1087`
   说固定 64，但那是 warmup 注释，不是 `chunk_o` 的源码。
8. **vLLM 的 `delta_h` 是否还有 `gk`（逐维 gate）与 `initial_state_indices`**：`chunk.py` 都没传，
   **推断**已删除（`qwen_gdn_linear_attn.py:1513` 改为调用方 gather），但未见原文。
9. **`recompute_w_u_fwd` 里 `b_w` 的 dot 未指定 `allow_tf32=False`**（SG-wy_fast:107）
   是上游有意为之还是遗漏，无从判断；影响的是“能否 bit-comparable”。

> 复核方式（等有网时一条命令即可补齐）：
> `python .work/fetch_vllm_api.py` 风格的脚本，把
> `chunk_delta_h / chunk_o / chunk_scaled_dot_kkt / wy_fast` 四个 `.py` 从
> `raw.githubusercontent.com/vllm-project/vllm/v0.23.0/vllm/model_executor/layers/fla/ops/`
> 落到 `runtime/vllm-api/fla_<name>.py`，然后逐条替换本文第 2/3/5 节的
> `SG-*` 行号为 `fla_<name>.py` 的行号。**`chunk.py` 那一级的结论（§1、§4、§2 的签名与 dtype）
> 不依赖这些文件，已经可以直接开工。**
