# nano-vllm 投机解码（ngram 草稿）：设计说明

分支：`yczhou25/nano-vllm` 的 `spec-decode`（基于 `async-scheduling`）。用 `LLM(path, num_speculative_tokens=k)` 开启，默认关闭。暂时不能和 `async_scheduling` 同时开。

## 要解决的问题

decode 每步每条序列只出一个 token。batch 小的时候，GPU 每步的时间主要花在读权重和 KV cache 上，算力大部分空着。如果能先猜出后面几个 token，让模型一次前向把它们都验一遍，猜对几个就一步出几个 token，而且多算的几个位置几乎不花额外时间。

ngram（prompt lookup）是最便宜的猜法：不用草稿模型，直接在已有文本里找和结尾相同的片段，把它后面跟着的内容当草稿。代码修改、摘要、RAG 问答这类输出大段复述输入的任务，命中率很高。

## 做法

```
schedule():   每条 decode 序列：草稿 d1..dk = ngram(序列)，预留 k+1 个位置的块
run():        输入 [t, d1..dk]，位置 L-1..L+k-1，一次前向得到 k+1 行 logits
              拒绝采样：接受前 a 个草稿，再采一个 token，共 a+1 个
postprocess():追加 a+1 个 token（碰到 EOS / max_tokens 就截断），
              num_cached_tokens 回退到 L+a，只给 KV 有效的块做哈希
```

### 关键点

1. **草稿（`nanovllm/engine/ngram.py`）**：每条序列维护一个字典，从 n-gram（n = 4 到 1）映射到它最近一次出现后的位置，随序列增长增量更新。查找时先试长的 n-gram。字典里只放结尾之前的 n-gram，所以不会匹配到结尾自己。如果匹配位置离结尾不到 k 个 token，就按这个周期把重复的模式续下去（`a b a b` 会猜 `a b a b`）。完全找不到时，重复最后一个 token。
2. **验证**：每条序列的输入是 `[最后一个 token, d1..dk]`，所有序列都是 k+1 个 query，形状统一，所以 decode 仍然用 `flash_attn_with_kvcache`（q 是 `(B, k+1, H, D)`，causal 掩码按右下角对齐，`cache_seqlens = L+k`）。CUDA graph 还是按序列数抓图，缓冲区放大到 `max_bs*(k+1)` 行。lm_head 对 decode 的每一行都算 logits。
3. **拒绝采样（`RejectionSampler`）**：ngram 草稿是确定性的，相当于提议分布是 one-hot。按标准的投机采样规则：草稿 di 以概率 p_i(di) 接受；第一次拒绝时，从去掉 di 的 p_i 里采样（这就是 `max(0, p - q)` 归一化后的分布）；全部接受时，从最后一行再采一个 bonus token。这样输出的分布和一个一个采样完全一样。贪心行（temperature 0）只在草稿等于 argmax 时接受，输出和不开投机解码的贪心结果一致。实现上只对 k 行做 logsumexp 这类归约，最后只取每条序列一行 `(B, V)` 来采样，不额外存 `(B*(k+1), V)` 的 float 概率表。
4. **KV 和块的记账**：`commit` 先按 k+1 个 token 推进 `num_cached_tokens`；`postprocess` 知道接受了几个以后，再回退到 `L+a`。被拒绝的草稿留在 KV 里的值是脏的，但它们的位置都在下一步要写的范围里，下一步一定会先覆盖再读。块哈希（prefix cache）只覆盖 KV 有效的位置，所以脏数据不会进 prefix cache。块不用回收：下一步需要的块数一定不少于这一步。
5. **张量并行**：草稿随序列一起 pickle 给其他 rank，作为输入；只有 rank 0 做拒绝采样。

## 正确性验证（CPU）

`tests/engine_sim.py` 的随机模拟器加了两种模式：k=1 和 k=4。假模型现在对一半的请求输出“大致周期性”的文本（草稿常被接受），另一半是伪随机文本（草稿基本被拒）。decode 步的 logits 是 one-hot，交给真实的 `RejectionSampler` 验证，草稿取自输入 tensor，和 `ModelRunner.run` 一致。每个请求的输出必须和逐 token 生成的参考结果完全一致。另外新加了一项检查：每个登记到 prefix cache 的块，KV 必须和它的 token 对得上。

300+30 个用例里的覆盖情况：

| | k=1 | k=4 |
|---|---|---|
| decode 行数 | 20968 | 17391 |
| 接受 0 个草稿 | 72.1% | 81.9% |
| 接受 1 / 2 / 3 个 | 27.9% / - / - | 4.5% / 1.8% / 1.2% |
| 全部接受 | 27.9% | 10.6% |
| 被 max_tokens 截断 | 442 | 592 |
| 被草稿里的 EOS 截断 | 8 | 7 |
| 抢占次数 | 843 | 1259 |

拒绝采样器另有单测：贪心行的结果等于 argmax 链；随机采样时，10 万次采样里每个位置输出 token 的频率（以前面的 token 为条件）和目标 softmax 的差都在 0.02 以内。ngram 有 5 个单测。

我故意往代码里注入了 13 种 bug，全部被抓到（模拟器 330 个投机解码用例里失败的个数，以及单测）：

| 注入的 bug | 模拟器失败 | 单测 |
|---|---|---|
| 拒绝后不回退 `num_cached_tokens` | 165 | 通过 |
| 被拒草稿所在的块也做哈希 | 279 | 通过 |
| 草稿里碰到 EOS 不截断 | 241 | 通过 |
| `can_append` 不算草稿的位置 | 155 | 通过 |
| `may_append` 一步最多加一个块 | 97 | 通过 |
| 草稿的 slot 都算在最后一个块里 | 303 | 通过 |
| `context_lens` 不含草稿 | 316 | 通过 |
| 草稿没传给其他 TP rank | 316 | 5 个失败 |
| 拒绝之后还继续接受（sum 代替 cumprod） | 131 | 3 个失败 |
| 重采样时不去掉被拒的草稿 | 0 | 2 个失败 |
| 接受概率不考虑 temperature | 0 | 1 个失败 |
| bonus token 取自第一行 | 261 | 3 个失败 |
| ngram 匹配到结尾自己 | 316 | 5 个失败 |

后两种采样分布上的 bug 在 one-hot 的模拟器里看不出来，靠分布单测抓。

## 草稿模型（2026-10-23 加）

`LLM(path, num_speculative_tokens=k, speculative_model=draft_path)`：用同一家族的小模型（Qwen3-0.6B 给 Qwen3-8B）打草稿，代替 ngram。验证、拒绝采样、块记账全部复用上面的路径，草稿来源换成 GPU 上的小模型。提交 `962b67d`。

### 做法

```
prepare_decode():  目标模型的输入和之前一样：[t, 0, 0, 0, 0]（草稿位先留空），位置 L-1..L+k-1
                   另外准备草稿模型的输入：[t_{L-2}, t_{L-1}]、位置 L-2..L+k-2 的 slot
propose():         草稿模型跑 k 步：第 1 步输入最后两个 token（每条序列 2 个 query），之后每步输入
                   上一步的 argmax（1 个 query）。每步 argmax 直接写进目标模型输入的草稿位，不回 CPU
run_model():       目标模型一次前向，拒绝采样，和 ngram 版完全一样
```

1. **两套 KV cache，同一张块表**。草稿模型有自己的 `kv_cache`，块数和目标模型相同，块 id 共用：序列的 `block_table` 同时索引两套 cache，prefix cache 命中时两边一起命中。分配块数时按两个模型每块的字节数之和算。Qwen3-0.6B 和 8B 的每 token KV 分别是 115 KB 和 147 KB（都是 8 个 KV 头、head_dim 128，层数 28 对 36），所以草稿 cache 几乎让 KV 内存翻倍，这是草稿模型相对 ngram 的主要代价。
2. **prefill 两边同步**。prefill 时草稿模型用和目标模型完全相同的输入、slot 和块表再跑一遍（不算 logits），两套 cache 在同样的位置上有效。
3. **草稿是贪心的，所以采样器不用改**。草稿模型每步取 argmax，提议分布还是 one-hot，`RejectionSampler` 原样精确。vLLM v1 的 EAGLE/draft-model 路径现在也是这么做的（`draft_probs=None`）。代价是温度采样下接受率比"从 q 采样、按 min(1, p/q) 接受"低一些，好处是不用存 `(B, k, V)` 的草稿概率。
4. **草稿 cache 会落后一个 token**。草稿模型每步只给前 k-1 个草稿算了 KV（第 k 个草稿是它最后一步的输出，没当过输入）。k 个草稿全部接受时，新的 `num_cached_tokens` 指向的最后一个位置在草稿 cache 里就是空的。两处处理：
   - 下一步草稿模型的第一个输入是最后两个 token `[t_{L-2}, t_{L-1}]`，位置 L-2 的 KV 重算一次就补上了。没落后的时候，位置 L-2 的 slot 填 -1，`store_kvcache` 跳过，避免往已经哈希、可能被别的序列共享的块里写。
   - 这个空位所在的块晚一步再做哈希（`Scheduler.postprocess` 里 `hash_blocks(seq, start, num_tokens - seq.draft_cache_lags)`），否则别的序列 prefix cache 命中这个块时，草稿模型会读到一个没写过的位置。`BlockManager.hash_blocks` 加了一个回扫：从 `start // block_size` 往前找到第一个没哈希的块再开始，这样晚一步的块不会被漏掉。`seq.draft_cache_lags` 随序列 pickle 给其他 TP rank，decode 时 pickle 的是最后两个 token 而不是一个。
5. **CUDA graph**。草稿模型抓两套图：每条序列 2 个 query（第一步）和 1 个 query（之后），和目标模型的 k+1 query 图共用输入缓冲区和内存池，按批大小分别抓。`ModelRunner.forward(model, graphs, outputs, ...)` 把"查图、填缓冲、replay"抽成一个方法给两个模型用。
6. **TP**。每个 rank 都跑草稿模型（权重按 TP 切），只有 rank 0 有完整 logits，每步 argmax 后 `dist.broadcast` 给其他 rank（k 次很小的广播）。还没在多卡上跑过。

### 正确性验证（CPU）

模拟器加了 `draft1`、`draft4` 两种模式。假草稿模型是一个可调用对象，塞给真正的 `ModelRunner.propose` 当草稿模型用：它像注意力层一样读全局 context（slot、context_lens、块表），维护自己的假 KV cache（哈希种子和目标模型不同），每次调用检查输入 token、位置、context_len，按 slot 写 KV，再把 context 内每个位置读一遍核对，输出的"隐状态"就是它猜的下一个 token。它猜的 token 有 3/4 的概率等于目标模型的下一个 token。另外加了一条检查：每条序列所有 KV 有效的整块都必须已经哈希（除了草稿模型还没补上的那一块），防止哈希被漏掉。

660 个 draft 用例全过（2005 个用例总计）。k=4 时 9147 个 decode 步里 33% 全部接受，补位置 L-2 的情况出现 2207 次，哈希回扫 90 次；k=1 时分别是 75%、10066 次、604 次。

注入 13 种 bug：

| 注入的 bug | 60 个种子里失败的用例数（共 132） |
|---|---|
| 哈希不延后 | 58 |
| 从不补位置 L-2 | 88 |
| 总是重写位置 L-2 | 102 |
| 草稿步的 context_len 多 1 | 121 |
| 取每条序列的第一行而不是最后一行算 logits | 121 |
| 草稿步用错列（位置和 slot 错位） | 58 |
| `hash_blocks` 不回扫 | 44 |
| `draft_cache_lags` 不 pickle 给其他 rank | 88 |
| 抢占后不清 `draft_cache_lags` | 0 |
| 草稿位置不减 1 | 121 |
| `propose` 之后不恢复目标模型的 context | 121 |
| `draft_cache_lags` 条件取反 | 88 |
| prefill 时不跑草稿模型 | 125 |

唯一没抓到的那种只是多算一次：抢占后重新 prefill，标记还在，第一步多重写一次 L-2 的 KV（值一样），哈希也跟着晚一步，输出和 cache 内容都对，所以模拟器认为没问题。

### 预期

草稿模型一步的开销 ≈ k 次 0.6B 前向。A40 上 0.6B 一步约 3–4 ms（读 1.2 GB 权重），8B 一步约 25–30 ms。k=4 时每步多花 ~15 ms，平均每步多出 2–3 个 token 的话，每 token 时延能降到原来的一半左右；batch 大了以后目标模型本身变成算力受限，多算的 k 个位置不再免费，收益会缩小甚至变负。`cluster/nanovllm-draft-a40.sbatch` 在 Qwen3-8B 上跑 edit/chat × batch 1/8/32 × {无草稿, ngram k=4, 草稿模型 k=2/4}，再加 T=0.6 和 k=6 几组。

## 异步调度 + 草稿模型（2026-10-24 加）

`LLM(path, async_scheduling=True, num_speculative_tokens=k, speculative_model=draft_path)`：三者可以同时开。ngram 版和异步调度仍然互斥（ngram 要在 CPU 上查上一步采样出来的 token，而异步调度的意义就是不等它回来）。

### 难点

异步调度下，调度第 N+1 步时第 N 步还在 GPU 上，CPU 不知道它接受了几个草稿，也就不知道序列现在有多长：下一步的位置、slot、草稿模型的两个输入 token 全都取决于一个还没回到 CPU 的数。

### 做法

1. **在途的是"最多可能加的 token 数"，不是确定的数**。`Scheduler.commit` 给 decode 步记 `num_inflight += 1 + k`（prefill 最后一块记 1），`postprocess` 时再减掉。这个数只用于两处判断：块表够不够（`BlockManager` 按 `len(seq) + num_inflight + k` 算块数）、要不要停（`num_completion_tokens + num_inflight >= max_tokens` 就不再调度，等在途那步落地，避免多跑一步把 token 扔掉）。
2. **decode 步的 KV 在 postprocess 里记账，prefill 块在 commit 里记账**。原来的异步版在 commit 时就把 `num_cached_tokens` 往前推；投机解码接受几个草稿只有落地才知道，所以 decode 步改为在 `postprocess` 里按真实接受数推进（`start = num_cached_tokens`，`num_tokens = seq.num_tokens - start - 1`），prefill 块仍在 commit 时推进（下一块要紧接着排）。`hash_blocks` 用 postprocess 时的 start，commit 时记的 start 对 decode 已经过期。
3. **所有位置都在 GPU 上算**。`prepare_decode_with_draft_model` 把 CPU 知道的序列长度、最后两个 token、`draft_cache_lags` 传上 GPU，然后从上一步输出 `last_token_ids`（形状 `(B, k+1)`，接受的 token 后面补 -1）里数出在途序列实际加了几个 token（`(output >= 0).sum`），据此算出目标模型的 k+1 个位置和 slot、草稿模型的两个输入（最新的 token 和倒数第二个：在途 ≥2 个时都取自输出，恰好 1 个时倒数第二个是 CPU 的 last_token，0 个时和同步版一样）、以及草稿 cache 是否落后（在途序列看 `num_inflight == k+1`，不能用 CPU 上过期的标记）。`run_async` 保证每一步的输出都是 `(B, k+1)`：prefill 步的一维采样结果补 -1 到同样的形状，这样 decode 步索引上一步输出时不用分情况。
4. **`AsyncOutput.result` 做 -1 的剥离**，交给 `postprocess` 的和同步版 `run` 返回的形状一样（每条序列一个 list）。

### 正确性验证（CPU）

模拟器加 `async_draft1`、`async_draft4` 两种模式，假草稿模型和假目标模型照旧核对每次 KV 读写、输入 token、位置。在途检查：模拟器记录每一步的输出，decode 时核对 `num_inflight` 等于 1 或 k+1、在途行的最新 token 就是上一步那条序列的最后一个输出。"KV 有效的整块都已哈希"这条检查原来只在同步模式开，现在异步也开了：在途 prefill 块的 KV 已计入 `num_cached_tokens` 但落地才哈希，检查时减掉 `engine.pending` 里记的那一块。

全部 CPU 用例：2680 个用例（8 种模式 × 300 个随机用例 + 块大小 256 的 30 个 + 单元测试）全过，GPU 用例跳过。

注入 12 种只影响异步路径的 bug（40 个种子，`async_draft1` + `async_draft4` 共 88 个用例）：

| 注入的 bug | 40 个种子里失败的用例数（共 88） |
|---|---|
| commit 时就把 decode 步的 KV 记进 `num_cached_tokens`（结果不错，但 decode 产生的块全部漏掉哈希） | 71（异步模式开"整块必须哈希"检查之前是 0） |
| 停止条件 `>=` 写成 `==` | 66 |
| 没有停止条件 | 77 |
| 在途只记 1 个 token 而不是 1+k | 71 |
| postprocess 用 commit 时记的 start | 0（等价：`num_cached_tokens` 和哈希的块都一样，只是多算了一遍已哈希的块） |
| 已结束的序列仍把在途输出追加进去 | 4 |
| 在途恰好 1 个 token 时倒数第二个 token 取错 | 67 |
| 在途 ≥2 个时倒数第二个取成最新的 | 56 |
| 在途序列的 `draft_cache_lags` 用 CPU 上过期的标记 | 55 |
| 序列长度不加在途数 | 76 |
| 最新 token 不从上一步输出取 | 71 |
| `inflight_row` 记错行 | 26 |

### 预期

异步调度省掉的是每步 CPU 调度和 host→device 拷贝的时间（0.6B 上 10% 左右，8B 上目标模型一步 25–30 ms，占比更小）；草稿模型 k 步小前向本来就在 GPU 队列里，所以两者的收益大致相加。A40 作业 `cluster/nanovllm-draft-a40.sbatch` 里 `async_draft` 模式有 GPU 测试，吞吐对比待加。

## GPU 实测：A40 冒烟（2026-10-24）

Slurm 作业 223221，1 张 A40，Qwen3-0.6B，13.5 分钟。整套测试在 GPU 上跑：2680 个用例全过（模拟器这次用的是真 CUDA 张量），`tests/test_model_gpu.py` 五种配置（同步、异步、ngram、草稿模型、异步+草稿模型）的贪心输出和 transformers fp32 的 teacher-forcing 对比都在 0.5 个 logit 以内，默认容差不用调。

| 负载 | 配置 | 吞吐 | 接受率 |
|---|---|---|---|
| `bench.py` 随机 token，64 条，输入/输出各 ≤256 | 基线 | 6200 tok/s | |
| | 异步调度 | 6842 tok/s（+10.4%） | |
| | ngram k=4 | 1804 tok/s | ≈0 |
| `bench_spec.py --workload edit --batch-size 8` 贪心 | 基线 | 1196 tok/s | |
| | 异步调度 | 1300 tok/s（+8.7%） | |
| | ngram k=4 | 2376 tok/s（×1.99） | 0.796 |
| | 草稿模型（0.6B 给自己打草稿）k=4 | 1023 tok/s | 1.000 |
| | 草稿模型 + 异步调度 k=4 | 1066 tok/s | 1.000 |

怎么读：

- 异步调度在两种负载上都有 9–10% 的提升，和设计说明里"省掉的是每步 CPU 调度和 host→device 拷贝"的估计一致；0.6B 一步只有几毫秒，CPU 开销占比最大，所以这是上限附近的数字，8B 上会小一些。
- ngram 在代码编辑负载上接受率 79.6%，吞吐翻倍；在随机 token 上慢 3.4 倍，因为每条序列都算 k+1 个位置但一个都猜不中。这正是"要老实展示的最坏情况"。
- 草稿模型用 0.6B 给自己打草稿，接受率 1.000 说明草稿 cache、落后一步的哈希、拒绝采样这一整条路在 GPU 上是对的，但两个模型一样大自然不会更快（慢 15%，就是 k 次草稿前向的开销）。真正的加速看下一节 Qwen3-8B 的实测。

## GPU 实测：Qwen3-8B + Qwen3-0.6B 草稿（2026-10-25）

Slurm 作业 225010，1 张 A40，bf16，1 小时 39 分（约 300 SU）。目标模型 Qwen3-8B，草稿模型 Qwen3-0.6B。GPU 测试 18 个全过（含草稿模型和自适应两种模式）。吞吐算整个 run（prefill + decode），单位 tok/s；接受率 = 接受的草稿 / 提出的草稿；倍数相对同一行的基线。"自适应"是草稿 k=4 加 `speculative_disable_by_batch_size=8`。原始 58 行结果在仓库的 `docs/benchmarks/qwen3-8b-a40.txt`。

| 负载 | batch | 基线 | ngram k=4 | 草稿 k=2 | 草稿 k=4 | 草稿 k=4 + 异步 | 自适应 |
|---|---|---|---|---|---|---|---|
| edit 贪心 | 1 | 33.4 | 94.3（×2.8，接受率 0.50） | 72.9（×2.2，0.93） | 96.4（×2.9，0.88） | 97.9 | 96.2 |
| | 8 | 194.7 | 404.4（×2.1，0.52） | 280.5（×1.4，0.91） | 404.3（×2.1，0.86） | 409.1 | 391.2 |
| | 32 | 592.7 | 923.1（×1.56，0.51） | 741.6（×1.25，0.91） | 859.0（×1.45，0.85） | 836.8 | 450.0（×0.76） |
| chat 贪心 | 1 | 33.8 | 40.0（×1.18，0.05） | 56.1（×1.66，0.55） | 59.4（×1.76，0.41） | 60.6 | 60.1 |
| | 8 | 250.1 | 237.2（×0.95，0.05） | 328.4（×1.31，0.57） | 346.5（×1.39，0.41） | 351.5 | 329.4 |
| | 32 | 878.4 | 674.6（×0.77，0.05） | 1019.8（×1.16，0.55） | 965.3（×1.10，0.40） | 984.4 | 746.0（×0.85） |
| edit T=0.6 | 8 | 195.4 | 372.6（×1.91，0.50） | 328.1（×1.68，0.91） | 404.6（×2.07，0.87） | 404.5 | 393.5 |
| chat T=0.6 | 1 | 33.8 | 39.9（×1.18，0.05） | 55.9（×1.65，0.55） | 60.1（×1.78，0.42） | 60.7 | 60.6 |

另外 batch 1 贪心下草稿 k=6：edit 109.0 tok/s（×3.3，接受率 0.82），chat 56.5（接受率 0.31）。单独开异步调度的行都在基线 ±2% 以内（edit 33.4/196.7/583.1，chat 34.2/252.9/888.4）。

怎么读：

- 0.6B 给 8B 打草稿在所有负载、所有 batch 上都更快：batch 1 时代码编辑 ×2.9、聊天 ×1.8；batch 32 还有 ×1.45 和 ×1.16。T=0.6 和贪心几乎一样，拒绝采样没有额外代价。
- ngram 是要老实展示的最坏情况：代码编辑（输出复述输入）×2.8，聊天接受率只有 0.05，batch 8 和 32 反而变慢。
- 异步调度在 0.6B 上 +10%，在 8B 上测不出来：它省掉的是每步固定的几毫秒 CPU 工作，相对 0.6B 的一步很大，相对 8B 的一步很小。这组对比本身就说明了它什么时候有用。
- k 的选择看接受率：edit 接受率 0.85–0.93，k 越大越好（k=6 最快）；chat 接受率 0.4–0.55，batch 32 时 k=2（1019.8）比 k=4（965.3）快，因为没猜中的位置也要算。
- **自适应投机的问题**：阈值以下和草稿 k=4 一样（符合设计），但 batch 32 关掉草稿后比基线还慢：edit ×0.76，chat ×0.85。原因是为了让草稿 cache 不掉队，不带草稿的步仍然跑一次草稿模型（2 个 query），而这一步要读草稿模型的整个 KV cache：Qwen3-0.6B 每层的 KV 和 8B 一样大（8 个 KV 头 × 128），28 层对 36 层，所以这一步的代价远不止"参数量 7%"。edit 的 prompt 更长，慢得更多（每步多 17 ms 对 6.5 ms），也和"读 KV"的解释一致。修法见下一节的"自适应 k 的修正"。

## 自适应 k（2026-10-24 加）

`LLM(..., speculative_disable_by_batch_size=n)`：调度 decode 步时，如果正在运行的序列超过 n 条，这一步不带草稿（每条序列 1 个 query），否则照常带 k 个。vLLM 的 `disable_by_batch_size` 是同一个思路。batch 大了以后目标模型本身已经算力受限，多算 k 个位置不再免费，接受率不够高就会变慢（设计说明前面"预期"一节说的情况）。

### 做法

- **调度器按步决定**：`k_step = 0 或 k`，`num_scheduled_tokens = 1 + k_step`，块分配、在途 token 预留（`num_inflight`）、ngram 查找都按 `k_step` 来。两种步交替出现，但同一步里所有序列一样，`(B, q)` 的形状仍然统一。
- **model runner 按步读 q**：decode 时从 `seqs[0].num_scheduled_tokens` 读出这一步的 k；CUDA graph 对 q=1 和 q=1+k 各抓一套（只在设了这个选项时抓 q=1 那套）；采样器按 q 选普通采样或拒绝采样；异步调度下上一步的输出仍然补成 `(B, 1+k)`，所以下一步不用知道上一步带没带草稿。
- **草稿模型的 cache 不能掉队（第一版）**：不带草稿的步如果完全跳过草稿模型，它的 KV cache 就会缺这一步的 token，下次恢复投机时每条序列要补的数量不同（变长，CUDA graph 和 2-query 的第一步都不适用）。所以第一版让不带草稿的步仍然跑草稿模型的第一步（2 个 query：最后两个 token，写入最新 token 的 KV），只是不取它的输出。当时估的代价是"一次小模型前向，0.6B 对 8B 约 7%"；8B 实测（上一节）这一步要读草稿模型整个 KV cache，batch 32 时比不投机慢 15–24%，所以要改成下面的做法。

### 正确性验证（CPU）

模拟器加 `spec4_adaptive`、`draft4_adaptive`、`async_draft4_adaptive`：阈值随机取 1–3，这样带草稿和不带草稿的步大约各占一半（40 个种子里 1:1 左右，两种步在 batch 1–6 都出现过）。假模型按每步的 `num_scheduled_tokens` 核对输入形状、位置、slot，不带草稿的步核对草稿模型只跑了一步、且输入列没被写。

全部 CPU 用例（11 种模式 × 100 个随机种子 + 块大小 256 的用例 + 单元测试，1235 个）全过。

注入 6 种 bug（40 个种子，3 种自适应模式共 132 个用例）：

| 注入的 bug | 失败的用例数（共 132） |
|---|---|
| 不带草稿的步完全跳过草稿模型（cache 掉队） | 51 |
| 不带草稿的步仍把草稿模型的输出写进输入 | 51 |
| 关掉投机时 ngram 的草稿没清掉 | 25 |
| 草稿模型的输入张量按 k 而不是 max(k, 1) 分配 | 51 |
| 不带草稿的步之后不清 `draft_cache_lags` | 0（等价：多重写一次同样的 KV，哈希晚一步） |
| 块分配按配置的 k 而不是这一步的 k | 25 |

### 自适应 k 的修正（2026-10-25，进行中）

不带草稿的步完全跳过草稿模型，让它的 cache 掉队；每条序列记下草稿 cache 里有多少个 token（`draft_len`）。恢复投机的那一步，先让草稿模型补齐：每条序列缺的 token 数不同，所以这一步走 prefill 那套变长注意力（`flash_attn_varlen_func` + 分页 KV，和前缀缓存命中时的 prefill 一样），不用 CUDA graph；补齐前向最后一个位置的 logits 就是第一个草稿 token，之后的 k-1 步照常走 1-query 的 CUDA graph。切换只在 batch 大小跨过阈值时发生，所以变长前向很少跑；正常投机的步（缺 1 或 2 个 token）仍走原来的 2-query 图。

## 还没做的

- GPU 上 Qwen3-8B 的吞吐对比：已做（见上）。原文：注意力的 `(B, k+1)` 形状、CUDA graph、编译后的拒绝采样器都只能在 GPU 上验证。A40 作业脚本 `cluster/nanovllm-a40.sbatch` 会跑 GPU 测试（贪心结果和 transformers 对比，同步、异步、投机三种模式），以及吞吐对比：随机 prompt（ngram 基本猜不中，看最坏情况）、代码修改任务（让模型给源文件加 docstring 后原样输出）和开放问答，batch 1/8/32，k=0/2/4。草稿模型的 GPU 测试（0.6B 给自己打草稿，接受率应接近 1）在 `tests/test_model_gpu.py` 的 `draft` 模式里。
- ngram 版和异步调度合并：要把 ngram 查找搬到 GPU 上（vLLM 的做法是 CPU 上查、但等上一步落地，等于放弃了异步）。草稿模型版已经合并（见上）。
- k 自适应：已做并在 8B 上测过（见上），正在改成"草稿 cache 可以掉队、恢复时补齐"；更细的做法是按接受率动态调 k（chat 上 k=2 比 k=4 好，edit 上 k=6 最好）。
- 从草稿模型的分布采样草稿（而不是 argmax），按 `min(1, p/q)` 接受：温度采样下接受率更高，要多存 `(B, k, V)` 的 q。
- EAGLE-3 草稿头（可选）。
- ngram 索引在 CPU 上：长 prompt 第一次 decode 要建索引，每个 token 4 次字典写入。

## 面试时可以讲的取舍

- 为什么 one-hot 提议下拒绝采样是精确的：接受概率 `min(1, p/q)` 在 q 是 one-hot 时就是 `p(d)`；残差 `max(0, p-q)` 就是把 d 的概率置零。任何只依赖上下文的确定性猜法都满足这个条件。
- 为什么 batch 里所有序列都是 k+1 个 query：形状统一才能用 `flash_attn_with_kvcache` 和 CUDA graph。代价是没猜中的序列也算 k+1 个位置；vLLM 用 varlen 注意力支持每条序列草稿数不同。
- 为什么不回收多分配的块：下一步写的位置从 `L+a` 开始，到 `L+a+k` 结束，需要的块数不会少于这一步，回收了马上又要分配。
- 什么时候有用：decode 受内存带宽限制（小 batch、大模型）而且输出复述输入的时候。随机文本或大 batch 下会变慢，这是 benchmark 里要老实展示的。
