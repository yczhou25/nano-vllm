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
| | 32 | 592.7 | 923.1（×1.56，0.51） | 741.6（×1.25，0.91） | 859.0（×1.45，0.85） | 836.8 | 450.0（×0.76，修前） |
| chat 贪心 | 1 | 33.8 | 40.0（×1.18，0.05） | 56.1（×1.66，0.55） | 59.4（×1.76，0.41） | 60.6 | 60.1 |
| | 8 | 250.1 | 237.2（×0.95，0.05） | 328.4（×1.31，0.57） | 346.5（×1.39，0.41） | 351.5 | 329.4 |
| | 32 | 878.4 | 674.6（×0.77，0.05） | 1019.8（×1.16，0.55） | 965.3（×1.10，0.40） | 984.4 | 746.0（×0.85，修前） |
| edit T=0.6 | 8 | 195.4 | 372.6（×1.91，0.50） | 328.1（×1.68，0.91） | 404.6（×2.07，0.87） | 404.5 | 393.5 |
| chat T=0.6 | 1 | 33.8 | 39.9（×1.18，0.05） | 55.9（×1.65，0.55） | 60.1（×1.78，0.42） | 60.7 | 60.6 |

另外 batch 1 贪心下草稿 k=6：edit 109.0 tok/s（×3.3，接受率 0.82），chat 56.5（接受率 0.31）。单独开异步调度的行都在基线 ±2% 以内（edit 33.4/196.7/583.1，chat 34.2/252.9/888.4）。

怎么读：

- 0.6B 给 8B 打草稿在所有负载、所有 batch 上都更快：batch 1 时代码编辑 ×2.9、聊天 ×1.8；batch 32 还有 ×1.45 和 ×1.16。T=0.6 和贪心几乎一样，拒绝采样没有额外代价。
- ngram 是要老实展示的最坏情况：代码编辑（输出复述输入）×2.8，聊天接受率只有 0.05，batch 8 和 32 反而变慢。
- 异步调度在 0.6B 上 +10%，在 8B 上测不出来：它省掉的是每步固定的几毫秒 CPU 工作，相对 0.6B 的一步很大，相对 8B 的一步很小。这组对比本身就说明了它什么时候有用。
- k 的选择看接受率：edit 接受率 0.85–0.93，k 越大越好（k=6 最快）；chat 接受率 0.4–0.55，batch 32 时 k=2（1019.8）比 k=4（965.3）快，因为没猜中的位置也要算。
- **自适应投机的问题**：阈值以下和草稿 k=4 一样（符合设计），但 batch 32 关掉草稿后比基线还慢：edit ×0.76，chat ×0.85。原因是为了让草稿 cache 不掉队，不带草稿的步仍然跑一次草稿模型（2 个 query），而这一步要读草稿模型的整个 KV cache：Qwen3-0.6B 每层的 KV 和 8B 一样大（8 个 KV 头 × 128），28 层对 36 层，所以这一步的代价远不止"参数量 7%"。edit 的 prompt 更长，慢得更多（每步多 17 ms 对 6.5 ms），也和"读 KV"的解释一致。修法和 GPU 复测见下一节的"自适应 k 的修正"。

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

### 自适应 k 的修正：草稿 cache 可以掉队（2026-10-25）

不带草稿的步完全跳过草稿模型，让它的 cache 掉队；每条序列记下草稿 cache 覆盖到哪（`seq.num_draft_cached_tokens`，取代原来的布尔 `draft_cache_lags`）。恢复投机的那一步，草稿模型先补齐再打草稿：

- **补齐步的形状仍然统一**：每条序列缺的 token 数不同，但不走变长注意力，而是所有序列都算最后 m 个 token（m = batch 里最大的缺口，至少 2），用和目标模型 `(B, k+1)` 一样的 `flash_attn_with_kvcache` 右对齐因果掩码。已经在 cache 里的位置 slot 置 -1（算了但不写），比 m 短的序列前面补 0（位置夹到 0，输出丢掉）。最后一列的 logits 就是第一个草稿 token，之后 k-1 步照常走 1-query 的 CUDA graph。
- **m=2 时走原来的 2-query 图**：正常投机的步缺 1 或 2 个 token，和改之前完全一样；只有刚从"关草稿"切回来的那一步 m>2，走 eager 前向，这种切换只在 batch 大小跨过阈值时发生。
- **异步调度下在途步的处理**：在途的步如果带草稿，它自己已经补齐过，落地后缺口只会是 1 或 2（全接受时 2），所以 m 的上界按 2 算；如果在途的步不带草稿，缺口 = CPU 已知的缺口 + 在途 token 数。补齐输入在 GPU 上拼：CPU 已知的尾巴 + 在途输出，再按每条序列的在途 token 数 gather 出最后 m 个。TP 的其他 rank 通过 pickle 拿到的是 `token_ids[num_draft_cached:]`，所以尾巴里已缓存的位置统一填 0，各 rank 算出的输入一致。
- **哈希只到两个 cache 都有的地方**：`hash_blocks(seq, start, min(num_cached, num_draft_cached) - start)`；关草稿期间填满的块先不哈希，补齐那一步落地后 `hash_blocks` 的回扫把它们一起补上。前缀缓存命中的块两个 cache 都有，所以分配时 `num_draft_cached = num_cached`；prefill 两个模型一起算，commit 时同步推进。

模拟器相应改成：不带草稿的步不许准备草稿输入；补齐步核对每条序列的位置是 `max(0, n-m..n-1)`、未屏蔽的 slot 和 token 都对、最新 token 一定写入、补 0 的行不写；缺口有没有补全靠假草稿模型读 cache 时逐位置核对。

全部 CPU 用例（1235 个）全过。注入 12 种 bug（6 种草稿模型模式 × 40 个种子 + 块 256 用例，共 264 个用例）：

| 注入的 bug | 失败的用例数（共 264） |
|---|---|
| 在途带草稿的步落地后，全接受时少算一个缺口 | 67 |
| 补齐窗口 m 少算 1 | 20 |
| 已缓存位置的判断 `>=` 写成 `>`（少写一个位置） | 238 |
| 不屏蔽已缓存位置，全部重写 | 229 |
| 后续草稿步的 slot 错位一列 | 154 |
| 后续草稿步的位置错位一列 | 154 |
| 补齐输入里已缓存的位置填真实 token 而不是 0（TP 各 rank 输入不一致） | 3 |
| 不带草稿的步把草稿 cache 标成"已跟上" | 40 |
| 全接受后不记"少一个" | 92 |
| 哈希只看目标模型的 cache | 83 |
| prefill commit 不推进草稿计数 | 217 |
| 抢占时不清草稿计数 | 0（等价：重新 prefill 分配块时会重设） |

**GPU 复测（作业 225388，2026-10-25，1 张 A40，18 分钟）**：18 个 GPU 测试全过（含草稿模型和自适应模式）。Qwen3-8B + Qwen3-0.6B，batch 32，单位 tok/s，自适应 = 草稿 k=4 且 `disable_by=8`，括号里是对同一次运行基线的倍数：

| 工作负载 | 不投机 | 草稿 k=4 | 自适应（修前，作业 225010） | 自适应（修后） | 自适应 + 异步 |
|---|---|---|---|---|---|
| edit | 591.2 | 859.7（×1.45，接受率 0.85） | 450.0（×0.76） | 578.2（×0.98） | 568.8（×0.96） |
| chat | 877.4 | 963.2（×1.10，接受率 0.40） | 746.0（×0.85） | 874.5（×1.00） | 889.1（×1.01） |

batch 8 的 edit（阈值以下，草稿还开着）：草稿 k=4 399.4，自适应 390.0，和修前一样接近。原始行在 `docs/benchmarks/qwen3-8b-a40-adaptive-recheck.txt`。

读法：

- 关草稿的步不再跑草稿模型，修前 15–24% 的损失没了。edit 上还剩 2% 左右：每步的调度判断、切回投机时那一次 eager 补齐，或者就是测量波动（两次运行的基线差 0.3%）。
- 更要紧的是阈值本身：这台 A40 上 8B 在 batch 32 仍是访存受限，草稿 k=4 一直开着才是最快的（edit ×1.45，chat ×1.10），"超过 8 条就关"等于白丢加速。阈值应该定在 target 变成算力受限的地方（这里至少 64），更好的开关是实测接受率：chat 上 n-gram 只有 0.05，该关；模型草稿 0.40–0.85，该开。

## 按接受率调 k（2026-10-25 加开关，2026-10-26 改成按草稿位调 k）

`LLM(..., num_speculative_tokens=K, speculative_min_acceptance=0.2)`：每个草稿位单独统计接受率，每个 decode 步只提出前面接受率够阈值的那几位，可以是 0 位（不打草稿）也可以是全部 K 位。n-gram 和草稿模型都能用，可以和 `speculative_disable_by_batch_size` 同时开（batch 太大就直接 0）。起因是两件事：按 batch 大小关草稿的阈值和硬件、模型绑在一起（这张 A40 上 8B + 0.6B 用 8 太低），真正该关的是接受率低的情况（chat 上 n-gram 只有 0.05，batch 8 以上反而变慢）；而 8B 的实测里最优 k 随负载变（edit 上 k=6 最好，chat 上 batch 32 时 k=2 比 k=4 好），一个开关不够，要调 k。第一版（14df55e）只是开关：全 batch 的接受率低于阈值就 k=0；这一版把它推广成按位调 k，开关是它的特例。

### 做法

- **统计**：第 j 位被接受的前提是前 j-1 位都被接受，所以第 j 位的接受率天然随 j 递减，"砍到哪一位"就是看它从哪一位开始低于阈值。`Scheduler.postprocess` 对每条投机序列、它这一步提出的每个位 j 记一个样本（提出了；是否接受：`j < len(new_token_ids)`），每位维护最近约 `ACCEPTANCE_WINDOW`（64）个序列-步的平均，但窗口至少覆盖 `ACCEPTANCE_MIN_STEPS`（8）步（batch 32 时 64 个样本两步就换完，噪声 ±0.05，k 会在阈值附近来回跳；2026-10-26 加）：没看满窗口前是普通平均，之后每步的样本按份额替换（`acc[j] = (acc[j]·seen + accepted[j]) / (seen + proposed[j])`，`seen` 封顶在窗口大小）。只提出了 k<K 位的步不改后面几位的统计。`reset_stats` 另外累计每个位提出/接受的总数和每步 k 的分布，`bench_spec.py` 打印成 `acc_pos=` 和 `k_steps=`。
- **决定**：`schedule` 取 k = 最长的前缀使 `acc[1..k]` 都不低于阈值（中间断一位就停，后面的位不单独算）。n-gram 提案截到前 k 个，草稿模型只跑 k 步。每 `PROBE_INTERVAL`（16）个 k<K 的 decode 步插一个 K 位的探测步，重新测量被砍掉的位，探测步里后面几位接受率高就回来了。没测过之前信任（初始 1.0）。batch 大小规则说 0 就 0，不再看接受率；探测计数照常累加，batch 变小后的第一步就是探测步。空调度步不推进探测计数。
- **模型侧**：以前 decode 步只有 q=1 和 q=K+1 两种形状，现在 1..K+1 都可能：开了 `speculative_min_acceptance` 时 target 给每个 q 各捕一套 CUDA graph，草稿模型的追赶/生成 graph 不变（q∈{2,1}）。异步调度下在途步带的草稿数不再等于 K：`prepare_decode_with_draft_model` 用 `seq.num_inflight > 1` 判断在途步带草稿、用"产出 token 数 == seq.num_inflight"判断全接受（原来写死 `1+K`），`postprocess` 里"全接受导致草稿 cache 少一个 token"的判断也改成按这一步实际的草稿数。
- **采样器编译**：`Sampler` 和 `RejectionSampler` 用 `torch.compile`，原来按具体形状编译，每个新的 (batch, k) 形状都在计时区间里重新编译几秒；自适应 k 让一次跑里出现 q=1..K+1 所有形状（加上尾批的各种 batch），编译开销能吃掉百分之十几的吞吐（228429 作业里 chat 上 n-gram 开了开关反而比不开慢 5%，怀疑就是这个）。改成 `dynamic=True` 只编译一次，并在启动时对每个 q 预热（`ModelRunner.warmup_samplers`），和 CUDA graph 的 q 集合共用 `decode_query_lengths`。
- 顺带修了 14df55e 的一个 GPU bug：只开 `speculative_min_acceptance`、不开 `speculative_disable_by_batch_size` 时没有捕 q=1 的 graph，第一个关草稿的步会 KeyError。CPU 测试不走 CUDA graph，没抓到；现在 graph 的 q 集合按两个开关算。
- **代价**：k<K 期间每 16 步一次探测。n-gram 在 chat 上的探测步每条序列多算 K 个位置（batch 32 时那一步慢约 30%），平摊到 16 步约 2%；草稿模型的探测步还要先补齐草稿 cache（最多 17 个 token 的 eager 前向）。

### 正确性验证（CPU）

- `tests/test_acceptance_switch.py` 直接驱动调度器（12 个用例）：没测过先信任；从第一个被拒的位往后砍；中间断一位就停；每隔 PROBE_INTERVAL 用全 K 位探测一次；短步计入探测计数、不改后面位的统计；探测成功后位回来；窗口会忘掉旧历史；按位、按整个 batch 统计；和 batch 大小规则叠加；阈值 0 永不砍；窗口至少 8 步；统计按步和按位计数。
- 模拟器的 `spec4_acc`、`draft4_acc`、`async_draft4_acc` 三种模式：阈值在 {0.2, 0.5, 0.8} 里随机，窗口 2–40、探测间隔 1–6 随机；假草稿模型按位分质量：一半请求全不命中，另一半第 1 位 7/8 命中、第 2 位 3/4、之后 1/2，所以 batch 里各位的接受率随成员和位置变，k 在 0..4 每个值都停过（60 个种子里每种模式约 1100 个 decode 步，草稿模型模式 k=0 约 540、k=1 约 90、k=2 约 90、k=3 约 33、k=4 约 340）。策略检查改成：模拟器自己从调度器的接受率数组算出这一步应有的 k（独立实现：最长不低于阈值的前缀；探测计数到了就是 K），每个 decode 步必须相等；在途 token 数允许 1..1+K。追赶步的检查改成精确的：m 必须正好是 2 或草稿 cache 缺得最多的那条序列缺的数，slot 必须恰好在已缓存的位置屏蔽（在途带草稿的步算它补过之后的 cache），多写一个 token 也算错（上一版只查"缺的都写了"，抓不住多算的）。KV 读写、哈希块、输出和参考完全一致的检查照旧。
- 全部 CPU 用例（14 种模式 × 100 个种子 + 单元测试，共 1577 个）全过。

注入 21 种 bug，跑 12 个单元测试 + 132 个模拟器用例（3 种 `_acc` 模式 × 40 个种子 + 块 256 用例）：

| 注入的 bug | 失败的用例数（共 144） |
|---|---|
| 从不探测（砍掉的位永远不回来） | 101 |
| 探测计数器从不清零 | 108 |
| 只提出了部分位的步也算探测（计数器清零） | 52 |
| 探测间隔差一（`<=`） | 101 |
| 初始接受率 0（没测过就不信任草稿） | 11 |
| 阈值比较用 `>` 而不是 `>=` | 42 |
| 中间断一位不停（数所有够阈值的位） | 1 |
| 窗口不保证至少 8 步（大 batch 两步就换完） | 1 |
| 每步 k 的统计只数探测步 | 1 |
| 窗口不封顶（变成累计平均，忘不掉历史） | 2 |
| 窗口只看当前步（没有历史） | 7 |
| 接受数把采样出的那个 token 也算上 | 9 |
| 没提出的位也算成被拒 | 4 |
| 空调度步也推进探测计数器 | 23（只有异步模式抓到：探测被空步吃掉） |
| batch 大小规则已经说 0 时接受率规则再按位给 k | 1 |
| n-gram 提案不截到 k 个 | 15 |
| 全接受时草稿 cache 从不掉队（`lag` 恒为假） | 17 |
| 只有 K 位全接受才算掉队（k<K 的步全接受算漏） | 15 |
| 追赶长度 m 只对在途 K 位的步取 2（k<K 的在途步按整段算） | 14（精确的追赶检查才抓到） |
| 在途步只有带满 K 位才算"自己补过 cache"（k<K 的在途步按旧 cache 屏蔽 slot） | 13（精确的追赶检查才抓到） |
| "全接受"按 K+1 个 token 判断（k<K 的在途步） | 10 |

### GPU 实测（作业 228429，2026-10-26，1 张 A40，53 分钟，代码 8c4dde3）

预期是四件事：chat 上 n-gram k=4 加 `min_acceptance=0.2` 应回到不投机的水平（batch 8/32 修前是 ×0.95 / ×0.77）；edit 上 n-gram 和 chat 上草稿模型的加速必须保住；混合 batch 看它怎么选；草稿模型 K=6 加 `min_acceptance=0.3` 时 edit 应接近固定 k=6 的 109 tok/s（batch 1），chat 应自动落到 k=2–3。实测（24 个 GPU 测试全过；原始行在 `docs/benchmarks/qwen3-8b-a40-adaptive-k.txt`；基线列来自作业 225010，同一配置两次运行能差 10%：chat batch 32 的 n-gram k=4 那次 674.6、这次 763.8）：

| 负载 | batch | 基线（225010） | 草稿 k=6 固定 | 草稿 K=6 + min_acc 0.3 | 草稿 K=4 + min_acc 0.2 |
|---|---|---|---|---|---|
| edit | 1 | 33.4 | 109.0（×3.3，接受率 0.82） | 123.9（×3.7，0.82） | |
| edit | 32 | 592.7 | 818.7（×1.38，0.79） | 969.5（×1.64，0.79） | |
| chat | 1 | 33.8 | 56.5（×1.67，0.31） | 60.1（×1.78，0.52） | |
| chat | 32 | 878.4 | 868.6（×0.99，0.31） | 1104.8（×1.26，0.52） | 1108.2（×1.26，0.48） |

- **草稿模型上按位调 k 是干净的赢面**：每一格都比固定的 k=4 和 k=6 快。chat batch 32 上比固定 k=6 快 27%、比固定 k=4（965.3）快 15%：后面几位平均只有 31% 被接受，砍掉之后每步少算几个位置，探测步每 16 步一次。edit batch 1 从 109.0 到 123.9，是这台 A40 上 8B 单请求的最好数字（×3.7）。接受率从 0.31 升到 0.52 正是"砍掉了不该提的位"的样子。
- **n-gram + 开关保住了 edit 的加速**（batch 32 1002.8、batch 1 111.5，都高于 225010 的 923.1 / 94.3），**但 chat 和混合负载没回到基线**：chat batch 32 729.0（不开开关 763.8，基线 878.4）、batch 8 267.3（基线 250.1，这个回到了）；混合 batch 32：不投机 842.9、n-gram 719.1、n-gram + 开关 727.2、草稿模型 925.6、草稿模型 + 开关 871.6；混合 batch 8：251.1 / 247.9 / 222.2。
- 事后在代码里找到两个原因，已在 44129e2 修掉：（1）`Sampler` / `RejectionSampler` 的 `torch.compile` 按具体形状编译，自适应 k 让计时区间里出现 q=1..K+1 所有形状，每个新形状重编译几秒（chat batch 32 一行总共才 42 秒）；改成一次动态编译并在启动时预热。（2）窗口 64 个样本在 batch 32 时两步就换完，位 1 的接受率估计噪声 ±0.05，正好在 0.2 阈值附近来回跳，k 在 0 和 1 之间抖；改成窗口至少 8 步。还有一个是设计上的限制：一步只有一个 k，整个 batch 共用，混合负载里 edit 一半接受率高、chat 一半低，平均下来 k 砍不掉，chat 那一半白算——要解决得每条序列一个 k（ragged 的 query，vLLM 的做法），这是下一步。
- 这一节的基线列来自另一次作业，下一节的复测证明这样比是不对的：复测那台 A40 的每个基线都比 225010 高 13–17%，这里的 ×3.7、×1.64、×1.26 都偏大，以下一节同场的数字为准。

### GPU 复测（作业 228654，2026-10-27，1 张 A40，64 分钟，代码 1f8127f）

脚本 `cluster/nanovllm-adaptive-k2-a40.sbatch`：29 行，每种负载/batch 先跑一行不投机的基线，所有对比都在同一次作业里；每行打印各草稿位的接受率 `acc_pos=` 和每步 k 的分布 `k_steps=`。24 个 GPU 测试全过；原始行在 `docs/benchmarks/qwen3-8b-a40-adaptive-k2.txt`。少了一行（chat batch 32 n-gram 加 `disable_by_batch_size=0`）：Config 要求这个值 ≥ 1，而脚本把 bench 的 stderr 丢了，所以日志里没有报错——下个脚本改成 `=1` 并保留 stderr。单位 tok/s，括号里是对本次基线的倍数和接受率。

| 负载 | batch | 基线 | 草稿 k=4 固定 | 草稿 K=4 + min_acc 0.2 | 草稿 k=6 固定 | 草稿 K=6 + min_acc 0.3 |
|---|---|---|---|---|---|---|
| edit | 1 | 38.4 | | | 124.3（×3.24，0.82） | 123.7（×3.22，0.82） |
| edit | 32 | 683.6 | 978.2（×1.43，0.85） | 970.8（×1.42，0.86） | 975.4（×1.43，0.79） | 963.9（×1.41，0.79） |
| chat | 1 | 39.0 | | | 63.6（×1.63，0.31） | 63.4（×1.63，0.52） |
| chat | 32 | 1025.3 | 1065.3（×1.04，0.40） | 1222.2（×1.19，0.48） | 854.1（×0.83，0.31） | 1103.4（×1.08，0.50） |
| 混合 | 32 | 847.1 | 798.9（×0.94，0.57） | 909.1（×1.07，0.57） | | |

| 负载 | batch | 基线 | n-gram k=4 固定 | n-gram k=1 固定 | n-gram K=4 + min_acc 0.2 | 开关下每步 k 的分布（k=0..4） |
|---|---|---|---|---|---|---|
| edit | 32 | 683.6 | | | 1051.5（×1.54，0.51） | 2, 0, 0, 0, 405 |
| chat | 1 | 39.0 | | | 39.5（×1.01，0.13） | 3268, 237, 0, 0, 219 |
| chat | 8 | 289.0 | | | 271.9（×0.94，0.10） | 936, 15, 0, 0, 60 |
| chat | 32 | 1025.3 | 754.3（×0.74，0.05） | 922.1（×0.90，0.14） | 902.8（×0.88，0.10） | 928, 23, 0, 0, 60 |
| 混合 | 32 | 847.1 | 715.6（×0.84，0.17） | | 746.8（×0.88，0.27） | 523, 144, 79, 11, 241 |

- **草稿模型上，开关在有位被拒的地方赚、其他地方不亏。** chat batch 32：K=4 从固定的 ×1.04 到 ×1.19（453 步里 380 步停在 k=3），K=6 从亏（×0.83）到 ×1.08（540 步里 331 步 k=2、141 步 k=3）；混合 batch 32 从 ×0.94 到 ×1.07。edit 上各位接受率 0.72–0.92 都过阈值，开关一直给 k=K，和固定 k 差 1% 以内。chat batch 1 上开关把 K=6 砍到多数步 k=2，吞吐纹丝不动（63.4 对 63.6）：batch 1 一步读一遍权重，q 多几个不要钱，砍掉的位本来就是免费的，和设计时的预期一致。
- **上一节 edit batch 1 "109.0 → 123.9（×3.7）" 是错的**：109.0 是 225010 里的固定 k=6，123.9 是 228429 里的自适应，两次作业的基线差 15%。同场测：固定 124.3、自适应 123.7，都是 ×3.2，开关在这格不赚也不亏（本来也不该赚：每一位都过阈值）。
- **n-gram + 开关：edit 保住 ×1.54，chat batch 1 回到基线（×1.01），batch 8（×0.94）和 32（×0.88）还差一截，混合 ×0.88。** 44129e2 的两处修复是有效的：228429 里开开关比不开还慢（729 对 764），这次是 903 对 754。chat batch 32 这一行 1011 步里 928 步 k=0、60 步探测（k=4）：探测步约是普通步的 1.5 倍，只能解释 12 个点里的 3 个；k=0 的步走的是和基线一模一样的代码路径（`prepare_decode`、`Sampler`、q=1 的 CUDA graph），剩下的 9 个点现在解释不了。下一个作业单独量这条路径（n-gram 加 `disable_by_batch_size=1`：每步都不出草稿、也不探测），并在一次作业的末尾再跑一遍基线，看同一作业内前后的漂移有多大。
- **混合负载上开关在抖**：batch 平均的各位接受率 0.31/0.27/0.25/0.24 刚好在 0.2 阈值上方，于是 523 步 k=0、241 步 k=4 来回换，而 k=4 的步里 chat 那一半的草稿照样全算全拒。这就是"一步一个 k"的设计限制，每条序列一个 k（ragged 草稿）是下一阶段。

### 剩下的 12 个点在哪（作业 230846，2026-10-28，1 张 A40，17 分钟，代码 6f10641）

脚本 `cluster/nanovllm-adaptive-k3-a40.sbatch`，只跑 chat，11 行，原始行在 `docs/benchmarks/qwen3-8b-a40-adaptive-k3.txt`；每行之前打印 GPU 温度/频率/功耗。

| 行（chat，batch 32） | tok/s | 对基线 |
|---|---|---|
| 不投机（第 1、5、7 行） | 878.9 / 876.4 / 879.2 | — |
| n-gram k=4 + `disable_by_batch_size=1`（每步都不出草稿、不探测，但 proposer 在） | 878.4 | ×1.00 |
| n-gram K=4 + min_acc 0.2，每 16 步探测（60 次探测 + 23 步 k=1） | 827.4 | ×0.94 |
| n-gram K=4 + min_acc 0.2，每 64 步探测（15 + 12） | 833.3 | ×0.95 |
| n-gram k=4 固定 | 679.0 | ×0.77 |

batch 8：基线 250.2 / 250.2，关草稿 250.1，开关 235.8（×0.94）。

- **同一作业内没有漂移**：三次基线差 0.3%，频率一直 1740 MHz、41–44 ℃、76–81 W。两次作业之间却差 15%（这台 878.9，228654 那台 1025.3，代码和参数一样），所以只能同作业内比。
- **关草稿的路径本身不要钱**：878.4 对 878.9。228654 里怀疑的"k=0 的步比基线慢"不成立。
- **开关的损失基本不随探测次数变**：探测减到 1/4，损失只从 5.9% 变到 5.2%。按 k=4 固定那行算，一步 k=4 是 48.3 ms、普通步 34.6 ms，60 次探测只多 0.8 s，而开关那行总共多了 2.4 s；探测 64 那行多 2.0 s，探测只解释 0.2 s。两行都剩约 1.5–1.8 s 的固定开销，和出了多少草稿无关。
- **固定开销是 `torch.compile` 的 0/1 特化**：dynamic=True 把每个维度当符号，唯独大小 0 和 1 会特化成常数。启动时的预热只用 batch 8 编译，bench 自己的预热（1 个 prompt、16 个 token）又把 batch 1 的 q=1 和 q=K+1 编了，但 batch 1、q=2（一条序列、一个草稿）谁都没编——每个 batch 的尾巴都会收缩到一条序列，碰上一步 k=1 就在计时区间里重编译一次（GPU 上约 1.5 s）。CPU 上开着 dynamo 的重编译日志复现了这条序列：全程 6 张图，其中 1 张在尾巴那步才编。第一版修法是 `warmup_samplers` 把 batch 1 也预热一遍（c4cf5ac），CPU 上同一条序列计时区间里零编译。
- **GPU 上复测（作业 231468，开着 `TORCH_LOGS=recompiles`）却没好**：开关行 961.9 对基线 1038.6（×0.93）。日志里 186 条 Recompiling 说明了原因：每一行（包括基线）在第一个真实请求时把所有编译过的层（两个 layernorm、rotary、activation）和两个采样器**全部重编译一遍**，守卫是 `utils_device.CURRENT_DEVICE == device('cuda')`——`ModelRunner` 加载模型和预热时把 torch 的默认设备设成 cuda，服务时又改回 cpu，而 `torch.compile` 会对默认设备加守卫。所以启动时的预热全部作废；bench 自己的预热调用把这一串重编译挡在了计时器外面，但那一次是 batch 1 的 prompt，于是采样器被 0/1 特化成 batch 1，计时区间里再重编译两次（`temperatures` 从 1 到 32；`2 <= draft_token_ids.size()[1]`，k 从 ≥2 到 1）。对真实用户来说这是第一个请求的延迟：十几次编译、几十秒。
- **第二版修法**：预热和 CUDA graph 捕获都挪到默认设备恢复成 cpu 之后，预热用的张量显式写 `device="cuda"` 和模型的 dtype（ac3f3be）。
- **复测（作业 231709，`docs/benchmarks/qwen3-8b-a40-adaptive-k5.txt`）**：默认设备那条守卫没了（日志从 186 条降到 134 条，层不再重编译），但开关行还是 ×0.94（829.7 对基线 880.2 / 880.8，关草稿 881.0）。剩下的守卫是 `requires_grad mismatch`：`ModelRunner.run` / `run_async` 没有放在 `torch.inference_mode` 下（只有模型的 forward 在），所以真实步的 lm_head logits 带着 autograd、预热的不带，两个采样器在第一个真实步又编了一遍；之后照旧被 bench 的 batch 1 预热 0/1 特化，再加一条 stride 守卫（真实草稿是 `input_ids.view(B, q)[:, 1:]`，stride 是 (q, 1)，预热用的是连续张量），拒绝采样器在计时区间里又编了两次。草稿模型那两行照样是 ×1.08（固定 k=4，953.4）和 ×1.20（K=4 + 开关，1052.4）。
- **第三版修法**：`run` / `run_async` 加 `@torch.inference_mode()`，采样器预热也在 inference mode 下跑，草稿用 `(batch, q)` 张量的列切片。CPU 上开着重编译日志：预热编 6 张图，之后 bench 式的服务序列零编译。
- **复测（作业 232099，2026-10-28，代码 0e2b732，`docs/benchmarks/qwen3-8b-a40-adaptive-k6.txt`）**：开关行 871.8 对基线 881.2 / 881.6（×0.99，231709 里是 ×0.94），batch 8 上 249.9 对 251.6（×0.99，原来 ×0.94），关草稿 881.4 照旧免费；草稿模型固定 k=4 1000.3（×1.14）、K=4 + 开关 1060.7（×1.20）。重编译日志 117 条，没有一条 `requires_grad`，采样器的编译全部落在每行启动后 13 秒内（启动预热里 batch 1 和草稿宽 1 的 0/1 特化，以及它们之后的符号图），计时区间里一次都没有——231709 里开关行是在 +74 s 和 +87 s 重编译的。到这里 228654 那 12 个点算清了：7 个点是两台 A40 的差别，4 个点是启动编译漏进计时区间，1 个点是探测。

## 每条序列一个 k（ragged 草稿，2026-10-29）

`LLM(..., num_speculative_tokens=K, speculative_min_acceptance=0.2, speculative_per_sequence=True)`：接受率按序列统计，每条序列自己决定这一步提几位草稿，一个 batch 里各序列的草稿数可以不同。起因是上一节的混合负载：batch 平均的接受率 0.31/0.27/0.25/0.24 刚好在阈值上方，开关在 k=0 和 k=4 之间抖，k=4 的步里 chat 那一半照样全算全拒。每条序列一个 k 是 vLLM 的做法（varlen 注意力）；这里的原则还是不开开关时一切不变。

### 做法

- **统计和决定（调度器）**：每条序列有自己的 `acceptance[1..K]`。窗口 `ACCEPTANCE_WINDOW_PER_SEQUENCE` = 16 步（一条序列每步每位只有一个样本，满窗口后每步按 1/17 替换；窗口小是故意的，一条序列从“抄 prompt”切到“自由生成”要在几步内被发现）。每个 decode 步对每条序列算 `num_acceptable_drafts(seq)`：最长的够阈值的前缀，没测过先信任（新序列第一步提满 K），但信任背后垫着 `ACCEPTANCE_PRIOR_PER_SEQUENCE` = 1 个“接受”的先验样本：一条序列每步每位只有一个样本，没有先验时开头一次没中这一位就是 0，要等下一次探测（16 步）才回来；有先验是 1/2，再没中一次才砍。这是作业 233732 的混合 batch GPU 用例抓到的：抄写任务第一步的 n-gram 草稿必然不中（prompt 末尾的聊天模板在前文没出现过），四个问题在它的下一次探测之前就答完了，整场没有一个 ragged 步。探测还是整个 batch 的：`PROBE_INTERVAL` 个“有序列没提满 K”的步之后，所有序列同一步都提 K，这是一个统一形状的步，走 CUDA graph。第一版（5b62af5）让每条序列各自计数、各自探测，A40 作业 233573 里 chat 上 21% 的步因此变成 ragged 步（各序列的探测落在不同的步，每个都是 eager 步），n-gram 开关从 ×0.99 掉到 ×0.95；改成 batch 同步之后探测步和 batch 级开关一样。`num_scheduled_tokens = 1 + num_drafts_i`，块分配按各自的数，n-gram 提案截到各自的数。batch 大小规则优先（超过就全部 0）。batch 平均的接受率照旧统计（`acc_pos=` 还是它），`num_steps_by_k` 在这个模式里数的是序列-步；另加 `num_decode_steps` / `num_ragged_steps`（各序列草稿数不同的步）/ `num_capped_steps`（被 batch 大小规则清零的步），bench 打印成 `ragged=` 和 `capped=` 两个比例。
- **输入布局（`ModelRunner.ragged_layout`）**：decode 步的输入按序列拼成一维，序列 i 占 `1 + num_drafts_i` 行（最后一个 token，然后它的草稿）。布局函数返回三个张量：`token_index (B, 1+max k)`，每条序列各行的行号，不够的重复它最后一行；`write_index (B, max k)`，第 i 个草稿写到哪一行，没有这个草稿的序列写到末尾多出的一行 scratch 行；`num_drafts (B,)`。所有序列草稿数相同的步（含全 0）就是原来的 `(B, k+1)` 网格，走原来的路：`flash_attn_with_kvcache`、CUDA graph、按 view 切草稿的拒绝采样。只有草稿数不同的步（ragged 步）走下面的新路。
- **注意力**：ragged 步设置 `cu_seqlens_q`，注意力层走 chunked prefill 那条路：`flash_attn_varlen_func` 带 `block_table` 读整个 KV cache，causal 掩码右下角对齐，每条序列的 q 长度不同也没关系。`cu_seqlens_q` 在 CPU 算；`cu_seqlens_k` 在 GPU 上从 `context_lens` 累加，因为异步调度下在途 token 数只有 GPU 知道（CPU 给一个上界当 `max_seqlen_k`）。ragged 步不走 CUDA graph（eager），这是它的代价，要在 GPU 上量。
- **拒绝采样器**：多一个分支，输入 `token_index` 和 `num_drafts`：先按 `token_index` gather 出 `(B, k+1, V)` 的 logits（补齐的位重复最后一行），正常算接受，再把补齐的位强制算拒绝（`accepted &= arange(k) < num_drafts`），“全接受要从最后一行采 bonus”按各自的 `num_drafts` 判断。每条序列的输出和它单独放在一个 k=num_drafts_i 的统一 batch 里完全一样（有测试逐条比）。
- **草稿模型**：追赶步只补有草稿的序列，0 草稿的序列这一步整个跳过草稿模型（slot 全屏蔽），cache 掉队，和它处在 k=0 的步一样，`num_draft_cached_tokens` 的记账不用改。生成第 i 个草稿时，草稿数不到 i 的序列算一个空行：输入是它自己的最后一行（位置夹住，不越界读块表），slot 置 -1 不写 KV，context_len 是它自己的；结果按 `write_index` 写到 scratch 行。target 的前向只取前 n 行（不含 scratch 行）。n-gram 路径更简单：提案已经各自截断，只要布局和 context。
- **异步调度**：在途输出 `last_token_ids` 照旧 `(B, K+1)` 填 -1；ragged 步的输出宽度是这一步最大的 k+1，不够 K+1 的补到 K+1。在途步的草稿数由 `seq.num_inflight` 给出，每条序列不同也没问题。

### 正确性验证（CPU）

- `tests/test_per_sequence_k.py` 直接驱动调度器（9 个用例）：每条序列自己的 k；batch 的统计不漏到序列上；新序列从满 K 位开始；探测是 batch 的一个统一步（有序列没提满 K 的步计数，到了所有序列同一步提 K）；短步计入探测、不改后面的位；窗口会忘掉旧历史（满窗口后每步 1/5，窗口 4）；batch 大小规则对每条序列都生效；统计按序列-步和按 ragged 步计数；n-gram 提案按序列截断。
- `tests/test_ragged_sampler.py`（5 个用例）：布局函数本身；贪心下 ragged batch 的每一行和“把这条序列单独放进它自己 k 的统一 batch”逐条相等；全部相同的 batch 两条路结果一样；温度采样下（T=0.7 和 1.0，5 万对序列）带草稿和不带草稿的序列各自的输出频率都在目标分布的 1% 以内，被拒绝之后的位全是 -1。
- 模拟器加 5 种模式：`spec4_perseq`、`draft4_perseq`、`async_draft4_perseq`、`spec2_perseq`、`draft4_adaptive_perseq`（最后一个叠加 batch 大小规则），窗口 1–12、探测间隔随机。假模型对每条序列独立算出它应有的 k（和 `_acc` 模式一样的独立实现，只是按序列），每步核对；ragged 步核对输入的行偏移、布局张量和 `ModelRunner.ragged_layout` 一致、scratch 行存在且不进 target；统一步必须没有 `cu_seqlens_q` 和采样器布局；追赶步要跳过 0 草稿的序列；假草稿模型对空行核对“没有 slot、context_len 是自己的、位置不越界”，并给它一个固定输出。context 里的 `slot_mapping` / `context_lens` / `cu_seqlens_*` 必须是 int32（flash-attn 的要求；作业 233573 抓到的 bug，见下）。KV 读写、哈希块、输出和参考完全一致的检查照旧。GPU 测试加 `spec_per_sequence`、`draft_per_sequence`、`async_draft_per_sequence` 三种模式和一个混合 batch 用例（一条“原样复述这段文字”的抄写任务加四个问题：抄写任务的 n-gram 草稿留着、问题的被砍掉，必须出现 ragged 步，输出逐 token 对参考）。
- 全部 CPU 用例 2142 个全过（19 种模式各 100 个种子加块 256 用例，单元测试 52 个）。

注入 15 种 bug，跑 14 个单元测试 + 220 个模拟器用例（5 种 `_perseq` 模式 × 40 个种子 + 块 256 用例，共 234；在第一版 5b62af5 上跑的，两条探测计数的注入针对的是当时每条序列自己的计数器）：

| 注入的 bug | 失败的用例数（共 234） |
|---|---|
| 每条序列的接受率从不更新（永远用初始值） | 8（只有单元测试抓到：模拟器按调度器自己的接受率数组算预期 k，数组不动它也不动） |
| 每条序列从不探测（计数器永远清零） | 120 |
| 探测计数器从不清零 | 135 |
| 每条序列的窗口不封顶（累计平均） | 2 |
| 阈值比较用 `>` 而不是 `>=` | 31 |
| n-gram 提案不按序列截断 | 69 |
| 调度 token 数按 batch 最大 k 算（块分配多算） | 178 |
| 拒绝采样器不屏蔽补齐的草稿位 | 19 |
| 拒绝采样器"全接受"按最大 k 判断（补齐位全过时不从最后一行采样） | 19 |
| `token_index` 不按序列的草稿数夹住（读到下一条序列的行） | 37 |
| `write_index` 把没有的草稿写进下一条序列的行 | 26 |
| n-gram 的 ragged batch 当成统一形状（不设 varlen context） | 27 |
| 草稿模型追赶时把 0 草稿的序列也补上 | 23 |
| 草稿模型给草稿少的序列的空行也写 KV | 24 |
| target 的前向把 scratch 行也算进去 | 0：模拟器自己拼 decode 步（prepare、propose、采样器），不经过 `ModelRunner.decode_logits`，看不到这一刀；GPU 测试走真实路径（n+1 行对 n 个 slot，写 KV 时就错），留给作业抓 |

### 预期

- 混合负载（edit 和 chat 各半，batch 32）：n-gram 应是 edit 一半 k=4、chat 一半 0；草稿模型应是 edit 一半 k=4、chat 一半 2–3。开关不再抖，`ragged=` 应接近 1。
- 代价是 ragged 步没有 CUDA graph。8B 在 batch 32 下 eager 比 graph 慢多少，之前没单独量过；bench 有 `--enforce-eager`，作业里加一行 eager 的基线就能看出 graph 值多少。如果 eager 的损失吃掉混合负载的收益，下一步是按总行数分桶抓 CUDA graph（输入 pad 到桶大小，布局张量放进缓冲区），varlen kernel 本身可以进 graph。
- 纯 edit 和纯 chat 上应和 batch 级开关一样（各序列的 k 大多相同，ragged 步少），这是回归检查。

### 实测（作业 233573，2026-10-29，代码 5b62af5，49 分钟）

脚本 `cluster/nanovllm-perseq-a40.sbatch`，GPU 测试加三种模式，18 行 bench（混合负载 batch 32：基线、eager 基线、n-gram 开关 batch 级 / 按序列、草稿模型固定 k=4、开关 batch 级 / 按序列；chat、edit batch 32 和混合 batch 8 做回归；最后再跑一次基线看漂移）。原始行在 `docs/benchmarks/qwen3-8b-a40-perseq.txt`。结果有三条教训，两处 bug：

- **草稿模型的 ragged 步在 GPU 上崩了**：`seqlens_k must have dtype int32`。`propose` 里每一步的 context_len 是 `context_lens - num_drafts + min(num_drafts, i)`，`num_drafts` 是 int64，减出来就是 int64，而 `flash_attn_with_kvcache` 要 int32。统一形状的路是 `context_lens - k + i`（Python 整数），dtype 不变，所以以前没碰到。CPU 上假模型只看值不看 dtype，没抓到；现在模拟器核对 context 里所有 int32 张量的 dtype（去掉修法后 4/12 个用例立刻报错）。GPU 测试里 `draft_per_sequence` 的贪心用例过了（草稿模型给自己打草稿，阈值 0.9 之下没出现 ragged 步），采样用例崩了，之后 `async_draft_per_sequence` 三个用例因为崩掉的引擎没释放显存在建引擎时就失败；两行草稿模型按序列的 bench 也没出数。
- **按序列探测把 chat 上 21% 的步变成了 ragged 步**：chat batch 32 上 n-gram 开关按序列是 836.5 对基线 880.5（×0.95，batch 级在 232099 里是 ×0.99）。各序列的探测计数在第一次不一致之后就错开，每条序列在自己的步上提满 K，和其他序列（k=0）凑成 ragged 步，走 eager。改成 batch 同步探测（见上面的做法）。
- **eager 的代价在 batch 8 上压过了收益**：混合负载 batch 8，草稿模型开关 batch 级 310.2（×1.43），按序列 275.3（×1.27），80% 的步是 ragged 步。batch 32 上 eager 基线比 CUDA graph 基线慢 4%（692.6 对 720.2），batch 8 上没单独量（第二次作业加了这一行），但 8B 在 batch 8 一步只有二三十毫秒，几十个 kernel 的启动开销占比更大。所以 ragged 步必须走 CUDA graph，不是可选项；做法见下一小节（补空行，不用分桶）。
- **n-gram 混合负载按序列没赚**：679.4 对 batch 级 675.6、基线 720.2（都是 ×0.94）。接受率从 0.27 升到 0.44（chat 那一半不再白算），ragged 步 48%，但吞吐没动。一部分是 eager（约 4 个点），一部分是错开的探测；第二次作业再看。
- 回归正常：edit batch 32 草稿模型开关按序列 904.0 对 batch 级 908.1（ragged 步 6%，差 0.5%）；两次混合基线 720.2 / 719.7。
- GPU 测试里 `spec_per_sequence` 的贪心用例断言“必须出现 ragged 步”，四个短问题的 n-gram 草稿全被砍到 0，没出现；换成上面说的混合 batch 用例。

第二次作业（`cluster/nanovllm-perseq2-a40.sbatch`，同样的行加 batch 8 的 eager 基线）在排。

### ragged 步的第二种跑法：补空行、保住 CUDA graph（2026-10-29 加）

`LLM(..., speculative_ragged="pad")`（默认 `"varlen"`）。上面 batch 8 的教训是 ragged 步不能 eager。原来想的是按总行数分桶给 varlen kernel 抓 CUDA graph，但有个更省事的做法，就是草稿模型追赶步已经在用的那一招：**所有序列都算 k_max+1 行，草稿少的序列在前面补空行，自己的行靠右对齐**。`flash_attn_with_kvcache` 的因果掩码按右下角对齐（最后一个 query 对准 `cache_seqlens-1`），所以序列 i 的真实行 j 正好看到 `L_i-1+j` 之前的 key，和它单独算一模一样；前面的空行看到的 key 更少，输出是垃圾，但 slot 置 -1 不写 KV、位置夹到 0，采样器按 `token_index` 只取真实行。于是 ragged 步和统一形状的步走同一条路：同一个 kernel、同一套按 q 抓的 CUDA graph、同一个采样器分支（`token_index` 换成 padded 布局），`propose` 一行不改（草稿少的序列的多余提案写进它自己的空行）。

代价是多算的空行：batch 32 混合负载里 16 条 k=4、16 条 k=0 的步，varlen 算 96 行，pad 算 160 行；n-gram 在 chat batch 32 上 k=4（160 行、全拒）比不投机慢 35%（作业 225010），所以 batch 大时空行不是免费的，而 varlen 的 eager 只亏 4%。batch 8 时反过来：行几乎免费，eager 亏 11%。两种模式都留着，第三次作业在 batch 8 和 32 上各测一遍再定默认值（或者按行数自动选）。

- `ModelRunner.ragged_layout(num_drafts, pad=True)`：每条序列 `1+k` 行，`token_index` 指向靠右的真实行，`write_index` 里没有的草稿指向本序列的空行（不再需要 scratch 行）。
- n-gram 路径：按序列先放 `k - k_i` 个空行（token 0、位置 0、slot -1）再放真实行；context 和统一步一样。草稿模型路径：位置 `L-1+c-front` 夹到 0，slot 按 `c >= front` 屏蔽，最后一个 token 放到第 `front` 列；追赶步不变。
- 模拟器：`_perseq` 模式里 varlen 和 pad 随机各半；pad 步必须没有 `cu_seqlens_q`（走 graph 那条路）、布局和 `ragged_layout(pad=True)` 一致、空行 slot 都是 -1、真实行的位置/slot/KV 检查照旧，采样器用 padded 布局。注入 5 种 bug（空行不屏蔽 slot、输入 token 放错列、n-gram 空行写 KV、多余草稿写进别的序列的行、pad 步仍设 varlen context），30 个种子里各有 17–18 个用例抓到。
- GPU 测试加 `spec_per_sequence_pad`、`async_draft_per_sequence_pad`。

## 还没做的

- GPU 上 Qwen3-8B 的吞吐对比：已做（见上）。原文：注意力的 `(B, k+1)` 形状、CUDA graph、编译后的拒绝采样器都只能在 GPU 上验证。A40 作业脚本 `cluster/nanovllm-a40.sbatch` 会跑 GPU 测试（贪心结果和 transformers 对比，同步、异步、投机三种模式），以及吞吐对比：随机 prompt（ngram 基本猜不中，看最坏情况）、代码修改任务（让模型给源文件加 docstring 后原样输出）和开放问答，batch 1/8/32，k=0/2/4。草稿模型的 GPU 测试（0.6B 给自己打草稿，接受率应接近 1）在 `tests/test_model_gpu.py` 的 `draft` 模式里。
- ngram 版和异步调度合并：要把 ngram 查找搬到 GPU 上（vLLM 的做法是 CPU 上查、但等上一步落地，等于放弃了异步）。草稿模型版已经合并（见上）。
- k 自适应：按 batch 大小关草稿已做并复测（batch 32 关草稿时 edit ×0.98、chat ×1.00）；按每个草稿位的接受率调 k 已做并同场复测（草稿模型上 edit batch 1 ×3.2、chat batch 32 ×1.19，固定 k=4 只有 ×1.04；n-gram 在 chat batch 32 上从 ×0.74 回到 ×0.99，差距查清是三条 `torch.compile` 守卫，见上）。每条序列一个 k（混合负载需要）已做（见上）：第一次 A40 作业抓到两个 bug 并发现 eager 的 ragged 步在 batch 8 上亏 11%，于是加了补空行、保住 CUDA graph 的 `speculative_ragged="pad"` 模式；两种模式的 GPU 对比在排。还没做的是把 batch 大小的阈值按 target 何时算力受限来定（这台 A40 上 8B + 0.6B 用 8 太低），让 K 本身跟着负载走，以及按行数在 varlen 和 pad 之间自动选。
- 从草稿模型的分布采样草稿（而不是 argmax），按 `min(1, p/q)` 接受：温度采样下接受率更高，要多存 `(B, k, V)` 的 q。
- EAGLE-3 草稿头（可选）。
- ngram 索引在 CPU 上：长 prompt 第一次 decode 要建索引，每个 token 4 次字典写入。

## 面试时可以讲的取舍

- 为什么 one-hot 提议下拒绝采样是精确的：接受概率 `min(1, p/q)` 在 q 是 one-hot 时就是 `p(d)`；残差 `max(0, p-q)` 就是把 d 的概率置零。任何只依赖上下文的确定性猜法都满足这个条件。
- 为什么 batch 里所有序列都是 k+1 个 query：形状统一才能用 `flash_attn_with_kvcache` 和 CUDA graph。代价是没猜中的序列也算 k+1 个位置；vLLM 用 varlen 注意力支持每条序列草稿数不同。后来加的 `speculative_per_sequence` 就是这条路：只有草稿数不同的步才走 varlen 和 eager，其余步形状照旧。
- 为什么不回收多分配的块：下一步写的位置从 `L+a` 开始，到 `L+a+k` 结束，需要的块数不会少于这一步，回收了马上又要分配。
- 什么时候有用：decode 受内存带宽限制（小 batch、大模型）而且输出复述输入的时候。随机文本或大 batch 下会变慢，这是 benchmark 里要老实展示的。
