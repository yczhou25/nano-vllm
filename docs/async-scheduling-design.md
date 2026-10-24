# nano-vllm 异步调度：设计说明

分支：`yczhou25/nano-vllm` 的 `async-scheduling`（基于 `dev`）。用 `LLM(path, async_scheduling=True)` 开启，默认关闭。

## 要解决的问题

原来的 `LLMEngine.step` 是串行的：调度、准备输入、跑 GPU、`.tolist()` 等 GPU 跑完、后处理。CPU 在做调度和后处理时，GPU 是空着的；GPU 在算的时候，CPU 也在等。模型越小、batch 越大，CPU 这部分占比越高。

## 做法

第 k 步发射到 GPU 之后，不等它跑完，先回头处理第 k-1 步的结果，然后马上调度第 k+1 步。这样 GPU 上始终排着下一步的工作。

```
step():
    seqs = schedule()                 # 第 k 步；第 k-1 步的 token 还在 GPU 上
    out  = run_async(seqs)            # 发射第 k 步，立刻返回
    batch = commit(seqs)              # 推进 KV 进度，记下“在途 token”
    postprocess(上一步的 batch, 上一步的 out.result())   # 等第 k-1 步，追加 token、判断结束
```

### 关键点

1. **在途 token（in-flight）**：调度第 k 步时，第 k-1 步采样出的 token 还没回到 CPU。`Sequence` 新增两个字段：`num_inflight` 是有几个 token 在途，`inflight_row` 是它在上一步输出里的行号。decode 时位置和长度都按“已确认 + 在途”算，输入 token 直接在 GPU 上从上一步的输出里取（`torch.where(rows >= 0, last_token_ids[rows], ids)`），不经过 CPU。
2. **把 postprocess 拆成两半**：`commit` 在发射后立刻推进 `num_cached_tokens`，所以下一次调度能接着排 chunked prefill 的下一段；`postprocess` 在结果回来后做块哈希（prefix cache）、追加 token、判断结束。块哈希用 commit 时记下的 `(start, num_tokens)`，因为那时序列可能已经被调度到下一步了。
3. **结束晚一步**：碰到 EOS 的序列，下一步已经被调度了一次。它的结果直接丢掉；块在 postprocess 时释放。这时 GPU 上那一步还在写这些块，但同一个 CUDA stream 上的操作按顺序执行，之后拿到这些块的序列，写入一定排在后面，所以是安全的。`max_tokens` 事先知道，不会多排。
4. **抢占**：被抢占的序列可能还有一个在途 token。它的块已经释放，postprocess 只追加 token，不做哈希。在 token 回来之前，它不会被重新 prefill。
5. **张量并行**：rank 0 采样后用 `dist.broadcast` 把 token 发给其他 rank，作为下一步的输入。另外，原来 rank 0 等 GPU 跑完时，worker 一定已经读走了共享内存里的上一条消息。异步之后这个前提不成立，所以 `write_shm` 会先等所有 worker 清掉事件再写。

## 正确性验证（CPU）

`tests/engine_sim.py` 用真实的 `LLMEngine.step`、调度器、块管理器和 `prepare_prefill`/`prepare_decode` 跑随机负载，同步和异步都跑。假模型会检查每一次 KV 写入和读取，并要求在途 token 的输入等于上一步采样出的 token。每个请求的输出都必须和不做任何 batching 的参考输出完全一致。

在 300 个异步随机用例里，有 85% 的 decode 输入来自在途 token，608 次带着在途 token 被抢占，98 次结束晚一步，1090 次空调度（只等上一步结果）。模拟器还会统计每条序列采样过的 token，确认多算的最多一步，而且只发生在碰到 EOS 时。

我故意往代码里注入了 8 种异步相关的 bug，测试全部抓到了（150 个异步用例里失败的个数）：

| 注入的 bug | 失败用例 |
|---|---|
| decode 不用在途 token，直接用 CPU 上的旧 token | 161 |
| commit 时不记在途 token 的行号 | 161 |
| may_append 不算在途 token（少分配块） | 160 |
| can_append 不算在途 token | 78 |
| 已结束的序列被 postprocess 再处理一次 | 14 |
| 最后一个 token 在途时仍然调度（多算） | 164 |
| 被抢占的序列也去做块哈希 | 73 |
| `write_shm` 不等 worker 读完就覆盖（双进程测试） | 1/1 |

另外有一个保护条件（“在途 token 没回来前不重新 prefill”）注入后没有任何用例失败。推下来发现它在一步流水线里根本走不到：被抢占序列的在途 token，一定会在下一次调度之前处理完。所以把它删掉了。

## 还没做的

- GPU 上的实测：同步和异步的吞吐、每步 GPU 空闲时间（nsys），Qwen3-0.6B，batch 1 到 256。
- 张量并行的异步路径只在 CPU 上推演过，要在 2 张卡上跑一遍。
- 采样还是每步同步拷回 CPU 一次（非阻塞拷贝加 event）。以后可以把 EOS 检查也放到 GPU 上。

## 面试时可以讲的取舍

- 为什么只往前看一步：看得更远，在途 token 会变多，抢占、结束和块分配都要按多个在途 token 处理，收益却很小。CPU 只要比 GPU 快，往前一步就足够把 GPU 填满。
- 浪费：每个碰到 EOS 的序列多算一步。和 vLLM 的 async scheduling 一样，用这一步换掉每步的同步等待。
- 和 vLLM 的对应关系：vLLM V1 的 `AsyncScheduler` 用 output placeholder 表示在途 token，model runner 保留上一步的 `sampled_token_ids` 作为下一步的输入。这里的 `num_inflight`/`inflight_row` 是同一个思路的简化版。
