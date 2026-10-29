<p align="center">
<img width="300" src="assets/logo.png">
</p>

# nano-vllm-spec

[![Tests](https://github.com/yczhou25/nano-vllm-spec/actions/workflows/tests.yml/badge.svg)](https://github.com/yczhou25/nano-vllm-spec/actions/workflows/tests.yml) [![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

[nano-vllm](https://github.com/GeeeekExplorer/nano-vllm) (a ~1,200-line vLLM written from scratch by Xingkai Yu) extended with **async scheduling** and **speculative decoding**, plus a test suite that checks every KV-cache write and read.

在 nano-vllm 的基础上加了**异步调度**和**投机解码**（n-gram 草稿和小模型草稿，两者都可以和异步调度组合），并补了一套在 CPU 上就能跑的测试：用假模型驱动真实的调度器、块管理器和输入准备，逐一检查每次 KV cache 的写和读。

## What's new

| Feature | Where | Notes |
|---|---|---|
| Async scheduling | `nanovllm/engine/llm_engine.py`, `scheduler.py` | The CPU schedules step *n+1* and prepares its inputs while the GPU runs step *n*; sampled tokens are filled in one step late. `LLM(..., async_scheduling=True)` |
| N-gram speculative decoding | `nanovllm/engine/ngram.py`, `model_runner.py` | Prompt-lookup drafts: propose `k` tokens from earlier text, verify them in one forward pass. `LLM(..., num_speculative_tokens=k)` |
| Draft-model speculative decoding | `nanovllm/engine/model_runner.py`, `block_manager.py` | A smaller model of the same family drafts; it keeps its own KV cache and block tables. `LLM(..., speculative_model=PATH)`; combines with `async_scheduling` |
| Rejection sampling | `nanovllm/layers/sampler.py` | Accept/reject drafts so outputs follow the target model's distribution; greedy with `temperature=0` |
| Adaptive speculation | `nanovllm/engine/scheduler.py`, `model_runner.py` | The scheduler picks each decode step's `k`: `LLM(..., speculative_disable_by_batch_size=n)` drops drafts while more than `n` sequences are running, where the GPU is busy anyway; `speculative_min_acceptance=r` tracks the acceptance rate of every draft position over its last 64 samples (at least 8 steps) and proposes only the leading positions that reach `r` (down to none), with one full-`k` probe step every 16 steps to let positions back in. Steps without drafts skip the draft model; its KV cache lags behind and the next step with drafts catches it up in one forward pass |
| Tensor-parallel fixes | `sequence.py`, `model_runner.py` | Decode slot mapping used the model's block size instead of the runner's; prefill sent unscheduled tokens to TP workers (both found by auditing the code) |
| Test suite + CI | `tests/`, `.github/workflows/tests.yml` | Runs on CPU with stubs for flash-attn and triton; GPU tests compare greedy outputs with `transformers` |

Design notes: [async scheduling](docs/async-scheduling-design.md), [speculative decoding](docs/spec-decode-design.md). Write-up (Chinese): [在 1,400 行的推理引擎里实现异步调度和投机解码](docs/blog/spec-decode-in-1400-lines.md).

## Quick start

```bash
pip install git+https://github.com/yczhou25/nano-vllm-spec.git
huggingface-cli download --resume-download Qwen/Qwen3-8B --local-dir ~/huggingface/Qwen3-8B/
huggingface-cli download --resume-download Qwen/Qwen3-0.6B --local-dir ~/huggingface/Qwen3-0.6B/
```

```python
from nanovllm import LLM, SamplingParams

llm = LLM("~/huggingface/Qwen3-8B/",
          async_scheduling=True,
          num_speculative_tokens=4,
          speculative_model="~/huggingface/Qwen3-0.6B/")
outputs = llm.generate(["Hello, nano-vllm."], SamplingParams(temperature=0.6, max_tokens=256))
print(outputs[0]["text"])
```

Drop `speculative_model` to draft with n-grams instead (n-gram drafts do not combine with `async_scheduling`). Add `speculative_disable_by_batch_size=8` to speculate only for small batches, or `speculative_min_acceptance=0.2` to let the measured acceptance rate of each draft position pick `k` per step, from `num_speculative_tokens` down to none (n-gram drafts on chat, for example, get switched off). Without any of these options the engine behaves like upstream nano-vllm.

## Tests

```bash
pip install pytest
pytest                                                        # CPU: scheduler, block manager, input prep, sampler, n-gram, adaptive k, simulator
NANOVLLM_TEST_MODEL=~/huggingface/Qwen3-0.6B/ pytest -m gpu   # GPU: greedy outputs vs. transformers with teacher forcing
```

`tests/engine_sim.py` is a randomized simulator: a fake model (and fake draft model) records what the engine writes to and reads from the KV cache, across chunked prefill, prefix caching, preemption, async scheduling and both kinds of speculation. `NANOVLLM_SIM_CASES=500 pytest tests/test_engine_sim.py` runs more cases.

## Benchmarks

Measured on one NVIDIA A40 (bf16, CUDA graphs on). `bench.py` is upstream's throughput benchmark on random tokens; `bench_spec.py` adds `--async-scheduling`, `--num-speculative-tokens` and `--speculative-model` and uses workloads with real text: a code-editing task whose output repeats most of its input, and open-ended chat. On the same GPU the whole test suite passes (2,680 tests), including greedy outputs that match `transformers` in fp32 within 0.5 logits for all five engine configurations.

**Qwen3-0.6B.** The draft model in this run is Qwen3-0.6B itself, so its drafts are always accepted and it cannot be faster than the baseline; those rows only exercise the verification path.

| Benchmark | Configuration | Throughput | Acceptance |
|---|---|---|---|
| `bench.py`, 64 random prompts, up to 256 in / 256 out | baseline | 6,200 tok/s | |
| | `async_scheduling` | 6,842 tok/s (+10%) | |
| | n-gram drafts, k=4 | 1,804 tok/s | ~0 (random tokens: nothing to look up) |
| `bench_spec.py --workload edit --batch-size 8`, greedy | baseline | 1,196 tok/s | |
| | `async_scheduling` | 1,300 tok/s (+9%) | |
| | n-gram drafts, k=4 | 2,376 tok/s (1.99x) | 0.80 |
| | draft model (0.6B itself), k=4 | 1,023 tok/s | 1.00 |
| | draft model + `async_scheduling`, k=4 | 1,066 tok/s | 1.00 |

**Qwen3-8B with Qwen3-0.6B drafting.** `bench_spec.py --model Qwen3-8B --speculative-model Qwen3-0.6B`, bf16 on the same A40 (Slurm job 225010, 1 h 39 min; raw rows in `docs/benchmarks/qwen3-8b-a40.txt`). Throughput counts the whole run (prefill + decode) in tok/s; acceptance = accepted drafts / proposed drafts; speedups are against the baseline of the same row. "Adaptive" is draft k=4 with `speculative_disable_by_batch_size=8`.

| Workload | Batch | Baseline | n-gram k=4 | Draft k=2 | Draft k=4 | Draft k=4 + async | Adaptive |
|---|---|---|---|---|---|---|---|
| edit, greedy | 1 | 33.4 | 94.3 (2.8x, acc 0.50) | 72.9 (2.2x, 0.93) | 96.4 (2.9x, 0.88) | 97.9 | 96.2 |
| | 8 | 194.7 | 404.4 (2.1x, 0.52) | 280.5 (1.4x, 0.91) | 404.3 (2.1x, 0.86) | 409.1 | 391.2 |
| | 32 | 592.7 | 923.1 (1.56x, 0.51) | 741.6 (1.25x, 0.91) | 859.0 (1.45x, 0.85) | 836.8 | 578.2 (0.98x)† |
| chat, greedy | 1 | 33.8 | 40.0 (1.18x, 0.05) | 56.1 (1.66x, 0.55) | 59.4 (1.76x, 0.41) | 60.6 | 60.1 |
| | 8 | 250.1 | 237.2 (0.95x, 0.05) | 328.4 (1.31x, 0.57) | 346.5 (1.39x, 0.41) | 351.5 | 329.4 |
| | 32 | 878.4 | 674.6 (0.77x, 0.05) | 1019.8 (1.16x, 0.55) | 965.3 (1.10x, 0.40) | 984.4 | 874.5 (1.00x)† |
| edit, T=0.6 | 8 | 195.4 | 372.6 (1.91x, 0.50) | 328.1 (1.68x, 0.91) | 404.6 (2.07x, 0.87) | 404.5 | 393.5 |
| chat, T=0.6 | 1 | 33.8 | 39.9 (1.18x, 0.05) | 55.9 (1.65x, 0.55) | 60.1 (1.78x, 0.42) | 60.7 | 60.6 |

Also at batch 1, greedy: draft k=6 gives 109.0 tok/s on edit (3.3x, acceptance 0.82) and 56.5 on chat (acceptance 0.31). Async scheduling alone is within 2% of the baseline on 8B in every row. The GPU test suite (18 tests, every engine mode) passed on this model pair as well.

† Re-measured in job 225388 after the draft-cache fix below (commit 5550679; raw rows in `docs/benchmarks/qwen3-8b-a40-adaptive-recheck.txt`), against that run's own baseline (591.2 on edit, 877.4 on chat). Before the fix, job 225010 measured 450.0 (0.76x) and 746.0 (0.85x) in these two cells.

**Adaptive k (job 228654, 2026-10-27, 64 min; raw rows in `docs/benchmarks/qwen3-8b-a40-adaptive-k2.txt`).** `speculative_min_acceptance` picks each step's k from the measured acceptance rate of every draft position. Same models and A40; every cell is read against this run's own no-draft baseline, measured in the same job for every workload and batch. (The first check, job 228429, compared against job 225010's baselines; this run's node is 13-17% faster on every baseline, so the speedups published from it were inflated. Its rows stay in `docs/benchmarks/qwen3-8b-a40-adaptive-k.txt`.) Cells are tok/s (speedup against the baseline, acceptance rate).

| Workload | Batch | Baseline | Draft k=4, fixed | Draft K=4, `min_acceptance=0.2` | Draft k=6, fixed | Draft K=6, `min_acceptance=0.3` |
|---|---|---|---|---|---|---|
| edit, greedy | 1 | 38.4 | | | 124.3 (3.24x, acc 0.82) | 123.7 (3.22x, 0.82) |
| | 32 | 683.6 | 978.2 (1.43x, 0.85) | 970.8 (1.42x, 0.86) | 975.4 (1.43x, 0.79) | 963.9 (1.41x, 0.79) |
| chat, greedy | 1 | 39.0 | | | 63.6 (1.63x, 0.31) | 63.4 (1.63x, 0.52) |
| | 32 | 1025.3 | 1065.3 (1.04x, 0.40) | 1222.2 (1.19x, 0.48) | 854.1 (0.83x, 0.31) | 1103.4 (1.08x, 0.50) |
| mixed edit+chat, greedy | 32 | 847.1 | 798.9 (0.94x, 0.57) | 909.1 (1.07x, 0.57) | | |

| Workload | Batch | Baseline | n-gram k=4, fixed | n-gram k=1, fixed | n-gram K=4, `min_acceptance=0.2` | its decode steps by k=0..4 |
|---|---|---|---|---|---|---|
| edit, greedy | 32 | 683.6 | | | 1051.5 (1.54x, 0.51) | 2, 0, 0, 0, 405 |
| chat, greedy | 1 | 39.0 | | | 39.5 (1.01x, 0.13) | 3268, 237, 0, 0, 219 |
| | 8 | 289.0 | | | 271.9 (0.94x, 0.10) | 936, 15, 0, 0, 60 |
| | 32 | 1025.3 | 754.3 (0.74x, 0.05) | 922.1 (0.90x, 0.14) | 902.8 (0.88x, 0.10) | 928, 23, 0, 0, 60 |
| mixed edit+chat, greedy | 32 | 847.1 | 715.6 (0.84x, 0.17) | | 746.8 (0.88x, 0.27) | 523, 144, 79, 11, 241 |

With the draft model the switch pays exactly where draft positions get rejected, and costs nothing elsewhere. On chat at batch 32 it turns fixed k=4 from 1.04x into 1.19x and fixed k=6 from a loss (0.83x) into 1.08x, by settling on k=3 (380 of 453 steps) and k=2-3 (331 + 141 of 540 steps) respectively; on the mixed batch it turns 0.94x into 1.07x. Where every position clears the threshold (edit: positions accepted 0.72 to 0.92 of the time) it keeps k=K and matches fixed k within 1%. At batch 1 on chat it cuts K=6 to k=2 for most steps and the throughput does not move (63.4 against 63.6): a batch-1 step reads the weights once whatever its q, so the dropped positions were free, as the design note predicted. The 3.7x at batch 1 on edit that job 228429 reported was fixed k=6 from one job against adaptive k from another; measured together they are the same 3.2x.

N-gram drafts with the switch keep the edit speedup (1.54x) and are back at the baseline on chat at batch 1 (1.01x), but not at batch 8 (0.94x) or 32 (0.88x, up from 0.74x without the switch, and from 0.83x of baseline in job 228429 before the sampler and window fixes of 44129e2), nor on the mixed batch (0.88x). On chat at batch 32 the switch ran 928 of 1011 steps without drafts and probed with k=4 sixty times. Job 230846 (raw rows in `docs/benchmarks/qwen3-8b-a40-adaptive-k3.txt`) took that gap apart on its own node, where the baseline repeated three times within 0.3% (878.9 / 876.4 / 879.2, clocks pinned at 1740 MHz): the drafts-off path with the proposer present costs nothing (878.4 with drafts off at every step), and the switch costs 5.9% with a probe every 16 steps but still 5.2% with one every 64, so most of it is a fixed cost per run rather than a cost per probe. The probes themselves, at 1.4x a plain step, are about 1 point. The fixed part was `torch.compile`: it specializes sizes 0 and 1, and the samplers were only warmed up for a symbolic batch, so the first step with one sequence (every batch's tail) and one draft recompiled the rejection sampler inside the timed run, about 1.5 s per row (reproduced on CPU with the recompile log: six graphs in all, one of them compiled at the tail). Warming up batch size 1 did not remove it, and the recompile log of job 231468 (`docs/benchmarks/qwen3-8b-a40-adaptive-k4.txt`) showed why: `torch.compile` also guards on torch's default device, which `ModelRunner` had set to `cuda` while loading the model and warming up, and back to `cpu` for serving, so every compiled layer and both samplers were compiled again at the first real request and the startup warmup never counted (for a user that is the first request's latency; the bench's own warmup call hid it from the timer, but then the batch-1 warmup prompt 0/1-specialized the samplers and the timed run recompiled the rejection sampler twice). The warmup and the graph capture now run under the serving default device, with explicit device and dtype on every warmup tensor. Job 231709 (`docs/benchmarks/qwen3-8b-a40-adaptive-k5.txt`) confirmed that guard gone and the switch still at 0.94x (829.7 against 880.2 / 880.8, drafts off 881.0): the samplers were compiled again at the first real step on a `requires_grad` guard, because `run` and `run_async` were not under `torch.inference_mode` (only the model forward was), so the real logits carried autograd and the warmup's did not; after that the batch-1 warmup prompt 0/1-specialized them and a stride guard (the real drafts are a column slice of the step's input ids, the warmup's were contiguous) recompiled the rejection sampler twice more inside the timed run. The decode step and the sampler warmup now run under inference mode and the warmup drafts are a column slice like the real ones; on CPU the recompile log shows six graphs at warmup and none in a bench-like serving sequence. Job 232099 (`docs/benchmarks/qwen3-8b-a40-adaptive-k6.txt`, same node class as 231709) closed it: the switch is at 0.99x of the baseline on chat at batch 32 (871.8 against 881.2 / 881.6) and batch 8 (249.9 against 251.6), the drafts-off path is still free (881.4), and the recompile log has no sampler compile inside any timed run; every one falls in the startup warmup. What is left of the switch's cost is the probes. The draft model rows of the same job: fixed k=4 1.14x, K=4 with the switch 1.20x. So the 12 points of job 228654 were 7 points of node, 4 points of startup compilation leaking into the timed run, and 1 point of probes. On the mixed batch the switch oscillates (523 steps at k=0, 241 at k=4) because the batch-wide acceptance rates (0.31, 0.27, 0.25, 0.24) sit just above the 0.2 threshold: the edit half of the batch wants k=4 and the chat half wants 0, and one k per step cannot serve both. That is the design limit this switch has: a k per sequence (ragged drafts, as vLLM does) is the next phase.

What the numbers say:

- A 0.6B drafter speeds up the 8B model on every workload and batch tested: 2.9x on code editing and 1.8x on chat at batch 1, still 1.45x and 1.16x at batch 32. Sampling at T=0.6 costs nothing over greedy, so the rejection sampler is doing its job.
- N-gram drafts are the honest worst case: 2.8x on code editing, where the output copies the input (acceptance 0.5), but useless on chat (acceptance 0.05), where they slow decoding down at batch 8 and 32; the acceptance switch turns them off there and brings chat back to within 1% of the baseline (job 232099).
- Async scheduling gave +10% on Qwen3-0.6B and nothing measurable on Qwen3-8B: the CPU work it hides is a fixed few milliseconds per step, large next to a 0.6B forward pass and small next to an 8B one.
- Adaptive speculation behaves as designed below its threshold (same numbers as draft k=4) and now costs nothing above it: 0.98x on edit and 1.00x on chat at batch 32, where it switches drafts off (0.96x and 1.01x with async scheduling on top). Its first version measured 0.76x and 0.85x there, because the draft model still ran one step per decode step to keep its KV cache current, and that step reads the draft's whole KV cache, which for Qwen3-0.6B is 28 layers of the same per-layer size as the 8B's 36. The fix lets the draft cache lag while drafts are off and catches it up in one forward pass when they resume (the design note has the mutation table).
- The threshold is the remaining lesson: on this A40 the 8B target is still memory-bound at batch 32, so leaving draft k=4 on (1.45x on edit, 1.10x on chat) beats switching it off at 8 sequences. The threshold belongs where the target turns compute-bound, and the better switch is measured acceptance: it would turn n-gram drafts off on chat (acceptance 0.05) and leave model drafts (0.40 to 0.85) on. `speculative_min_acceptance` is that switch, extended to pick `k` per step from each draft position's acceptance rate (position `j` is only accepted when the `j-1` before it were, so the rates fall with the position and the cut lands where they drop below the threshold); the tables above show one setting giving 3.2x on edit at batch 1 and 1.19x on chat at batch 32, where fixed k=4 gets 1.04x and fixed k=6 loses.

## Credits

Upstream code and logo: [GeeeekExplorer/nano-vllm](https://github.com/GeeeekExplorer/nano-vllm), MIT License. This repository keeps upstream's history; everything after commit `bb823b3` is new.
