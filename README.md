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
| Adaptive speculation | `nanovllm/engine/scheduler.py`, `model_runner.py` | `LLM(..., speculative_disable_by_batch_size=n)`: decode steps run without drafts, and without the draft model, while more than `n` sequences are running, where the GPU is busy anyway; the draft model's KV cache lags behind and the next step with drafts catches it up in one forward pass |
| Tensor-parallel fixes | `sequence.py`, `model_runner.py` | Decode slot mapping used the model's block size instead of the runner's; prefill sent unscheduled tokens to TP workers (both found by auditing the code) |
| Test suite + CI | `tests/`, `.github/workflows/tests.yml` | Runs on CPU with stubs for flash-attn and triton; GPU tests compare greedy outputs with `transformers` |

Design notes: [async scheduling](docs/async-scheduling-design.md), [speculative decoding](docs/spec-decode-design.md).

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

Drop `speculative_model` to draft with n-grams instead (n-gram drafts do not combine with `async_scheduling`). Add `speculative_disable_by_batch_size=8` to speculate only for small batches. Without any of these options the engine behaves like upstream nano-vllm.

## Tests

```bash
pip install pytest
pytest                                                        # CPU: scheduler, block manager, input prep, sampler, n-gram, simulator
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
| | 32 | 592.7 | 923.1 (1.56x, 0.51) | 741.6 (1.25x, 0.91) | 859.0 (1.45x, 0.85) | 836.8 | 450.0 (0.76x) |
| chat, greedy | 1 | 33.8 | 40.0 (1.18x, 0.05) | 56.1 (1.66x, 0.55) | 59.4 (1.76x, 0.41) | 60.6 | 60.1 |
| | 8 | 250.1 | 237.2 (0.95x, 0.05) | 328.4 (1.31x, 0.57) | 346.5 (1.39x, 0.41) | 351.5 | 329.4 |
| | 32 | 878.4 | 674.6 (0.77x, 0.05) | 1019.8 (1.16x, 0.55) | 965.3 (1.10x, 0.40) | 984.4 | 746.0 (0.85x) |
| edit, T=0.6 | 8 | 195.4 | 372.6 (1.91x, 0.50) | 328.1 (1.68x, 0.91) | 404.6 (2.07x, 0.87) | 404.5 | 393.5 |
| chat, T=0.6 | 1 | 33.8 | 39.9 (1.18x, 0.05) | 55.9 (1.65x, 0.55) | 60.1 (1.78x, 0.42) | 60.7 | 60.6 |

Also at batch 1, greedy: draft k=6 gives 109.0 tok/s on edit (3.3x, acceptance 0.82) and 56.5 on chat (acceptance 0.31). Async scheduling alone is within 2% of the baseline on 8B in every row. The GPU test suite (18 tests, every engine mode) passed on this model pair as well.

What the numbers say:

- A 0.6B drafter speeds up the 8B model on every workload and batch tested: 2.9x on code editing and 1.8x on chat at batch 1, still 1.45x and 1.16x at batch 32. Sampling at T=0.6 costs nothing over greedy, so the rejection sampler is doing its job.
- N-gram drafts are the honest worst case: 2.8x on code editing, where the output copies the input (acceptance 0.5), but useless on chat (acceptance 0.05), where they slow decoding down at batch 8 and 32.
- Async scheduling gave +10% on Qwen3-0.6B and nothing measurable on Qwen3-8B: the CPU work it hides is a fixed few milliseconds per step, large next to a 0.6B forward pass and small next to an 8B one.
- Adaptive speculation behaves as designed below its threshold (same numbers as draft k=4) but is *slower than the baseline* at batch 32 (0.76x on edit, 0.85x on chat), where it switches drafts off. The draft model still runs one step per decode step to keep its KV cache current, and that step reads the draft's whole KV cache, which for Qwen3-0.6B is 28 layers of the same per-layer size as the 8B's 36. Letting the draft cache lag while drafts are off, and catching it up when they resume, is the fix in progress; until then the option only pays off when the target is compute-bound.

## Credits

Upstream code and logo: [GeeeekExplorer/nano-vllm](https://github.com/GeeeekExplorer/nano-vllm), MIT License. This repository keeps upstream's history; everything after commit `bb823b3` is new.
