"""Time the samplers' kernels on their own: DraftSampler, Sampler and RejectionSampler (one-hot
drafts and sampled drafts) on random logits of the model's vocabulary, per batch size, with the
eager argmax and logsumexp of the same rows for comparison. Prints microseconds per call (CUDA
events, after a warmup that also compiles). Runs with the repository's sampler.py, so the same
script before and after a change to it measures that change.

    python bench_sampler.py --rows 1 8 32 --vocab 151936 --k 4 --temperature 0.8

--profile also runs each op under torch.profiler and prints its CUDA kernels: launches per call,
microseconds per call, so a call's wall time splits into GPU time and launch gaps.
"""
import argparse
from collections import defaultdict

import torch

from nanovllm.layers.sampler import DraftSampler, RejectionSampler, Sampler


def microseconds(fn, iters: int = 50) -> float:
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters * 1000


def kernel_profile(fn, iters: int = 20) -> list[tuple[str, float, float]]:
    """(kernel name, launches per call, CUDA microseconds per call) of the CUDA kernels fn launches, largest first."""
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
        for _ in range(iters):
            fn()
        torch.cuda.synchronize()
    totals: dict[str, list[float]] = defaultdict(lambda: [0, 0.0])
    for event in prof.events():
        if event.device_type == torch.autograd.DeviceType.CUDA:
            totals[event.name][0] += 1
            totals[event.name][1] += event.self_device_time_total if hasattr(event, "self_device_time_total") else event.self_cuda_time_total
    rows = [(name, calls / iters, total / iters) for name, (calls, total) in totals.items()]
    return sorted(rows, key=lambda row: -row[2])


def time_rows(rows: int, vocab: int, k: int, temperature: float, dtype: torch.dtype, samplers, profile: bool = False):
    sampler, rejection_sampler, draft_sampler = samplers
    temperatures = torch.full((rows,), temperature, device="cuda")
    logits = torch.randn(rows, vocab, dtype=dtype, device="cuda")
    step_logits = torch.randn(rows * (k + 1), vocab, dtype=dtype, device="cuda")
    drafts = torch.randint(vocab, (rows, k), device="cuda")
    draft_logits = torch.randn(rows, k, vocab, dtype=dtype, device="cuda")
    draft_log_norms = torch.logsumexp(draft_logits.float() / temperature, dim=-1)
    ops = {
        "eager argmax": lambda: logits.argmax(dim=-1),
        "eager logsumexp (float, / T)": lambda: torch.logsumexp(logits.float() / temperature, dim=-1),
        "Sampler": lambda: sampler(logits, temperatures),
        "DraftSampler": lambda: draft_sampler(logits, temperatures),
        "RejectionSampler, one-hot drafts": lambda: rejection_sampler(step_logits, temperatures, drafts),
        "RejectionSampler, sampled drafts": lambda: rejection_sampler(step_logits, temperatures, drafts, draft_logits=draft_logits, draft_log_norms=draft_log_norms),
    }
    for name, fn in ops.items():
        print(f"{rows:>4} {name:<40} {microseconds(fn):>8.1f}", flush=True)
        if profile:
            kernels = kernel_profile(fn)
            print(f"       GPU {sum(us for _, _, us in kernels):.1f} us/call in {sum(n for _, n, _ in kernels):.0f} launches:")
            for kernel, launches, us in kernels:
                print(f"       {us:>7.1f} us  x{launches:<5.1f} {kernel[:90]}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, nargs="+", default=[1, 8, 32])
    parser.add_argument("--vocab", type=int, default=151936)
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--profile", action="store_true", help="also list each op's CUDA kernels (launches and us per call)")
    args = parser.parse_args()
    samplers = Sampler(), RejectionSampler(), DraftSampler()
    torch.manual_seed(0)
    print(f"{'rows':>4} {'op':<40} {'us/call':>8}")
    with torch.inference_mode():
        for rows in args.rows:
            time_rows(rows, args.vocab, args.k, args.temperature, getattr(torch, args.dtype), samplers, args.profile)


if __name__ == "__main__":
    main()
