"""Speculative decoding benchmark: throughput and draft acceptance without drafts, with n-gram
drafts, and with a draft model (--speculative-model).

The "edit" workload asks the model to return a source file with docstrings added, so most of the
output repeats the prompt (where prompt lookup helps); the "chat" workload asks open questions.
"""
import argparse
import glob
import os
import time

from nanovllm import LLM, SamplingParams

QUESTIONS = [
    "Explain how a KV cache speeds up transformer inference.",
    "Write a short story about a robot learning to paint.",
    "What are the trade-offs between tensor parallelism and pipeline parallelism?",
    "Describe the water cycle to a ten-year-old.",
    "List ten tips for writing clean Python code, with an example for each.",
    "Compare TCP and UDP.",
    "How does a hash map handle collisions?",
    "Summarize the history of the printing press.",
]


def build_prompts(tokenizer, workload, num_prompts, max_prompt_tokens):
    if workload == "edit":
        files = sorted(glob.glob(os.path.join(os.path.dirname(os.path.abspath(__file__)), "nanovllm", "**", "*.py"), recursive=True))
        texts = []
        for path in files:
            source = open(path).read().strip()
            if source and len(tokenizer.encode(source)) <= max_prompt_tokens:
                texts.append(f"Add a one-line docstring to every function and class in this file. Reply with the complete file and nothing else.\n\n```python\n{source}\n```")
    else:
        texts = QUESTIONS
    texts = [texts[i % len(texts)] for i in range(num_prompts)]
    return [tokenizer.apply_chat_template([{"role": "user", "content": t}], tokenize=True, add_generation_prompt=True, enable_thinking=False) for t in texts]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=os.path.expanduser("~/huggingface/Qwen3-0.6B/"))
    parser.add_argument("--workload", choices=["edit", "chat"], default="edit")
    parser.add_argument("--num-speculative-tokens", type=int, default=0)
    parser.add_argument("--speculative-model", default=None, help="draft model directory; n-gram drafts if omitted")
    parser.add_argument("--batch-size", type=int, default=1, help="requests per generate call")
    parser.add_argument("--num-prompts", type=int, default=8)
    parser.add_argument("--max-prompt-tokens", type=int, default=2048)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--async-scheduling", action="store_true")
    args = parser.parse_args()

    llm = LLM(args.model, enforce_eager=args.enforce_eager, max_model_len=4096, async_scheduling=args.async_scheduling, num_speculative_tokens=args.num_speculative_tokens,
              speculative_model=args.speculative_model)
    prompts = build_prompts(llm.tokenizer, args.workload, args.num_prompts, args.max_prompt_tokens)
    sampling_params = SamplingParams(temperature=args.temperature, max_tokens=args.max_tokens)
    llm.generate(prompts[:1], SamplingParams(max_tokens=16), use_tqdm=False)    # warmup

    scheduler = llm.scheduler
    scheduler.num_draft_tokens = scheduler.num_accepted_tokens = 0
    num_tokens = 0
    t = time.perf_counter()
    for i in range(0, len(prompts), args.batch_size):
        outputs = llm.generate(prompts[i:i + args.batch_size], sampling_params, use_tqdm=False)
        num_tokens += sum(len(output["token_ids"]) for output in outputs)
    t = time.perf_counter() - t
    acceptance = scheduler.num_accepted_tokens / scheduler.num_draft_tokens if scheduler.num_draft_tokens else 0.0
    drafter = "none" if not args.num_speculative_tokens else "model" if args.speculative_model else "ngram"
    print(f"workload={args.workload} drafter={drafter} k={args.num_speculative_tokens} async={int(args.async_scheduling)} batch={args.batch_size} T={args.temperature} "
          f"tokens={num_tokens} time={t:.2f}s throughput={num_tokens / t:.1f}tok/s acceptance={acceptance:.3f}")


if __name__ == "__main__":
    main()
