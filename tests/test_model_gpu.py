"""Greedy outputs checked token by token against Hugging Face transformers on a real model.

nano-vllm runs in bf16 and the reference in fp32, so near-ties can legitimately flip. Instead of
comparing two free-running generations, the reference scores nano-vllm's own output with teacher
forcing, and every chosen token must be within LOGIT_TOL of the reference's best logit.
"""
import atexit
import os

import pytest
import torch

from nanovllm.engine.sequence import Sequence

pytestmark = pytest.mark.gpu

MODEL = os.path.expanduser(os.environ.get("NANOVLLM_TEST_MODEL", "~/huggingface/Qwen3-0.6B/"))
LOGIT_TOL = float(os.environ.get("NANOVLLM_TEST_LOGIT_TOL", "0.5"))
MAX_TOKENS = 64

QUESTIONS = [
    "introduce yourself",
    "list all prime numbers within 100",
    "What is the capital of France?",
    "Write a haiku about GPUs.",
]
DOCUMENT = " ".join(f"Paragraph {i}: the scheduler batches requests, the block manager hands out KV cache blocks, "
                    f"and the model runner turns them into input tensors for step {i}." for i in range(60))
DOCUMENT_QUESTIONS = ["Summarize the document in one sentence.", "Which component hands out KV cache blocks?"]


@pytest.fixture(scope="module", params=[False, True], ids=["sync", "async"])
def llm(request):
    if not os.path.isdir(MODEL):
        pytest.skip(f"no model at {MODEL}; set NANOVLLM_TEST_MODEL")
    from nanovllm import LLM
    # A small token budget sends the long prompts through chunked prefill.
    engine = LLM(MODEL, max_model_len=4096, max_num_batched_tokens=1024, gpu_memory_utilization=0.5,
                 async_scheduling=request.param)
    yield engine
    engine.exit()
    atexit.unregister(engine.exit)


@pytest.fixture(scope="module")
def reference(llm):
    from transformers import AutoModelForCausalLM
    return AutoModelForCausalLM.from_pretrained(MODEL).float().cuda().eval()


def chat(tokenizer, text):
    return tokenizer.encode(tokenizer.apply_chat_template([{"role": "user", "content": text}], tokenize=False, add_generation_prompt=True))


@torch.inference_mode()
def check(reference, prompt_ids, completion_ids):
    ids = torch.tensor([prompt_ids + completion_ids], device="cuda")
    logits = reference(ids).logits[0, len(prompt_ids) - 1:-1].float()
    chosen = logits.gather(1, torch.tensor(completion_ids, device="cuda")[:, None])[:, 0]
    gap = logits.max(dim=1).values - chosen
    worst = gap.argmax().item()
    assert gap[worst].item() <= LOGIT_TOL, f"token {worst} is {gap[worst].item():.3f} logits below the reference argmax"


def generate(llm, prompts):
    from nanovllm import SamplingParams
    outputs = llm.generate(prompts, SamplingParams(temperature=0, max_tokens=MAX_TOKENS), use_tqdm=False)
    return [output["token_ids"] for output in outputs]


def test_greedy_matches_reference(llm, reference):
    prompts = [chat(llm.tokenizer, q) for q in QUESTIONS]
    for prompt, completion in zip(prompts, generate(llm, prompts)):
        assert completion
        check(reference, prompt, completion)


def test_chunked_prefill_and_prefix_cache_match_reference(llm, reference):
    prompts = [chat(llm.tokenizer, f"{DOCUMENT}\n\n{q}") for q in DOCUMENT_QUESTIONS]
    assert len(prompts[0]) > 1024
    first = generate(llm, prompts)
    # The second call must hit the prefix cache that the first call left behind.
    assert llm.scheduler.block_manager.can_allocate(Sequence(prompts[0])) > 0
    second = generate(llm, prompts)
    for prompt, a, b in zip(prompts, first, second):
        check(reference, prompt, a)
        check(reference, prompt, b)
