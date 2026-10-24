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
# The draft model defaults to the target itself, whose greedy drafts should nearly all be accepted.
DRAFT_MODEL = os.path.expanduser(os.environ.get("NANOVLLM_TEST_DRAFT_MODEL", MODEL))
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


ENGINE_OPTIONS = {
    "sync": {},
    "async": {"async_scheduling": True},
    "spec": {"num_speculative_tokens": 4},
    "draft": {"num_speculative_tokens": 4, "speculative_model": DRAFT_MODEL},
    "async_draft": {"async_scheduling": True, "num_speculative_tokens": 4, "speculative_model": DRAFT_MODEL},
    # Drafts only while at most 2 sequences run: the 4 questions decode without them, the 2 documents with them.
    "draft_adaptive": {"num_speculative_tokens": 4, "speculative_model": DRAFT_MODEL, "speculative_disable_by_batch_size": 2},
}


@pytest.fixture(scope="module", params=list(ENGINE_OPTIONS))
def llm(request):
    if not os.path.isdir(MODEL):
        pytest.skip(f"no model at {MODEL}; set NANOVLLM_TEST_MODEL")
    from nanovllm import LLM
    # A small token budget sends the long prompts through chunked prefill.
    engine = LLM(MODEL, max_model_len=4096, max_num_batched_tokens=1024, gpu_memory_utilization=0.5,
                 **ENGINE_OPTIONS[request.param])
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
    scheduler = llm.scheduler
    scheduler.num_draft_tokens = scheduler.num_accepted_tokens = 0
    for prompt, completion in zip(prompts, generate(llm, prompts)):
        assert completion
        check(reference, prompt, completion)
    if scheduler.has_draft_model and scheduler.num_draft_tokens and os.path.samefile(DRAFT_MODEL, MODEL):
        # The target drafting for itself: every draft matches the greedy token unless bf16 noise
        # between the draft and verification passes flips a near-tie.
        assert scheduler.num_accepted_tokens / scheduler.num_draft_tokens > 0.8


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


def test_sampling_returns_max_tokens(llm):
    from nanovllm import SamplingParams
    prompts = [chat(llm.tokenizer, q) for q in QUESTIONS]
    outputs = llm.generate(prompts, SamplingParams(temperature=0.8, max_tokens=MAX_TOKENS, ignore_eos=True), use_tqdm=False)
    assert [len(output["token_ids"]) for output in outputs] == [MAX_TOKENS] * len(prompts)
