import pytest
import torch

from nanovllm.layers.sampler import (
    RejectionSampler,
    Sampler,
    accepted_token_ids,
    sample_drafts,
)
from nanovllm.sampling_params import SamplingParams

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def test_greedy_rows_take_argmax():
    torch.manual_seed(0)
    logits = torch.randn(64, 1000, device=DEVICE)
    temperatures = torch.tensor([0.0, 1.0] * 32, device=DEVICE)
    tokens = Sampler()(logits, temperatures)
    assert torch.equal(tokens[0::2], logits[0::2].argmax(dim=-1))


@pytest.mark.parametrize("temperature", [0.5, 1.0, 2.0])
def test_sampling_follows_softmax(temperature):
    torch.manual_seed(0)
    n = 20000
    row = torch.tensor([2.0, 1.0, 0.5, 0.0, -1.0], device=DEVICE)
    tokens = Sampler()(row.repeat(n, 1), torch.full((n,), temperature, device=DEVICE))
    freq = torch.bincount(tokens, minlength=row.numel()).float() / n
    expected = torch.softmax(row / temperature, dim=-1)
    assert (freq - expected).abs().max().item() < 0.015


def test_sampling_params_allow_greedy():
    SamplingParams(temperature=0)
    with pytest.raises(AssertionError):
        SamplingParams(temperature=-1)


def test_rejection_sampler_greedy_keeps_drafts_that_match_argmax():
    torch.manual_seed(0)
    k, vocab = 3, 50
    logits = torch.randn(4, k + 1, vocab, device=DEVICE)
    argmax = logits.argmax(dim=-1)
    drafts = argmax[:, :k].clone()
    drafts[1, 0] = (argmax[1, 0] + 1) % vocab    # rejected at the first draft
    drafts[2, 2] = (argmax[2, 2] + 1) % vocab    # rejected at the last draft
    drafts[3, 1] = (argmax[3, 1] + 1) % vocab    # rejected in the middle; the draft after it is ignored
    out = RejectionSampler()(logits.view(-1, vocab), torch.zeros(4, device=DEVICE), drafts)
    assert accepted_token_ids(out.tolist()) == [
        argmax[0].tolist(),
        argmax[1, :1].tolist(),
        argmax[2, :3].tolist(),
        argmax[3, :2].tolist(),
    ]


@pytest.mark.parametrize("temperature", [0.7, 1.0])
def test_rejection_sampler_follows_target_distribution(temperature):
    # Each emitted token must follow the target distribution at its position, given the tokens
    # before it, whatever the drafts were.
    torch.manual_seed(0)
    n = 100000
    position_logits = torch.tensor([[2.0, 1.5, 0.5, 0.0, -1.0, -2.0],
                                    [0.0, 1.0, 2.0, 0.5, -0.5, 1.5],
                                    [1.0, 1.0, 1.0, 0.0, 0.0, 3.0]], device=DEVICE)
    probs = torch.softmax(position_logits / temperature, dim=-1)
    drafts = torch.tensor([[0, 5]], device=DEVICE).repeat(n, 1)   # the likeliest token, then an unlikely one
    out = RejectionSampler()(position_logits.repeat(n, 1), torch.full((n,), temperature, device=DEVICE), drafts)

    def freq(tokens):
        return torch.bincount(tokens, minlength=6).float() / tokens.numel()

    assert (freq(out[:, 0]) - probs[0]).abs().max().item() < 0.01
    first_accepted = out[:, 0] == 0
    assert (out[~first_accepted, 1:] == -1).all()
    assert (freq(out[first_accepted, 1]) - probs[1]).abs().max().item() < 0.015
    both_accepted = first_accepted & (out[:, 1] == 5)
    assert (out[first_accepted & ~both_accepted, 2] == -1).all()
    assert both_accepted.sum().item() > 1000
    assert (freq(out[both_accepted, 2]) - probs[2]).abs().max().item() < 0.02


def test_sample_drafts_follow_the_draft_distribution_and_keep_it():
    # A draft model's proposals: greedy rows take the argmax, the others a sample from
    # softmax(logits / T), and the distribution comes back for the rejection sampler.
    torch.manual_seed(0)
    n = 20000
    row = torch.tensor([2.0, 1.0, 0.5, 0.0, -1.0], device=DEVICE)
    temperatures = torch.full((n,), 0.7, device=DEVICE)
    temperatures[0::2] = 0
    tokens, probs = sample_drafts(row.repeat(n, 1), temperatures)
    assert (tokens[0::2] == 0).all()
    expected = torch.softmax(row / 0.7, dim=-1)
    freq = torch.bincount(tokens[1::2], minlength=5).float() / (n // 2)
    assert (freq - expected).abs().max().item() < 0.015
    assert torch.allclose(probs[1::2], expected.expand(n // 2, -1), atol=1e-6)
    assert (probs[0::2].argmax(dim=-1) == 0).all()


TARGET_LOGITS = [[2.0, 1.5, 0.5, 0.0, -1.0, -2.0],
                 [0.0, 1.0, 2.0, 0.5, -0.5, 1.5],
                 [1.0, 1.0, 1.0, 0.0, 0.0, 3.0]]
DRAFT_LOGITS = [[1.0, 2.0, 0.0, 0.5, -1.0, 0.0],
                [2.0, 0.0, 1.0, 0.5, 0.5, -1.0],
                [0.0, 0.0, 2.0, 1.0, 0.0, 1.0]]


def check_speculative_sampling(out, drafts, p, q, tolerance=0.012):
    """out (n, k + 1): every emitted token follows the target distribution at its position; draft
    j is accepted (kept at out[:, j]) with probability sum(min(p_j, q_j)); the token after a
    rejected draft is never that draft (the residual gives it probability 0)."""
    n, k = drafts.shape

    def freq(tokens):
        return torch.bincount(tokens, minlength=p.size(1)).float() / tokens.numel()

    present = torch.ones(n, dtype=torch.bool, device=out.device)
    for j in range(k + 1):
        tokens = out[present, j]
        assert (tokens >= 0).all() and (freq(tokens) - p[j]).abs().max().item() < tolerance, f"position {j}"
        if j < k:
            accepted = present & (out[:, j] == drafts[:, j])
            rate = accepted.sum().item() / present.sum().item()
            assert abs(rate - torch.minimum(p[j], q[j]).sum().item()) < tolerance, f"acceptance at {j}: {rate}"
            assert (out[present & ~accepted, j + 1] == -1).all()
            present = accepted
    assert present.sum().item() > 1000    # enough samples reached the last position


@pytest.mark.parametrize("temperature", [0.7, 1.0, 1.5])
def test_sampled_drafts_give_the_target_distribution(temperature):
    # Exact speculative sampling (Leviathan et al.): drafts sampled from q, accepted with
    # min(1, p/q), the residual norm(max(0, p - q)) after a rejection. The output is distributed as
    # the target's own sampling whatever q is, and the acceptance rate is sum(min(p, q)).
    torch.manual_seed(0)
    n = 100000
    target, draft = torch.tensor(TARGET_LOGITS, device=DEVICE), torch.tensor(DRAFT_LOGITS, device=DEVICE)
    p, q = torch.softmax(target / temperature, dim=-1), torch.softmax(draft / temperature, dim=-1)
    drafts = torch.stack([torch.multinomial(q[j], n, replacement=True) for j in range(2)], dim=1)    # two drafts, three positions
    draft_probs = q[:2].unsqueeze(0).expand(n, -1, -1).contiguous()
    out = RejectionSampler()(target.repeat(n, 1), torch.full((n,), temperature, device=DEVICE), drafts, draft_probs=draft_probs)
    check_speculative_sampling(out, drafts, p, q)


def test_one_hot_draft_probs_match_the_one_hot_path():
    # A draft with q one-hot at it (argmax drafts, n-gram) goes through the same arithmetic either
    # way: accepted with p(d), the residual is p with d removed.
    torch.manual_seed(0)
    n, vocab = 20000, 6
    target = torch.tensor(TARGET_LOGITS, device=DEVICE)
    drafts = torch.tensor([[0, 5]], device=DEVICE).repeat(n, 1)
    draft_probs = torch.nn.functional.one_hot(drafts, vocab).float()
    temperatures = torch.full((n,), 1.0, device=DEVICE)
    torch.manual_seed(1)
    without = RejectionSampler()(target.repeat(n, 1), temperatures, drafts)
    torch.manual_seed(1)
    with_q = RejectionSampler()(target.repeat(n, 1), temperatures, drafts, draft_probs=draft_probs)
    accepted = (without != -1).sum(dim=1)
    assert torch.equal(accepted, (with_q != -1).sum(dim=1))    # the same drafts accepted (the same random draws)
    assert (without == with_q).all(dim=1).float().mean().item() > 0.999    # the residual sample too, up to rounding


def test_greedy_rows_ignore_draft_probs():
    torch.manual_seed(0)
    logits = torch.randn(4 * 3, 50, device=DEVICE)
    drafts = torch.randint(0, 50, (4, 2), device=DEVICE)
    drafts[0] = logits.view(4, 3, -1)[0, :2].argmax(dim=-1)    # all accepted
    draft_probs = torch.softmax(torch.randn(4, 2, 50, device=DEVICE), dim=-1)
    temperatures = torch.zeros(4, device=DEVICE)
    assert torch.equal(RejectionSampler()(logits, temperatures, drafts), RejectionSampler()(logits, temperatures, drafts, draft_probs=draft_probs))
