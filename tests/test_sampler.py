import pytest
import torch

from nanovllm.layers.sampler import RejectionSampler, Sampler, accepted_token_ids
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
