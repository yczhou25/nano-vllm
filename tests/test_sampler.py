import pytest
import torch

from nanovllm.layers.sampler import Sampler
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
