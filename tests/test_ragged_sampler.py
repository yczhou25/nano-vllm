"""RejectionSampler on a ragged batch (sequences with different numbers of drafts, see
ModelRunner.ragged_layout): the padded positions count as rejected, and every sequence gets what
it would get in a batch of its own."""
import pytest
import torch

from nanovllm.engine.model_runner import ModelRunner
from nanovllm.layers.sampler import RejectionSampler, accepted_token_ids

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def test_padded_ragged_layout():
    # Every sequence gets 1 + max drafts rows; its own are the last ones, behind dummy rows, and
    # the drafts it does not have are written to its dummy rows.
    token_index, write_index, num_drafts = ModelRunner.ragged_layout([2, 0, 1], pad=True)
    assert token_index.tolist() == [[0, 1, 2], [5, 5, 5], [7, 8, 8]]
    assert write_index.tolist() == [[1, 2], [3, 4], [8, 6]]
    assert num_drafts.tolist() == [2, 0, 1]
    token_index, write_index, _ = ModelRunner.ragged_layout([2, 2], pad=True)    # uniform: the plain grid
    assert token_index.tolist() == [[0, 1, 2], [3, 4, 5]] and write_index.tolist() == [[1, 2], [4, 5]]


def test_ragged_layout():
    token_index, write_index, num_drafts = ModelRunner.ragged_layout([2, 0, 1])
    assert token_index.tolist() == [[0, 1, 2], [3, 3, 3], [4, 5, 5]]    # padded by repeating the last row
    assert write_index.tolist() == [[1, 2], [6, 6], [5, 6]]            # 6: the scratch row after the 6 real ones
    assert num_drafts.tolist() == [2, 0, 1]
    token_index, write_index, num_drafts = ModelRunner.ragged_layout([3, 3])
    assert token_index.tolist() == [[0, 1, 2, 3], [4, 5, 6, 7]] and write_index.tolist() == [[1, 2, 3], [5, 6, 7]]


def ragged_call(logits, temperatures, input_ids, num_drafts):
    token_index, _, num_drafts = ModelRunner.ragged_layout(num_drafts)
    token_index, num_drafts = token_index.to(DEVICE), num_drafts.to(DEVICE)
    return RejectionSampler()(logits, temperatures, input_ids[token_index[:, 1:]], token_index, num_drafts)


def test_greedy_rows_match_the_uniform_sampler_sequence_by_sequence():
    torch.manual_seed(0)
    vocab, num_drafts = 50, [4, 0, 2, 1, 4, 3]
    n = sum(num_drafts) + len(num_drafts)
    logits = torch.randn(n, vocab, device=DEVICE)
    argmax = logits.argmax(dim=-1)
    input_ids = torch.randint(0, vocab, (n,), device=DEVICE)
    token_index, _, _ = ModelRunner.ragged_layout(num_drafts)
    for i, k in enumerate(num_drafts):    # drafts that match the argmax before them, with two rejections
        for j in range(k):
            input_ids[token_index[i, j + 1]] = argmax[token_index[i, j]]
    input_ids[token_index[0, 2]] = (argmax[token_index[0, 1]] + 1) % vocab    # sequence 0: rejected at its second draft
    input_ids[token_index[4, 4]] = (argmax[token_index[4, 3]] + 1) % vocab    # sequence 4: rejected at its last draft
    out = accepted_token_ids(ragged_call(logits, torch.zeros(len(num_drafts), device=DEVICE), input_ids, num_drafts).tolist())
    assert [len(row) for row in out] == [2, 1, 3, 2, 4, 4]
    for i, k in enumerate(num_drafts):    # each sequence as a batch of one, through the uniform path
        rows = token_index[i, :k + 1]
        alone = RejectionSampler()(logits[rows], torch.zeros(1, device=DEVICE), input_ids[rows[1:]].unsqueeze(0)) if k else logits[rows].argmax(dim=-1)
        assert out[i] == (accepted_token_ids(alone.tolist())[0] if k else alone.tolist())


def test_a_uniform_batch_gives_the_same_result_either_way():
    torch.manual_seed(1)
    vocab = 30
    logits = torch.randn(9, vocab, device=DEVICE)
    argmax = logits.argmax(dim=-1)
    input_ids = torch.zeros(9, dtype=torch.int64, device=DEVICE)
    for i in range(3):
        for j in range(2):
            input_ids[i * 3 + j + 1] = argmax[i * 3 + j]
    input_ids[4] = (argmax[3] + 1) % vocab
    uniform = RejectionSampler()(logits, torch.zeros(3, device=DEVICE), input_ids.view(3, 3)[:, 1:])
    assert torch.equal(uniform, ragged_call(logits, torch.zeros(3, device=DEVICE), input_ids, [2, 2, 2]))


@pytest.mark.parametrize("temperature", [0.7, 1.0])
def test_sampled_rows_follow_the_target_distribution(temperature):
    # Two sequences per batch: one with two drafts (the likeliest token, then an unlikely one), one
    # without any. Each emitted token must follow the target distribution at its position.
    torch.manual_seed(0)
    n = 50000
    position_logits = torch.tensor([[2.0, 1.5, 0.5, 0.0, -1.0, -2.0],
                                    [0.0, 1.0, 2.0, 0.5, -0.5, 1.5],
                                    [1.0, 1.0, 1.0, 0.0, 0.0, 3.0],
                                    [3.0, 0.0, 0.0, 0.0, 0.0, 0.0]], device=DEVICE)
    probs = torch.softmax(position_logits / temperature, dim=-1)
    logits = position_logits.repeat(n, 1)
    input_ids = torch.tensor([3, 0, 5, 3], device=DEVICE).repeat(n)    # input token, draft 0, draft 1, the other sequence's input token (a padded draft, masked)
    token_index, _, num_drafts = ModelRunner.ragged_layout([2, 0] * n)
    out = RejectionSampler()(logits, torch.full((2 * n,), temperature, device=DEVICE), input_ids[token_index[:, 1:].to(DEVICE)], token_index.to(DEVICE), num_drafts.to(DEVICE))
    with_drafts, without = out[0::2], out[1::2]

    def freq(tokens):
        return torch.bincount(tokens, minlength=6).float() / tokens.numel()

    assert (without[:, 1:] == -1).all() and (freq(without[:, 0]) - probs[3]).abs().max().item() < 0.01
    assert (freq(with_drafts[:, 0]) - probs[0]).abs().max().item() < 0.01
    first_accepted = with_drafts[:, 0] == 0
    assert (with_drafts[~first_accepted, 1:] == -1).all()
    assert (freq(with_drafts[first_accepted, 1]) - probs[1]).abs().max().item() < 0.015
    both_accepted = first_accepted & (with_drafts[:, 1] == 5)
    assert both_accepted.sum().item() > 500
    assert (freq(with_drafts[both_accepted, 2]) - probs[2]).abs().max().item() < 0.03
