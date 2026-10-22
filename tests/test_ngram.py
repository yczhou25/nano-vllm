from nanovllm.engine.ngram import NgramProposer
from nanovllm.engine.sequence import Sequence


def test_proposes_what_followed_the_latest_match():
    seq = Sequence([1, 2, 3, 4, 1, 2, 3, 5, 6, 7, 1, 2, 3])
    # "1 2 3" was followed by 4 the first time and by 5 the second; the latest wins.
    assert NgramProposer(3).propose(seq) == [5, 6, 7]


def test_prefers_longer_ngrams():
    seq = Sequence([9, 3, 4, 8, 3, 5, 9, 3])
    # The 1-gram "3" last occurred before 5, but the 2-gram "9 3" occurred before 4.
    assert NgramProposer(2).propose(seq) == [4, 8]


def test_continues_a_repeating_pattern():
    assert NgramProposer(5).propose(Sequence([7, 1, 2, 1, 2])) == [1, 2, 1, 2, 1]
    assert NgramProposer(3).propose(Sequence([4, 4, 4])) == [4, 4, 4]


def test_without_a_match_repeats_the_last_token():
    assert NgramProposer(2).propose(Sequence([1, 2, 3])) == [3, 3]
    assert NgramProposer(2).propose(Sequence([5])) == [5, 5]


def test_index_grows_with_the_sequence():
    proposer = NgramProposer(2)
    seq = Sequence([1, 2, 3])
    assert proposer.propose(seq) == [3, 3]
    for token_id in [1, 2]:
        seq.append_token(token_id)
    assert proposer.propose(seq) == [3, 1]
    seq.append_token(9)
    assert proposer.propose(seq) == [9, 9]
    proposer.forget(seq)
    assert seq.seq_id not in proposer.index
