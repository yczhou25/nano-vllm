"""Adaptive k (speculative_min_acceptance): the scheduler keeps, for every draft position j = 1..K,
the rate at which position j was accepted over about the last ACCEPTANCE_WINDOW sequence-steps
that proposed it. A decode step proposes the leading positions whose rate reaches the threshold
(none: it runs without drafts), except a probe step with all K positions every PROBE_INTERVAL
steps, which measures the positions left out again. The scheduler is driven directly, with
made-up sampler outputs."""
from engine_sim import make_config
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.sequence import Sequence
from nanovllm.sampling_params import SamplingParams

K = 4


def make_scheduler(num_seqs, min_acceptance=0.5, disable_by=None, window=16, probe=3):
    Sequence.block_size = 4
    scheduler = Scheduler(make_config(num_blocks=256, block_size=4, max_num_seqs=16, max_num_batched_tokens=10 ** 6,
                                      num_speculative_tokens=K, draft_model=True, speculative_disable_by_batch_size=disable_by,
                                      speculative_min_acceptance=min_acceptance))
    scheduler.ACCEPTANCE_WINDOW, scheduler.PROBE_INTERVAL = window, probe
    for _ in range(num_seqs):
        scheduler.add(Sequence([1, 2, 3], SamplingParams(max_tokens=10 ** 6, ignore_eos=True)))
    seqs, is_prefill = scheduler.schedule()
    assert is_prefill and len(seqs) == num_seqs
    scheduler.postprocess(scheduler.commit(seqs, True), [5] * num_seqs)
    return scheduler


def decode(scheduler, accepted):
    """One decode step. accepted: each sequence's accepted drafts, capped at the step's number of
    drafts. Returns that number."""
    seqs, is_prefill = scheduler.schedule()
    assert not is_prefill and seqs
    k = seqs[0].num_scheduled_tokens - 1
    assert all(seq.num_scheduled_tokens == k + 1 for seq in seqs)
    outputs = [[7] * (min(a, k) + 1) if k else 7 for a in accepted]
    scheduler.postprocess(scheduler.commit(seqs, False), outputs)
    return k


def rates(scheduler):
    return [round(a, 3) for a in scheduler.acceptance[1:]]


def test_drafts_are_trusted_until_measured():
    scheduler = make_scheduler(1)
    assert rates(scheduler) == [1.0] * K
    assert decode(scheduler, [0]) == K    # the first step drafts all K positions, and measures 0/1 for each
    assert rates(scheduler) == [0.0] * K
    assert decode(scheduler, [0]) == 0


def test_k_is_cut_behind_the_first_rejected_position():
    scheduler = make_scheduler(1, probe=100)
    assert decode(scheduler, [2]) == K       # positions 1, 2 accepted, 3, 4 rejected
    assert rates(scheduler) == [1.0, 1.0, 0.0, 0.0]
    assert decode(scheduler, [2]) == 2       # only the two trusted positions are proposed and measured
    assert rates(scheduler) == [1.0, 1.0, 0.0, 0.0] and scheduler.num_acceptance_samples == [0, 2, 2, 1, 1]
    assert decode(scheduler, [1]) == 2       # position 2 rejected once: 2/3
    assert rates(scheduler) == [1.0, 0.667, 0.0, 0.0]
    assert decode(scheduler, [1]) == 2       # 2/4 = 0.5, still not below
    assert rates(scheduler) == [1.0, 0.5, 0.0, 0.0]
    assert decode(scheduler, [1]) == 2       # 2/5
    assert rates(scheduler) == [1.0, 0.4, 0.0, 0.0]
    assert decode(scheduler, [1]) == 1
    assert decode(scheduler, [0]) == 1       # position 1: 6/7
    assert rates(scheduler) == [0.857, 0.4, 0.0, 0.0]
    assert scheduler.num_draft_tokens == 4 + 2 + 2 + 2 + 2 + 1 + 1 and scheduler.num_accepted_tokens == 2 + 2 + 1 + 1 + 1 + 1 + 0


def test_a_gap_stops_the_leading_positions():
    scheduler = make_scheduler(1, probe=100)
    scheduler.acceptance[2] = 0.2    # positions 1, 3, 4 trusted, position 2 not
    assert scheduler.num_acceptable_drafts() == 1
    assert decode(scheduler, [4]) == 1


def test_probes_with_all_positions_every_interval():
    scheduler = make_scheduler(1, probe=3)
    ks = [decode(scheduler, [0]) for _ in range(13)]
    assert ks == [K, 0, 0, 0, K, 0, 0, 0, K, 0, 0, 0, K]
    assert scheduler.num_draft_tokens == 4 * K and scheduler.num_accepted_tokens == 0


def test_a_shorter_step_counts_towards_the_probe_and_leaves_later_positions_alone():
    scheduler = make_scheduler(1, probe=5)
    assert decode(scheduler, [1]) == K
    assert rates(scheduler) == [1.0, 0.0, 0.0, 0.0] and scheduler.num_acceptance_samples == [0, 1, 1, 1, 1]
    for _ in range(5):
        assert decode(scheduler, [4]) == 1    # only position 1 is proposed and measured
    assert rates(scheduler) == [1.0, 0.0, 0.0, 0.0] and scheduler.num_acceptance_samples == [0, 6, 1, 1, 1]
    assert scheduler.num_draft_tokens == 4 + 5 and scheduler.num_accepted_tokens == 1 + 5
    assert decode(scheduler, [4]) == K    # the probe measures positions 2-4 again: 1/2 each
    assert rates(scheduler) == [1.0, 0.5, 0.5, 0.5]
    assert decode(scheduler, [4]) == K


def test_a_good_probe_brings_positions_back():
    scheduler = make_scheduler(1, window=4, probe=2)
    assert decode(scheduler, [0]) == K      # 0/1 each
    assert decode(scheduler, [4]) == 0
    assert decode(scheduler, [4]) == 0
    assert decode(scheduler, [4]) == K      # probe: (0*1 + 1) / 2 = 0.5, not below the threshold
    assert rates(scheduler) == [0.5] * K
    assert [decode(scheduler, [4]) for _ in range(3)] == [K, K, K]
    assert rates(scheduler) == [0.8] * K    # 0.5 -> 0.667 -> 0.75 -> 0.8: once 4 samples were seen each step replaces a quarter of the window
    assert decode(scheduler, [0]) == K      # -> 0.64
    assert decode(scheduler, [0]) == K      # -> 0.512
    assert decode(scheduler, [0]) == K      # -> 0.41
    assert decode(scheduler, [0]) == 0


def test_the_window_forgets_old_history():
    scheduler = make_scheduler(2, window=4, probe=1)    # 2 samples per position per step
    for _ in range(20):
        decode(scheduler, [0, 0])
    assert rates(scheduler) == [0.0] * K and scheduler.num_acceptance_samples == [0, 4, 4, 4, 4]
    # With probe=1 every other step is a probe (the 20th step was off, so the next one probes). Two
    # probes of full acceptance lift every position over the window: (0*4 + 2) / 6 = 0.333, then
    # (0.333*4 + 2) / 6 = 0.556.
    ks = [decode(scheduler, [4, 4]) for _ in range(4)]
    assert ks == [K, 0, K, K]
    assert rates(scheduler) == [0.704] * K
    assert [decode(scheduler, [4, 4]) for _ in range(3)] == [K, K, K]


def test_partial_acceptance_counts_each_position_across_the_batch():
    scheduler = make_scheduler(2, window=16)
    assert decode(scheduler, [4, 0]) == K    # every position: 1/2 = 0.5, not below
    assert rates(scheduler) == [0.5] * K
    assert decode(scheduler, [1, 0]) == K    # position 1: 2/4, positions 2-4: 1/4
    assert rates(scheduler) == [0.5, 0.25, 0.25, 0.25]
    assert decode(scheduler, [4, 4]) == 1
    assert rates(scheduler) == [0.667, 0.25, 0.25, 0.25]


def test_batch_size_rule_still_applies():
    scheduler = make_scheduler(2, disable_by=1)
    assert [decode(scheduler, [4, 4]) for _ in range(5)] == [0] * 5    # trusted drafts, but the batch is too large
    assert rates(scheduler) == [1.0] * K


def test_threshold_zero_never_cuts_k():
    scheduler = make_scheduler(1, min_acceptance=0.0)
    assert [decode(scheduler, [0]) for _ in range(5)] == [K] * 5
