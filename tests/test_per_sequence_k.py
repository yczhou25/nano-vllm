"""Per-sequence k (speculative_per_sequence): every sequence keeps its own acceptance rate per
draft position, over its last ACCEPTANCE_WINDOW_PER_SEQUENCE steps that proposed it (starting
from ACCEPTANCE_PRIOR_PER_SEQUENCE accepted samples), and proposes the leading positions whose rate
reaches the threshold, all of them on its first step and on the batch's probe steps. Sequences in one batch can have different numbers
of drafts (a ragged batch). The scheduler is driven directly, with made-up sampler outputs."""
from engine_sim import make_config
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.sequence import Sequence
from nanovllm.sampling_params import SamplingParams

K = 4


def make_scheduler(num_seqs, min_acceptance=0.5, disable_by=None, window=4, probe=3, draft_model=True):
    Sequence.block_size = 4
    scheduler = Scheduler(make_config(num_blocks=256, block_size=4, max_num_seqs=16, max_num_batched_tokens=10 ** 6,
                                      num_speculative_tokens=K, draft_model=draft_model, speculative_disable_by_batch_size=disable_by,
                                      speculative_min_acceptance=min_acceptance, speculative_per_sequence=True))
    scheduler.ACCEPTANCE_WINDOW_PER_SEQUENCE, scheduler.PROBE_INTERVAL = window, probe
    for _ in range(num_seqs):
        scheduler.add(Sequence([1, 2, 3], SamplingParams(max_tokens=10 ** 6, ignore_eos=True)))
    seqs, is_prefill = scheduler.schedule()
    assert is_prefill and len(seqs) == num_seqs
    scheduler.postprocess(scheduler.commit(seqs, True), [5] * num_seqs)
    return scheduler


def decode(scheduler, accepted):
    """One decode step. accepted: each sequence's accepted drafts, capped at its number of drafts.
    Returns the number of drafts per sequence."""
    seqs, is_prefill = scheduler.schedule()
    assert not is_prefill and seqs
    ks = [seq.num_scheduled_tokens - 1 for seq in seqs]
    outputs = [[7] * (min(a, k) + 1) if max(ks) else 7 for a, k in zip(accepted, ks)]
    scheduler.postprocess(scheduler.commit(seqs, False), outputs)
    return ks


def rates(seq):
    return [round(a, 3) for a in seq.acceptance[1:]]


def test_each_sequence_gets_its_own_k():
    scheduler = make_scheduler(2, probe=100)
    assert decode(scheduler, [4, 0]) == [K, K]    # first step: every position, trusted until measured
    good, bad = scheduler.running
    assert rates(good) == [1.0] * K and rates(bad) == [0.5] * K    # (1 prior + 0) / 2: one rejection does not strip a sequence
    assert decode(scheduler, [4, 0]) == [K, K]
    assert rates(bad) == [0.333] * K
    assert decode(scheduler, [4, 0]) == [K, 0]    # a ragged batch
    assert decode(scheduler, [1, 0]) == [K, 0]    # good: positions 2-4 accepted 4/5 (window 4 full, the step weighs 1/5)
    assert rates(good) == [1.0, 0.8, 0.8, 0.8]
    assert [decode(scheduler, [1, 0]) for _ in range(3)] == [[K, 0]] * 3    # 0.64, 0.512, 0.41
    assert decode(scheduler, [1, 0]) == [1, 0]
    assert rates(good) == [1.0, 0.41, 0.41, 0.41]    # positions it did not propose keep their rate
    assert scheduler.num_ragged_steps == 6 and scheduler.num_decode_steps == 8


def test_the_batch_rates_do_not_leak_between_sequences():
    scheduler = make_scheduler(3, probe=100)
    decode(scheduler, [4, 4, 0])
    assert [rates(seq) for seq in scheduler.running] == [[1.0] * K, [1.0] * K, [0.5] * K]
    assert scheduler.acceptance[1:] == [2 / 3] * K    # the batch-wide rates are kept for the statistics, without the prior
    assert decode(scheduler, [4, 4, 0]) == [K, K, K]
    assert decode(scheduler, [4, 4, 4]) == [K, K, 0]


def test_a_new_sequence_starts_with_every_position():
    scheduler = make_scheduler(1, probe=100)
    assert decode(scheduler, [0]) == [K]
    assert decode(scheduler, [0]) == [K]    # 1/2
    assert decode(scheduler, [0]) == [0]    # 1/3
    scheduler.add(Sequence([1, 2, 3], SamplingParams(max_tokens=10 ** 6, ignore_eos=True)))
    seqs, is_prefill = scheduler.schedule()
    assert is_prefill and len(seqs) == 1
    scheduler.postprocess(scheduler.commit(seqs, True), [5])
    assert decode(scheduler, [0, 4]) == [0, K]


def test_probes_are_the_batchs_one_uniform_step():
    # The probe counter is the batch's: it counts the steps in which some sequence proposed fewer
    # than K, and the probe gives every sequence K in the same step (a uniform step, which keeps the
    # CUDA graphs), rather than each sequence probing at its own time in a ragged step.
    scheduler = make_scheduler(2, probe=3)
    assert decode(scheduler, [0, 4]) == [K, K]
    assert decode(scheduler, [0, 4]) == [K, K]    # the first sequence at 1/2
    assert [decode(scheduler, [0, 4]) for _ in range(6)] == [[0, K], [0, K], [0, K], [K, K], [0, K], [0, K]]
    assert scheduler.num_steps_by_k == [5, 0, 0, 0, 11]    # sequence-steps in this mode


def test_a_shorter_step_counts_towards_the_probe_and_leaves_later_positions_alone():
    scheduler = make_scheduler(1, probe=5)
    assert decode(scheduler, [1]) == [K]
    seq, = scheduler.running
    assert rates(seq) == [1.0, 0.5, 0.5, 0.5] and seq.num_acceptance_samples == [0, 2, 2, 2, 2]    # the prior and one sample
    assert decode(scheduler, [1]) == [K]
    assert rates(seq) == [1.0, 0.333, 0.333, 0.333] and seq.num_acceptance_samples == [0, 3, 3, 3, 3]
    for _ in range(5):
        assert decode(scheduler, [4]) == [1]
    assert rates(seq) == [1.0, 0.333, 0.333, 0.333] and seq.num_acceptance_samples == [0, 4, 3, 3, 3]    # window 4
    assert decode(scheduler, [4]) == [K]    # the probe: (1/3 * 3 + 1) / 4
    assert rates(seq) == [1.0, 0.5, 0.5, 0.5]


def test_the_window_forgets_old_history():
    scheduler = make_scheduler(1, window=4, probe=100)
    seq, = scheduler.running
    for _ in range(10):
        decode(scheduler, [4])
    assert rates(seq) == [1.0] * K and seq.num_acceptance_samples == [0, 4, 4, 4, 4]
    ks = [decode(scheduler, [0])[0] for _ in range(5)]
    assert ks == [K, K, K, K, 0]    # 0.8, 0.64, 0.512, 0.41: a full window weighs a new step at 1/5
    assert rates(seq) == [0.41] * K    # the step without drafts measured nothing


def test_batch_size_rule_still_applies_to_every_sequence():
    scheduler = make_scheduler(2, disable_by=1)
    assert [decode(scheduler, [4, 4]) for _ in range(3)] == [[0, 0]] * 3
    assert scheduler.num_capped_steps == 3 and scheduler.num_ragged_steps == 0
    for seq in scheduler.running:
        assert rates(seq) == [1.0] * K    # nothing proposed, nothing measured
    assert scheduler.num_steps_without_probe == 3    # capped steps count towards the probe, as in the batch-wide mode


def test_statistics_count_sequence_steps_and_positions():
    scheduler = make_scheduler(2, probe=100)
    assert decode(scheduler, [4, 1]) == [K, K]
    assert decode(scheduler, [4, 1]) == [K, K]
    assert decode(scheduler, [4, 1]) == [K, 1]
    assert scheduler.num_steps_by_k == [0, 1, 0, 0, 5]
    assert scheduler.num_proposed_by_position == [0, 6, 5, 5, 5] and scheduler.num_accepted_by_position == [0, 6, 3, 3, 3]
    assert (scheduler.num_draft_tokens, scheduler.num_accepted_tokens) == (21, 15)
    scheduler.reset_stats()
    assert scheduler.num_steps_by_k == [0] * 5 and scheduler.num_decode_steps == scheduler.num_ragged_steps == 0


def test_ngram_drafts_are_cut_per_sequence():
    scheduler = make_scheduler(2, probe=100, draft_model=False)
    assert decode(scheduler, [4, 0]) == [K, K]
    assert decode(scheduler, [4, 0]) == [K, K]
    seqs, _ = scheduler.schedule()
    assert [len(seq.draft_token_ids) for seq in seqs] == [K, 0] and [seq.num_scheduled_tokens for seq in seqs] == [K + 1, 1]
