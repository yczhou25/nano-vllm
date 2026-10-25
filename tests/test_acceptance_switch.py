"""The acceptance switch (speculative_min_acceptance): drafts stay on until the acceptance rate over
about the last ACCEPTANCE_WINDOW drafts drops below the threshold; then decode steps run without
drafts, with one probe step every PROBE_INTERVAL steps, and drafts come back once a probe lifts the
rate over the threshold again. The scheduler is driven directly, with made-up sampler outputs."""
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
    """One decode step. accepted: each sequence's accepted drafts, used when the step has drafts.
    Returns the step's number of drafts."""
    seqs, is_prefill = scheduler.schedule()
    assert not is_prefill and seqs
    k = seqs[0].num_scheduled_tokens - 1
    assert all(seq.num_scheduled_tokens == k + 1 for seq in seqs)
    outputs = [[7] * (min(a, k) + 1) if k else 7 for a in accepted]
    scheduler.postprocess(scheduler.commit(seqs, False), outputs)
    return k


def test_drafts_are_trusted_until_measured():
    scheduler = make_scheduler(1)
    assert scheduler.acceptance == 1.0
    assert decode(scheduler, [0]) == K    # the first step drafts, and measures 0/4
    assert scheduler.acceptance == 0.0
    assert decode(scheduler, [0]) == 0


def test_switches_off_below_threshold_and_probes_every_interval():
    scheduler = make_scheduler(1, probe=3)
    ks = [decode(scheduler, [0]) for _ in range(13)]
    assert ks == [K, 0, 0, 0, K, 0, 0, 0, K, 0, 0, 0, K]
    assert scheduler.num_draft_tokens == 4 * K and scheduler.num_accepted_tokens == 0


def test_steps_without_drafts_leave_the_statistics_alone():
    scheduler = make_scheduler(1, probe=5)
    decode(scheduler, [1])
    acceptance, drafts, seen = scheduler.acceptance, scheduler.num_draft_tokens, scheduler.num_acceptance_drafts
    assert acceptance == 0.25
    for _ in range(5):
        assert decode(scheduler, [4]) == 0
    assert (scheduler.acceptance, scheduler.num_draft_tokens, scheduler.num_acceptance_drafts) == (acceptance, drafts, seen)


def test_a_good_probe_turns_drafts_back_on():
    scheduler = make_scheduler(1, window=8, probe=2)
    assert decode(scheduler, [0]) == K      # 0/4 -> 0.0
    assert decode(scheduler, [4]) == 0
    assert decode(scheduler, [4]) == 0
    assert decode(scheduler, [4]) == K      # probe: 4/4 -> (0*4 + 4) / 8 = 0.5, not below the threshold
    assert scheduler.acceptance == 0.5
    assert [decode(scheduler, [4]) for _ in range(3)] == [K, K, K]
    assert 0.85 < scheduler.acceptance < 0.86    # 0.5 -> 0.667 -> 0.778 -> 0.852: each step's 4 drafts replace half the window
    assert decode(scheduler, [0]) == K      # -> 0.568
    assert decode(scheduler, [0]) == K      # -> 0.379
    assert decode(scheduler, [0]) == 0


def test_the_window_forgets_old_history():
    scheduler = make_scheduler(2, window=16, probe=1)    # 8 drafts per step
    for _ in range(20):
        decode(scheduler, [0, 0])
    assert scheduler.acceptance == 0.0 and scheduler.num_acceptance_drafts == 16
    # With probe=1 every other step is a probe (the 20th step was off, so the next one probes). Two
    # probes of full acceptance cover the window: (0*16 + 8) / 24 = 0.333, then (0.333*16 + 8) / 24 = 0.556.
    ks = [decode(scheduler, [4, 4]) for _ in range(4)]
    assert ks == [K, 0, K, K]
    assert scheduler.acceptance > 0.5
    assert [decode(scheduler, [4, 4]) for _ in range(3)] == [K, K, K]


def test_partial_acceptance_counts_tokens_across_the_batch():
    scheduler = make_scheduler(2, window=16)
    assert decode(scheduler, [4, 0]) == K    # 4/8 = 0.5: not below
    assert decode(scheduler, [1, 0]) == K    # (0.5*8 + 1) / 16 = 0.3125
    assert decode(scheduler, [4, 4]) == 0


def test_batch_size_rule_still_applies():
    scheduler = make_scheduler(2, disable_by=1)
    assert [decode(scheduler, [4, 4]) for _ in range(5)] == [0] * 5    # trusted drafts, but the batch is too large
    assert scheduler.acceptance == 1.0


def test_threshold_zero_never_switches_off():
    scheduler = make_scheduler(1, min_acceptance=0.0)
    assert [decode(scheduler, [0]) for _ in range(5)] == [K] * 5
