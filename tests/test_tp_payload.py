"""Ranks > 0 get each step's sequences from rank 0 through pickle over a 1 MiB shared memory
segment (ModelRunner.write_shm/read_shm) and must build the same inputs as rank 0."""
import pickle
import random

import pytest

from engine_sim import make_config, make_runner, prepare
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.sequence import Sequence
from nanovllm.sampling_params import SamplingParams

SHM_SIZE = 2**20


@pytest.mark.parametrize("num_tokens,block_table", [(300, [0]), (512, [5]), (1000, [3, 7]), (1025, [1, 2, 4])])
def test_decode_slot_mapping_on_worker_rank(num_tokens, block_table):
    block_size = 512
    Sequence.block_size = block_size        # what LLMEngine sets in the main process
    seq = Sequence(list(range(num_tokens)))
    seq.block_table = block_table
    seq.is_prefill = False
    runner = make_runner(block_size)
    rank0 = prepare(runner, [seq], False)
    assert rank0["slot_mapping"] == [block_table[-1] * block_size + (num_tokens - 1) % block_size]
    payload = pickle.dumps([seq])
    Sequence.block_size = 256               # ranks > 0 are spawned and keep the class default
    assert prepare(runner, pickle.loads(payload), False) == rank0


def test_prefill_payload_fits_shm_with_prefix_cache_hits():
    # 128 requests that share a 3840-token system prompt. After the first step, each prefill only
    # schedules its last 64 tokens, but the payload used to carry every full prompt (about 2 MB).
    block_size = 256
    Sequence.block_size = block_size
    rng = random.Random(0)
    scheduler = Scheduler(make_config(4000, block_size, max_num_seqs=512, max_num_batched_tokens=16384, eos=-1))
    system = [rng.randrange(151936) for _ in range(3840)]
    for _ in range(128):
        scheduler.add(Sequence(system + [rng.randrange(151936) for _ in range(64)], SamplingParams(max_tokens=16)))
    runner = make_runner(block_size)
    for _ in range(3):
        seqs, is_prefill = scheduler.schedule()
        payload = pickle.dumps(["run", seqs, is_prefill])
        assert len(payload) + 4 <= SHM_SIZE
        assert prepare(runner, pickle.loads(payload)[1], is_prefill) == prepare(runner, seqs, is_prefill)
        scheduler.postprocess(seqs, [1] * len(seqs), is_prefill)
