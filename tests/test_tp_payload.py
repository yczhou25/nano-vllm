"""Ranks > 0 get each step's sequences from rank 0 through pickle over a 1 MiB shared memory
segment (ModelRunner.write_shm/read_shm) and must build the same inputs as rank 0."""
import multiprocessing as mp
import os
import pickle
import random
import time
from multiprocessing.shared_memory import SharedMemory

import pytest

import cpu_shims  # noqa: F401  (before nanovllm, also in the spawned worker)
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
        scheduler.postprocess(scheduler.commit(seqs, is_prefill), [1] * len(seqs))


def _slow_worker(name, event, queue, n):
    runner = make_runner(256, world_size=2, rank=1)
    runner.shm = SharedMemory(name=name)
    runner.event = event
    for _ in range(n):
        time.sleep(0.01)    # falls behind rank 0
        queue.put(runner.read_shm())
    runner.shm.close()


def test_write_shm_waits_until_workers_read():
    # With async scheduling rank 0 no longer waits for the GPU (and so for the workers) between
    # calls, so write_shm must not overwrite a message that a worker has not read yet.
    ctx = mp.get_context("spawn")
    name = f"nanovllm_test_{os.getpid()}"
    shm = SharedMemory(name=name, create=True, size=SHM_SIZE)
    event, queue = ctx.Event(), ctx.Queue()
    worker = ctx.Process(target=_slow_worker, args=(name, event, queue, 20))
    worker.start()
    try:
        runner = make_runner(256, world_size=2, rank=0)
        runner.shm, runner.event = shm, [event]
        for i in range(20):
            runner.write_shm("run", i)
        assert [queue.get(timeout=10) for _ in range(20)] == [("run", [i]) for i in range(20)]
    finally:
        worker.join(timeout=10)
        worker.kill()
        shm.close()
        shm.unlink()
