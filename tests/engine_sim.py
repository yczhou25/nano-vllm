"""Randomized engine simulation: the real Scheduler, BlockManager, Sequence and
ModelRunner.prepare_prefill/prepare_decode, driven by a fake model instead of a GPU.

For every position p of a sequence, the fake model writes prefix_hash(tokens[:p+1]) into a paged
KV cache at the slot that slot_mapping gives, then "attends" by reading every context position
through block_tables and checks each value it reads. The next token is a function of the prefix
hash, so each request has a reference output that does not depend on batching, chunked prefill,
prefix caching or preemption, and the engine must reproduce it exactly.
"""
import hashlib
import pickle
import random
import types

from nanovllm.engine.model_runner import ModelRunner
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.sequence import Sequence
from nanovllm.sampling_params import SamplingParams
from nanovllm.utils.context import get_context, reset_context

VOCAB = 40
EOS = 0


class Violation(AssertionError):
    pass


def make_config(num_blocks, block_size, max_num_seqs, max_num_batched_tokens, eos=EOS):
    # Config itself requires a model directory and block sizes that are multiples of 256.
    return types.SimpleNamespace(num_kvcache_blocks=num_blocks, kvcache_block_size=block_size,
                                 max_num_seqs=max_num_seqs, max_num_batched_tokens=max_num_batched_tokens, eos=eos)


def make_runner(block_size, world_size=1, rank=0):
    # Only the attributes that prepare_prefill/prepare_decode read; no model or KV cache.
    runner = ModelRunner.__new__(ModelRunner)
    runner.block_size = block_size
    runner.world_size = world_size
    runner.rank = rank
    runner.enforce_eager = True
    return runner


def prefix_hashes(token_ids):
    out, h = [], 12345
    for t in token_ids:
        h = int.from_bytes(hashlib.blake2b(f"{h}:{t}".encode(), digest_size=8).digest(), "little")
        out.append(h)
    return out


def next_token(h):
    return (h >> 7) % VOCAB


def reference(prompt, sp: SamplingParams):
    token_ids = list(prompt)
    out = []
    while True:
        t = next_token(prefix_hashes(token_ids)[-1])
        token_ids.append(t)
        out.append(t)
        if (not sp.ignore_eos and t == EOS) or len(out) == sp.max_tokens:
            return out


def prepare(runner, seqs, is_prefill):
    input_ids, positions = runner.prepare_prefill(seqs) if is_prefill else runner.prepare_decode(seqs)
    c = get_context()
    meta = dict(
        input_ids=input_ids.tolist(),
        positions=positions.tolist(),
        slot_mapping=c.slot_mapping.tolist(),
        cu_seqlens_q=c.cu_seqlens_q.tolist() if c.cu_seqlens_q is not None else None,
        cu_seqlens_k=c.cu_seqlens_k.tolist() if c.cu_seqlens_k is not None else None,
        max_seqlen_q=c.max_seqlen_q,
        max_seqlen_k=c.max_seqlen_k,
        context_lens=c.context_lens.tolist() if c.context_lens is not None else None,
        block_tables=c.block_tables.tolist() if c.block_tables is not None else None,
    )
    reset_context()
    return meta


class FakeModel:

    def __init__(self, scheduler: Scheduler, block_size: int, num_blocks: int, check_tp: bool = True):
        self.block_size = block_size
        self.kv = [None] * (num_blocks * block_size)
        self.runner = make_runner(block_size)
        self.block_manager = scheduler.block_manager
        self.check_tp = check_tp

    def run(self, seqs, is_prefill):
        meta = prepare(self.runner, seqs, is_prefill)
        if self.check_tp:
            # Ranks > 0 receive the sequences through pickle (write_shm/read_shm) and must build
            # exactly the same inputs as rank 0.
            remote_meta = prepare(self.runner, pickle.loads(pickle.dumps(seqs)), is_prefill)
            if remote_meta != meta:
                raise Violation(f"rank > 0 inputs differ from rank 0: {meta} vs {remote_meta}")
        ids, pos, slots = meta["input_ids"], meta["positions"], meta["slot_mapping"]
        if not len(ids) == len(pos) == len(slots):
            raise Violation(f"length mismatch: ids={len(ids)} positions={len(pos)} slots={len(slots)}")
        return self._prefill(seqs, meta) if is_prefill else self._decode(seqs, meta)

    def _prefill(self, seqs, meta):
        bs = self.block_size
        ids, pos, slots = meta["input_ids"], meta["positions"], meta["slot_mapping"]
        cu_q, cu_k, block_tables = meta["cu_seqlens_q"], meta["cu_seqlens_k"], meta["block_tables"]
        hidden = [None] * len(ids)
        for i, seq in enumerate(seqs):
            q0, q1 = cu_q[i], cu_q[i + 1]
            k_len = cu_k[i + 1] - cu_k[i]
            start = seq.num_cached_tokens
            truth = prefix_hashes(seq.token_ids[:k_len])
            if pos[q0:q1] != list(range(start, start + seq.num_scheduled_tokens)):
                raise Violation(f"seq {seq.seq_id}: positions {pos[q0:q1]}, expected to start at {start}")
            if k_len != start + seq.num_scheduled_tokens:
                raise Violation(f"seq {seq.seq_id}: seqlen_k {k_len} != {start} + {seq.num_scheduled_tokens}")
            local = {}
            for j in range(q0, q1):
                p = pos[j]
                if ids[j] != seq.token_ids[p]:
                    raise Violation(f"seq {seq.seq_id}: wrong input id at position {p}")
                self._write(seq, p, slots[j], truth[p])
                local[p] = hidden[j] = truth[p]
            for p in range(k_len):
                if block_tables is not None:
                    value = self.kv[block_tables[i][p // bs] * bs + p % bs]
                elif p in local:
                    value = local[p]
                else:
                    raise Violation(f"seq {seq.seq_id}: needs cached position {p} but got no block tables")
                if value != truth[p]:
                    raise Violation(f"seq {seq.seq_id}: prefill read wrong KV at position {p}")
        # compute_logits takes the hidden state at cu_seqlens_q[1:] - 1.
        return [next_token(hidden[cu_q[i + 1] - 1]) for i in range(len(seqs))]

    def _decode(self, seqs, meta):
        bs = self.block_size
        ids, pos, slots = meta["input_ids"], meta["positions"], meta["slot_mapping"]
        context_lens, block_tables = meta["context_lens"], meta["block_tables"]
        truths = [prefix_hashes(seq.token_ids) for seq in seqs]
        for i, seq in enumerate(seqs):
            p = pos[i]
            if p != len(seq) - 1 or ids[i] != seq.token_ids[-1] or context_lens[i] != len(seq):
                raise Violation(f"seq {seq.seq_id}: wrong decode inputs pos={p} len={len(seq)} context_len={context_lens[i]}")
            self._write(seq, p, slots[i], truths[i][p])
        for i, seq in enumerate(seqs):
            for p in range(context_lens[i]):
                if self.kv[block_tables[i][p // bs] * bs + p % bs] != truths[i][p]:
                    raise Violation(f"seq {seq.seq_id}: decode read wrong KV at position {p} (len {len(seq)})")
        return [next_token(truths[i][-1]) for i in range(len(seqs))]

    def _write(self, seq, p, slot, value):
        bs = self.block_size
        block_id, offset = divmod(slot, bs)
        if p // bs >= len(seq.block_table) or seq.block_table[p // bs] != block_id or offset != p % bs:
            raise Violation(f"seq {seq.seq_id}: position {p} written to slot {slot} outside its blocks {seq.block_table}")
        block = self.block_manager.blocks[block_id]
        if block.hash != -1:
            raise Violation(f"seq {seq.seq_id}: position {p} written into hashed block {block_id}")
        if block.ref_count != 1:
            raise Violation(f"seq {seq.seq_id}: position {p} written into shared block {block_id} (ref_count {block.ref_count})")
        self.kv[slot] = value


def check_block_manager(block_manager, live_seqs):
    n = len(block_manager.blocks)
    refs = [0] * n
    for seq in live_seqs:
        for block_id in seq.block_table:
            refs[block_id] += 1
    for block_id in range(n):
        if block_manager.blocks[block_id].ref_count != refs[block_id]:
            raise Violation(f"block {block_id}: ref_count {block_manager.blocks[block_id].ref_count} != {refs[block_id]} live references")
    free = list(block_manager.free_block_ids)
    used = block_manager.used_block_ids
    if len(set(free)) != len(free):
        raise Violation("duplicate block in the free list")
    if set(free) & used or set(free) | used != set(range(n)):
        raise Violation("free and used blocks do not partition the pool")
    if any(refs[block_id] for block_id in free):
        raise Violation("a free block is still referenced")
    for h, block_id in block_manager.hash_to_block_id.items():
        if block_manager.blocks[block_id].hash != h:
            raise Violation(f"hash_to_block_id maps {h} to block {block_id} with hash {block_manager.blocks[block_id].hash}")


def run_engine(scheduler, model, requests, max_steps=50000):
    """Feed (arrival_step, prompt, sampling_params) requests to the engine, check the block manager
    after every step, and return {request index: completion token ids}."""
    pending = sorted(range(len(requests)), key=lambda i: requests[i][0])
    seqs = {}
    outputs = {}
    step = 0
    while True:
        while pending and requests[pending[0]][0] <= step:
            i = pending.pop(0)
            seq = Sequence(requests[i][1], requests[i][2])
            seqs[seq.seq_id] = (i, seq)
            scheduler.add(seq)
        if scheduler.is_finished():
            if not pending:
                break
            step += 1
            continue
        batch, is_prefill = scheduler.schedule()
        token_ids = model.run(batch, is_prefill)
        scheduler.postprocess(batch, token_ids, is_prefill)
        for seq in batch:
            if seq.is_finished:
                outputs[seqs[seq.seq_id][0]] = seq.completion_token_ids
        check_block_manager(scheduler.block_manager, [s for s in [*scheduler.running, *scheduler.waiting] if s.block_table])
        for _, seq in seqs.values():
            if seq.num_completion_tokens > seq.max_tokens:
                raise Violation(f"seq {seq.seq_id}: more than max_tokens completion tokens")
        step += 1
        if step > max_steps:
            raise Violation("engine did not finish")
    block_manager = scheduler.block_manager
    if len(block_manager.free_block_ids) != len(block_manager.blocks) or block_manager.used_block_ids:
        raise Violation("blocks leaked after all requests finished")
    return outputs


def random_requests(rng, block_size, capacity, pressure):
    """Prompts that share prefixes of various lengths, including ones that share later blocks but
    not the first (prefix hashes must chain), with max_tokens capped so a lone request fits."""
    bs = block_size
    bases = [[rng.randrange(1, VOCAB) for _ in range(rng.randint(1, 4 * bs + 1))] for _ in range(3)]
    requests = []
    for _ in range(rng.randint(4, 16) if pressure else rng.randint(1, 12)):
        r = rng.random()
        if r < 0.15:
            base = rng.choice(bases)
            prompt = [rng.randrange(1, VOCAB) for _ in range(bs)] + base[bs:] + [rng.randrange(1, VOCAB) for _ in range(rng.randint(0, bs))]
        elif r < 0.5:
            base = rng.choice(bases)
            k = max(1, rng.choice([len(base), len(base) // bs * bs, rng.randint(1, len(base))]))
            prompt = base[:k] + [rng.randrange(1, VOCAB) for _ in range(rng.choice([0, 0, 1, bs - 1, bs, rng.randint(0, 2 * bs)]))]
        else:
            length = max(1, rng.choice([bs - 1, bs, bs + 1, 2 * bs, 2 * bs + 1, rng.randint(1, 5 * bs)]))
            prompt = [rng.randrange(1, VOCAB) for _ in range(length)]
        max_tokens = rng.choice([1, 2, bs, bs + 1, rng.randint(1, 3 * bs)] + ([rng.randint(bs, 6 * bs)] * 3 if pressure else []))
        if len(prompt) + max_tokens > capacity:
            prompt = prompt[:max(1, capacity - max_tokens)]
            max_tokens = min(max_tokens, capacity - len(prompt))
        sp = SamplingParams(temperature=1.0, max_tokens=max_tokens, ignore_eos=rng.random() < 0.5)
        arrival = 0 if rng.random() < 0.6 else rng.randint(0, 30)
        requests.append((arrival, prompt, sp))
    return requests


def run_random_case(seed):
    rng = random.Random(seed)
    bs = rng.choice([2, 3, 4, 8])
    Sequence.block_size = bs
    pressure = rng.random() < 0.5
    num_blocks = rng.randint(2, 10) if pressure else rng.randint(3, 40)
    max_num_seqs = rng.randint(1, 8)
    max_num_batched_tokens = max(1, rng.choice([1, 2, bs - 1, bs, bs + 1, 2 * bs, 3 * bs + 1, rng.randint(1, 10 * bs), 10 ** 6]))
    scheduler = Scheduler(make_config(num_blocks, bs, max_num_seqs, max_num_batched_tokens))
    model = FakeModel(scheduler, bs, num_blocks)
    requests = random_requests(rng, bs, num_blocks * bs, pressure)
    outputs = run_engine(scheduler, model, requests)
    for i, (_, prompt, sp) in enumerate(requests):
        expected = reference(prompt, sp)
        if outputs.get(i) != expected:
            raise Violation(f"request {i}: output {outputs.get(i)} != reference {expected}")


def run_random_case_block_256(seed):
    """Same checks with the default block size and realistic token budgets."""
    rng = random.Random(seed)
    bs = 256
    Sequence.block_size = bs
    num_blocks = rng.randint(4, 16)
    max_num_batched_tokens = rng.choice([256, 300, 511, 512, 1000, 16384])
    scheduler = Scheduler(make_config(num_blocks, bs, rng.randint(1, 6), max_num_batched_tokens))
    model = FakeModel(scheduler, bs, num_blocks)
    base = [rng.randrange(1, VOCAB) for _ in range(1200)]
    requests = []
    for _ in range(rng.randint(2, 6)):
        k = rng.choice([255, 256, 257, 511, 512, 513, 768, rng.randint(1, 1200)])
        prompt = (base[:k] if rng.random() < 0.6 else [rng.randrange(1, VOCAB) for _ in range(k)]) + [rng.randrange(1, VOCAB) for _ in range(rng.choice([0, 1, 5]))]
        max_tokens = rng.choice([1, 2, 300, rng.randint(1, 600)])
        if len(prompt) + max_tokens > num_blocks * bs:
            max_tokens = max(1, num_blocks * bs - len(prompt))
            prompt = prompt[:num_blocks * bs - max_tokens]
        requests.append((0, prompt, SamplingParams(max_tokens=max_tokens, ignore_eos=rng.random() < 0.7)))
    outputs = run_engine(scheduler, model, requests)
    for i, (_, prompt, sp) in enumerate(requests):
        if outputs.get(i) != reference(prompt, sp):
            raise Violation(f"request {i}: output differs from reference")
