"""Randomized engine simulation: the real LLMEngine.step, Scheduler, BlockManager, Sequence and
ModelRunner.prepare_prefill/prepare_decode, driven by a fake model instead of a GPU, with and
without async scheduling, and with speculative decoding (n-gram drafts checked by the real
RejectionSampler).

For every position p of a sequence, the fake model writes prefix_hash(tokens[:p+1]) into a paged
KV cache at the slot that slot_mapping gives, then "attends" by reading every context position
through block_tables and checks each value it reads. The next token is a function of the prefix,
so each request has a reference output that does not depend on batching, chunked prefill, prefix
caching, preemption or speculation, and the engine must reproduce it exactly.
"""
import collections
import hashlib
import pickle
import random
import types

import torch

from nanovllm.engine.llm_engine import LLMEngine
from nanovllm.engine.model_runner import ModelRunner
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.sequence import Sequence
from nanovllm.layers.sampler import RejectionSampler, accepted_token_ids
from nanovllm.sampling_params import SamplingParams
from nanovllm.utils.context import get_context, reset_context

VOCAB = 40
EOS = 0


class Violation(AssertionError):
    pass


def make_config(num_blocks, block_size, max_num_seqs, max_num_batched_tokens, eos=EOS, num_speculative_tokens=0):
    # Config itself requires a model directory and block sizes that are multiples of 256.
    return types.SimpleNamespace(num_kvcache_blocks=num_blocks, kvcache_block_size=block_size,
                                 max_num_seqs=max_num_seqs, max_num_batched_tokens=max_num_batched_tokens, eos=eos,
                                 num_speculative_tokens=num_speculative_tokens)


def make_runner(block_size, world_size=1, rank=0, num_speculative_tokens=0):
    # Only the attributes that prepare_prefill/prepare_decode read; no model or KV cache.
    runner = ModelRunner.__new__(ModelRunner)
    runner.block_size = block_size
    runner.world_size = world_size
    runner.rank = rank
    runner.enforce_eager = True
    runner.num_speculative_tokens = num_speculative_tokens
    return runner


def make_engine(num_blocks, block_size, max_num_seqs, max_num_batched_tokens, async_scheduling=False, num_speculative_tokens=0):
    assert not (async_scheduling and num_speculative_tokens)
    engine = LLMEngine.__new__(LLMEngine)
    engine.scheduler = Scheduler(make_config(num_blocks, block_size, max_num_seqs, max_num_batched_tokens,
                                             num_speculative_tokens=num_speculative_tokens))
    engine.model_runner = FakeModel(engine.scheduler, block_size, num_blocks, num_speculative_tokens=num_speculative_tokens)
    engine.async_scheduling = async_scheduling
    engine.pending = None
    return engine


def prefix_hashes(token_ids, h=12345):
    out = []
    for t in token_ids:
        h = int.from_bytes(hashlib.blake2b(f"{h}:{t}".encode(), digest_size=8).digest(), "little")
        out.append(h)
    return out


def next_token(token_ids, h):
    """The fake model's next token after token_ids, whose prefix hash is h. Prompts that start with
    an even token get mostly periodic outputs, which n-gram drafts often predict; the others get
    pseudo-random outputs, whose drafts are mostly rejected."""
    r = h >> 7
    period = 1 + token_ids[0] // 2 % 3
    if token_ids[0] % 2 == 0 and r % 8 and len(token_ids) >= period:
        return token_ids[-period]
    return r % VOCAB


def reference(prompt, sp: SamplingParams):
    token_ids = list(prompt)
    h = prefix_hashes(token_ids)[-1]
    out = []
    while True:
        t = next_token(token_ids, h)
        token_ids.append(t)
        h = prefix_hashes([t], h)[0]
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


class FakeOutput:

    def __init__(self, token_ids):
        self.token_ids = token_ids

    def result(self):
        return self.token_ids


class FakeModel:
    """Stands in for ModelRunner.call("run" / "run_async"). Like the GPU, it keeps the last step's
    sampled tokens, which the real prepare_decode reads for tokens still in flight."""

    def __init__(self, scheduler: Scheduler, block_size: int, num_blocks: int, check_tp: bool = True, num_speculative_tokens: int = 0):
        self.block_size = block_size
        self.kv = [None] * (num_blocks * block_size)
        self.runner = make_runner(block_size, num_speculative_tokens=num_speculative_tokens)
        self.block_manager = scheduler.block_manager
        self.check_tp = check_tp
        self.num_speculative_tokens = num_speculative_tokens
        self.rejection_sampler = RejectionSampler()
        self.last_outputs = {}
        self.num_samples = collections.Counter()    # tokens sampled per seq, kept or not
        self.num_accepted = collections.Counter()   # decode steps by number of accepted drafts
        self.prefix_hashes = {}                     # seq_id -> prefix hashes of its tokens so far
        self.checked_blocks = set()                 # (block_id, hash) pairs whose KV was checked

    def call(self, method_name, seqs, is_prefill):
        token_ids = self.run(seqs, is_prefill)
        for seq, t in zip(seqs, token_ids):
            if not is_prefill or seq.num_cached_tokens + seq.num_scheduled_tokens == seq.num_tokens:
                self.num_samples[seq.seq_id] += len(t) if isinstance(t, list) else 1
        self.last_outputs = {seq.seq_id: t for seq, t in zip(seqs, token_ids)}
        if not self.num_speculative_tokens:
            self.runner.last_token_ids = torch.tensor(token_ids, dtype=torch.int64)
        return token_ids if method_name == "run" else FakeOutput(token_ids)

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
        outputs = []
        for i, seq in enumerate(seqs):
            q0, q1 = cu_q[i], cu_q[i + 1]
            k_len = cu_k[i + 1] - cu_k[i]
            start = seq.num_cached_tokens
            if seq.num_inflight:
                raise Violation(f"seq {seq.seq_id}: prefill scheduled with a token in flight")
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
                local[p] = truth[p]
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
            outputs.append(next_token(seq.token_ids[:k_len], truth[pos[q1 - 1]]))
        return outputs

    def _decode(self, seqs, meta):
        bs = self.block_size
        ids, pos, slots = meta["input_ids"], meta["positions"], meta["slot_mapping"]
        context_lens, block_tables = meta["context_lens"], meta["block_tables"]
        q = 1 + self.num_speculative_tokens
        if len(ids) != q * len(seqs):
            raise Violation(f"{len(ids)} decode inputs for {len(seqs)} seqs, expected {q} each")
        truths, targets = [], []
        for i, seq in enumerate(seqs):
            rows = range(i * q, (i + 1) * q)
            token_ids = seq.token_ids
            if seq.num_inflight:
                if seq.num_inflight != 1 or ids[rows[0]] != self.last_outputs.get(seq.seq_id):
                    raise Violation(f"seq {seq.seq_id}: in-flight input {ids[rows[0]]} is not the token sampled last step")
                token_ids = token_ids + [ids[rows[0]]]
            elif ids[rows[0]] != token_ids[-1]:
                raise Violation(f"seq {seq.seq_id}: wrong decode input id")
            drafts = ids[rows.start + 1:rows.stop]
            if drafts != seq.draft_token_ids:
                raise Violation(f"seq {seq.seq_id}: draft inputs {drafts} != proposed {seq.draft_token_ids}")
            n = len(token_ids)
            token_ids = token_ids + drafts
            if [pos[r] for r in rows] != list(range(n - 1, n + q - 1)) or context_lens[i] != n + q - 1:
                raise Violation(f"seq {seq.seq_id}: wrong decode inputs positions={[pos[r] for r in rows]} len={n} context_len={context_lens[i]}")
            truths.append(prefix_hashes(token_ids))
            for r in rows:
                self._write(seq, pos[r], slots[r], truths[i][pos[r]])
            targets.append([next_token(token_ids[:p + 1], truths[i][p]) for p in range(n - 1, n + q - 1)])
        for i, seq in enumerate(seqs):
            for p in range(context_lens[i]):
                if self.kv[block_tables[i][p // bs] * bs + p % bs] != truths[i][p]:
                    raise Violation(f"seq {seq.seq_id}: decode read wrong KV at position {p} (context_len {context_lens[i]})")
        if q == 1:
            return [t[0] for t in targets]
        # A target model that is certain of each next token, verified by the real rejection sampler
        # (compiled, on the GPU if there is one) with the draft tokens taken from the inputs, as
        # ModelRunner.run does.
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logits = torch.full((len(ids), VOCAB), -1000.0, device=device)
        logits[torch.arange(len(ids)), torch.tensor(sum(targets, []))] = 0.0
        temperatures = torch.tensor([seq.temperature for seq in seqs], dtype=torch.float32, device=device)
        draft_token_ids = torch.tensor(ids, dtype=torch.int64, device=device).view(len(seqs), q)[:, 1:]
        token_ids = accepted_token_ids(self.rejection_sampler(logits, temperatures, draft_token_ids).tolist())
        for i, seq in enumerate(seqs):
            accepted = next(j for j in range(q) if j == q - 1 or seq.draft_token_ids[j] != targets[i][j])
            if token_ids[i] != seq.draft_token_ids[:accepted] + [targets[i][accepted]]:
                raise Violation(f"seq {seq.seq_id}: rejection sampler returned {token_ids[i]} for drafts {seq.draft_token_ids} and targets {targets[i]}")
            self.num_accepted[accepted] += 1
        return token_ids

    def check_hashed_blocks(self, seqs):
        """Every block registered for prefix caching must hold the KV of its tokens (with speculative
        decoding, rejected drafts leave stale KV that must not be hashed)."""
        bs = self.block_size
        for seq in seqs:
            hashes = self.prefix_hashes.setdefault(seq.seq_id, [])
            if len(hashes) < seq.num_tokens:
                hashes += prefix_hashes(seq.token_ids[len(hashes):], hashes[-1] if hashes else 12345)
            for i, block_id in enumerate(seq.block_table):
                block = self.block_manager.blocks[block_id]
                if block.hash == -1 or (block_id, block.hash) in self.checked_blocks:
                    continue
                for p in range(i * bs, (i + 1) * bs):
                    if self.kv[block_id * bs + p % bs] != hashes[p]:
                        raise Violation(f"seq {seq.seq_id}: hashed block {block_id} holds wrong KV at position {p}")
                self.checked_blocks.add((block_id, block.hash))

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


def run_engine(engine, requests, max_steps=50000):
    """Feed (arrival_step, prompt, sampling_params) requests to the engine, check the block manager
    after every step, and return {request index: completion token ids}."""
    scheduler = engine.scheduler
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
        if engine.is_finished():
            if not pending:
                break
            step += 1
            continue
        for seq_id, token_ids in engine.step()[0]:
            if seqs[seq_id][0] in outputs:
                raise Violation(f"seq {seq_id} finished twice")
            outputs[seqs[seq_id][0]] = token_ids
        live_seqs = [s for s in [*scheduler.running, *scheduler.waiting] if s.block_table]
        check_block_manager(scheduler.block_manager, live_seqs)
        engine.model_runner.check_hashed_blocks(live_seqs)
        k = engine.model_runner.num_speculative_tokens
        for _, seq in seqs.values():
            if seq.num_completion_tokens > seq.max_tokens:
                raise Violation(f"seq {seq.seq_id}: more than max_tokens completion tokens")
            # Async scheduling may sample one token too many for a seq that hit EOS, never more.
            # Speculative decoding may produce up to k tokens past EOS or max_tokens in the last step.
            in_flight = 0 if seq.is_finished else seq.num_inflight
            wasted = engine.model_runner.num_samples[seq.seq_id] - seq.num_completion_tokens - in_flight
            allowed = (engine.async_scheduling and seq.is_finished and seq.num_completion_tokens < seq.max_tokens) + k * seq.is_finished
            if wasted > allowed:
                raise Violation(f"seq {seq.seq_id}: {wasted} sampled tokens thrown away")
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
            lowest = 0 if rng.random() < 0.3 else 1    # EOS in the prompt can be copied into drafts
            prompt = [rng.randrange(lowest, VOCAB) for _ in range(length)]
        max_tokens = rng.choice([1, 2, bs, bs + 1, rng.randint(1, 3 * bs)] + ([rng.randint(bs, 6 * bs)] * 3 if pressure else []))
        if len(prompt) + max_tokens > capacity:
            prompt = prompt[:max(1, capacity - max_tokens)]
            max_tokens = min(max_tokens, capacity - len(prompt))
        if rng.random() < 0.15:    # EOS near the end of the prompt shows up in periodic outputs and their drafts
            prompt[-rng.randint(1, min(3, len(prompt)))] = EOS
        sp = SamplingParams(temperature=rng.choice([0.0, 1.0]), max_tokens=max_tokens, ignore_eos=rng.random() < 0.5)
        arrival = 0 if rng.random() < 0.6 else rng.randint(0, 30)
        requests.append((arrival, prompt, sp))
    return requests


def run_random_case(seed, async_scheduling=False, num_speculative_tokens=0):
    rng = random.Random(seed)
    bs = rng.choice([2, 3, 4, 8])
    Sequence.block_size = bs
    pressure = rng.random() < 0.5
    num_blocks = rng.randint(2, 10) if pressure else rng.randint(3, 40)
    # A lone request must fit, and a decode step writes num_speculative_tokens slots past its last token.
    num_blocks = max(num_blocks, (num_speculative_tokens + 2 + bs - 1) // bs)
    max_num_seqs = rng.randint(1, 8)
    max_num_batched_tokens = max(1, rng.choice([1, 2, bs - 1, bs, bs + 1, 2 * bs, 3 * bs + 1, rng.randint(1, 10 * bs), 10 ** 6]))
    engine = make_engine(num_blocks, bs, max_num_seqs, max_num_batched_tokens, async_scheduling, num_speculative_tokens)
    requests = random_requests(rng, bs, num_blocks * bs - num_speculative_tokens, pressure)
    outputs = run_engine(engine, requests)
    for i, (_, prompt, sp) in enumerate(requests):
        expected = reference(prompt, sp)
        if outputs.get(i) != expected:
            raise Violation(f"request {i}: output {outputs.get(i)} != reference {expected}")


def run_random_case_block_256(seed, async_scheduling=False, num_speculative_tokens=0):
    """Same checks with the default block size and realistic token budgets."""
    rng = random.Random(seed)
    bs = 256
    Sequence.block_size = bs
    num_blocks = rng.randint(4, 16)
    capacity = num_blocks * bs - num_speculative_tokens
    max_num_batched_tokens = rng.choice([256, 300, 511, 512, 1000, 16384])
    engine = make_engine(num_blocks, bs, rng.randint(1, 6), max_num_batched_tokens, async_scheduling, num_speculative_tokens)
    base = [rng.randrange(1, VOCAB) for _ in range(1200)]
    requests = []
    for _ in range(rng.randint(2, 6)):
        k = rng.choice([255, 256, 257, 511, 512, 513, 768, rng.randint(1, 1200)])
        prompt = (base[:k] if rng.random() < 0.6 else [rng.randrange(1, VOCAB) for _ in range(k)]) + [rng.randrange(1, VOCAB) for _ in range(rng.choice([0, 1, 5]))]
        max_tokens = rng.choice([1, 2, 300, rng.randint(1, 600)])
        if len(prompt) + max_tokens > capacity:
            max_tokens = max(1, capacity - len(prompt))
            prompt = prompt[:capacity - max_tokens]
        requests.append((0, prompt, SamplingParams(max_tokens=max_tokens, ignore_eos=rng.random() < 0.7)))
    outputs = run_engine(engine, requests)
    for i, (_, prompt, sp) in enumerate(requests):
        if outputs.get(i) != reference(prompt, sp):
            raise Violation(f"request {i}: output differs from reference")
