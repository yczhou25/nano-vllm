from collections import deque

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.engine.block_manager import BlockManager
from nanovllm.engine.ngram import NgramProposer


class Scheduler:
    # The acceptance switch (speculative_min_acceptance) follows the acceptance rate over about the
    # last ACCEPTANCE_WINDOW drafts. While it is below the threshold, decode steps run without drafts,
    # except one probe step with drafts every PROBE_INTERVAL steps, which is how drafts come back
    # once the workload changes.
    ACCEPTANCE_WINDOW = 256    # drafts
    PROBE_INTERVAL = 16        # decode steps

    def __init__(self, config: Config):
        self.max_num_seqs = config.max_num_seqs
        self.max_num_batched_tokens = config.max_num_batched_tokens
        self.eos = config.eos
        self.block_size = config.kvcache_block_size
        self.block_manager = BlockManager(config.num_kvcache_blocks, config.kvcache_block_size)
        self.num_speculative_tokens = config.num_speculative_tokens
        self.has_draft_model = config.speculative_model is not None    # the model runner drafts on the GPU
        self.speculative_disable_by_batch_size = config.speculative_disable_by_batch_size
        self.speculative_min_acceptance = config.speculative_min_acceptance
        self.acceptance = 1.0    # draft acceptance rate over about the last ACCEPTANCE_WINDOW drafts; drafts are trusted until measured
        self.num_acceptance_drafts = 0    # drafts behind self.acceptance, capped at ACCEPTANCE_WINDOW
        self.num_steps_without_drafts = 0    # consecutive decode steps without drafts, for the probe
        self.proposer = NgramProposer(config.num_speculative_tokens) if config.num_speculative_tokens and not self.has_draft_model else None
        self.num_draft_tokens = self.num_accepted_tokens = 0    # speculative decoding statistics
        self.waiting: deque[Sequence] = deque()
        self.running: deque[Sequence] = deque()

    def is_finished(self):
        return not self.waiting and not self.running

    def add(self, seq: Sequence):
        self.waiting.append(seq)

    def schedule(self) -> tuple[list[Sequence], bool]:
        scheduled_seqs = []
        num_batched_tokens = 0

        # prefill
        while self.waiting and len(scheduled_seqs) < self.max_num_seqs:
            seq = self.waiting[0]
            remaining = self.max_num_batched_tokens - num_batched_tokens
            if remaining == 0:
                break
            if not seq.block_table:
                num_cached_blocks = self.block_manager.can_allocate(seq)
                if num_cached_blocks == -1:
                    break
                num_tokens = seq.num_tokens - num_cached_blocks * self.block_size
            else:
                num_tokens = seq.num_tokens - seq.num_cached_tokens
            if remaining < num_tokens and scheduled_seqs:  # only allow chunked prefill for the first seq
                break
            if not seq.block_table:
                self.block_manager.allocate(seq, num_cached_blocks)
                seq.num_draft_cached_tokens = seq.num_cached_tokens    # a hashed block holds both models' KV
            seq.num_scheduled_tokens = min(num_tokens, remaining)
            num_batched_tokens += seq.num_scheduled_tokens
            if seq.num_cached_tokens + seq.num_scheduled_tokens == seq.num_tokens:
                seq.status = SequenceStatus.RUNNING
                self.waiting.popleft()
                self.running.append(seq)
            scheduled_seqs.append(seq)

        if scheduled_seqs:
            return scheduled_seqs, True

        # decode
        k = self.num_speculative_tokens
        if k and self.speculative_disable_by_batch_size is not None and len(self.running) > self.speculative_disable_by_batch_size:
            k = 0    # a batch this large keeps the GPU busy by itself; the k extra positions would not be free
        if k and self.speculative_min_acceptance is not None and self.acceptance < self.speculative_min_acceptance \
                and self.num_steps_without_drafts < self.PROBE_INTERVAL:
            k = 0    # drafts are rejected too often to pay for themselves; the next probe step measures again
        num_decode_tokens = 1 + k
        kept_seqs = []
        while self.running and len(scheduled_seqs) < self.max_num_seqs:
            seq = self.running.popleft()
            if seq.num_completion_tokens + seq.num_inflight >= seq.max_tokens:    # the step in flight may finish it
                kept_seqs.append(seq)
                continue
            while not self.block_manager.can_append(seq, num_decode_tokens):
                if self.running:
                    self.preempt(self.running.pop())
                else:
                    self.preempt(seq)
                    break
            else:
                seq.num_scheduled_tokens = num_decode_tokens
                seq.is_prefill = False
                if self.proposer:
                    seq.draft_token_ids = self.proposer.propose(seq) if k else []
                self.block_manager.may_append(seq, num_decode_tokens)
                scheduled_seqs.append(seq)
                kept_seqs.append(seq)
        self.running.extendleft(reversed(kept_seqs))
        if scheduled_seqs:
            self.num_steps_without_drafts = 0 if k else self.num_steps_without_drafts + 1
        return scheduled_seqs, False

    def preempt(self, seq: Sequence):
        seq.status = SequenceStatus.WAITING
        seq.is_prefill = True
        seq.num_draft_cached_tokens = 0    # the prefill rebuilds both KV caches
        self.block_manager.deallocate(seq)
        self.waiting.appendleft(seq)

    def commit(self, seqs: list[Sequence], is_prefill: bool) -> list[tuple[Sequence, bool, int, int, int]]:
        """Advance sequences past a step that has been launched. Its sampled tokens stay in flight
        until postprocess, so with async scheduling the next step can be scheduled first. A prefill
        chunk's KV is counted now, so that the next chunk follows it; a decode step's KV is counted
        in postprocess, once the number of accepted draft tokens is known."""
        batch = []
        for row, seq in enumerate(seqs):
            start, num_tokens = seq.num_cached_tokens, seq.num_scheduled_tokens
            seq.num_scheduled_tokens = 0
            if is_prefill:
                seq.num_cached_tokens += num_tokens
                seq.num_draft_cached_tokens = seq.num_cached_tokens    # the draft model prefills alongside the target
                num_inflight = int(seq.num_cached_tokens == seq.num_tokens)    # samples once the prompt is done
            else:
                num_inflight = num_tokens    # the newest token and the drafts: the most the step can add
            if num_inflight:
                seq.num_inflight += num_inflight
                seq.inflight_row = row
            batch.append((seq, is_prefill, start, num_tokens, num_inflight))
        return batch

    def postprocess(self, batch: list[tuple[Sequence, bool, int, int, int]], token_ids: list[int] | list[list[int]]) -> list[Sequence]:
        """token_ids has one token per sequence, or with speculative decoding, a list per decoded
        sequence: its accepted draft tokens followed by one sampled token."""
        finished = []
        num_drafts = num_accepted = 0    # this step's drafts and how many were accepted
        for (seq, is_prefill, start, num_tokens, num_inflight), new_token_ids in zip(batch, token_ids):
            if seq.is_finished:    # finished while this step was in flight
                continue
            stop = False
            if num_inflight:
                seq.num_inflight -= num_inflight
                speculative = isinstance(new_token_ids, list)
                for token_id in new_token_ids if speculative else [new_token_ids]:
                    seq.append_token(token_id)
                    stop = (not seq.ignore_eos and token_id == self.eos) or seq.num_completion_tokens == seq.max_tokens
                    if stop:
                        break
                if not is_prefill:
                    # The step wrote the KV of its input token and of the drafts after it; only the
                    # input token's and the accepted drafts' is valid, later steps overwrite the rest.
                    start = seq.num_cached_tokens
                    num_tokens = seq.num_tokens - start - 1
                    seq.num_cached_tokens = start + num_tokens
                    if speculative or not self.has_draft_model:
                        # A step with drafts first caught the draft model's KV cache up to the newest
                        # token (ModelRunner.propose), then scored every draft but the last one, so when
                        # all of them were accepted the cache lacks the newest cached token. A step
                        # without drafts (see schedule) skips the draft model: its cache falls behind
                        # until the next step with drafts, and the blocks it lacks must not be hashed
                        # yet, or a prefix cache hit would read a stale draft KV.
                        lag = self.has_draft_model and len(new_token_ids) == self.num_speculative_tokens + 1
                        seq.num_draft_cached_tokens = seq.num_cached_tokens - lag
                if speculative:
                    num_drafts += self.num_speculative_tokens
                    num_accepted += len(new_token_ids) - 1
            if seq.block_table:    # empty if preempted while this step was in flight
                # Blocks both KV caches hold completely; the ones a lagging draft cache left for later are hashed
                # once it catches up (hash_blocks scans back to them).
                valid = start + num_tokens if is_prefill else min(seq.num_cached_tokens, seq.num_draft_cached_tokens)
                self.block_manager.hash_blocks(seq, start, valid - start)
            if stop:
                if seq.status == SequenceStatus.RUNNING:
                    self.running.remove(seq)
                else:
                    self.waiting.remove(seq)
                seq.status = SequenceStatus.FINISHED
                self.block_manager.deallocate(seq)
                if self.proposer:
                    self.proposer.forget(seq)
                finished.append(seq)
        if num_drafts:
            self.num_draft_tokens += num_drafts
            self.num_accepted_tokens += num_accepted
            # A running mean over about the last ACCEPTANCE_WINDOW drafts: a plain mean until that many
            # were seen, then each step's drafts replace that share of the window.
            seen = min(self.num_acceptance_drafts, self.ACCEPTANCE_WINDOW)
            self.acceptance = (self.acceptance * seen + num_accepted) / (seen + num_drafts)
            self.num_acceptance_drafts = min(seen + num_drafts, self.ACCEPTANCE_WINDOW)
        return finished
