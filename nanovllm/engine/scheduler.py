from collections import deque

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.engine.block_manager import BlockManager
from nanovllm.engine.ngram import NgramProposer


class Scheduler:

    def __init__(self, config: Config):
        self.max_num_seqs = config.max_num_seqs
        self.max_num_batched_tokens = config.max_num_batched_tokens
        self.eos = config.eos
        self.block_size = config.kvcache_block_size
        self.block_manager = BlockManager(config.num_kvcache_blocks, config.kvcache_block_size)
        self.num_speculative_tokens = config.num_speculative_tokens
        self.proposer = NgramProposer(config.num_speculative_tokens) if config.num_speculative_tokens else None
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
        num_decode_tokens = 1 + self.num_speculative_tokens
        kept_seqs = []
        while self.running and len(scheduled_seqs) < self.max_num_seqs:
            seq = self.running.popleft()
            if seq.num_completion_tokens + seq.num_inflight == seq.max_tokens:    # last token is in flight
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
                    seq.draft_token_ids = self.proposer.propose(seq)
                self.block_manager.may_append(seq, num_decode_tokens)
                scheduled_seqs.append(seq)
                kept_seqs.append(seq)
        self.running.extendleft(reversed(kept_seqs))
        return scheduled_seqs, False

    def preempt(self, seq: Sequence):
        seq.status = SequenceStatus.WAITING
        seq.is_prefill = True
        self.block_manager.deallocate(seq)
        self.waiting.appendleft(seq)

    def commit(self, seqs: list[Sequence], is_prefill: bool) -> list[tuple[Sequence, int, int, bool]]:
        """Advance sequences past a step that has been launched. Its sampled tokens stay in flight
        until postprocess, so with async scheduling the next step can be scheduled first."""
        batch = []
        for row, seq in enumerate(seqs):
            start, num_tokens = seq.num_cached_tokens, seq.num_scheduled_tokens
            seq.num_cached_tokens += num_tokens
            seq.num_scheduled_tokens = 0
            samples = not is_prefill or seq.num_cached_tokens == seq.num_tokens
            if samples:
                seq.num_inflight += 1
                seq.inflight_row = row
            batch.append((seq, start, num_tokens, samples))
        return batch

    def postprocess(self, batch: list[tuple[Sequence, int, int, bool]], token_ids: list[int] | list[list[int]]) -> list[Sequence]:
        """token_ids has one token per sequence, or with speculative decoding, a list per decoded
        sequence: its accepted draft tokens followed by one sampled token."""
        finished = []
        for (seq, start, num_tokens, samples), new_token_ids in zip(batch, token_ids):
            if seq.is_finished:    # finished while this step was in flight
                continue
            stop = False
            if samples:
                seq.num_inflight -= 1
                speculative = isinstance(new_token_ids, list)
                for token_id in new_token_ids if speculative else [new_token_ids]:
                    seq.append_token(token_id)
                    stop = (not seq.ignore_eos and token_id == self.eos) or seq.num_completion_tokens == seq.max_tokens
                    if stop:
                        break
                if speculative:
                    self.num_draft_tokens += self.num_speculative_tokens
                    self.num_accepted_tokens += len(new_token_ids) - 1
                    # Only the KV of the input token and the accepted drafts is valid; later steps
                    # overwrite the rest.
                    num_tokens = seq.num_tokens - start - 1
                    seq.num_cached_tokens = start + num_tokens
            if seq.block_table:    # empty if preempted while this step was in flight
                self.block_manager.hash_blocks(seq, start, num_tokens)
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
        return finished
