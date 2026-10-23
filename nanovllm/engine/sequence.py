from copy import copy
from enum import Enum, auto
from itertools import count

from nanovllm.sampling_params import SamplingParams


class SequenceStatus(Enum):
    WAITING = auto()
    RUNNING = auto()
    FINISHED = auto()


class Sequence:
    block_size = 256
    counter = count()

    def __init__(self, token_ids: list[int], sampling_params = SamplingParams()):
        self.seq_id = next(Sequence.counter)
        self.status = SequenceStatus.WAITING
        self.token_ids = copy(token_ids)
        self.last_token = token_ids[-1]
        self.num_tokens = len(self.token_ids)
        self.num_prompt_tokens = len(token_ids)
        self.num_cached_tokens = 0
        self.num_scheduled_tokens = 0
        self.num_inflight = 0       # tokens sampled on the GPU but not appended yet (async scheduling)
        self.inflight_row = -1      # row of the newest in-flight token in the last step's output
        self.draft_token_ids = []   # tokens proposed for this decode step (speculative decoding with n-gram drafts)
        self.draft_cache_lags = False   # the draft model's KV cache lacks the newest cached token (see Scheduler.postprocess)
        self.is_prefill = True
        self.block_table = []
        self.temperature = sampling_params.temperature
        self.max_tokens = sampling_params.max_tokens
        self.ignore_eos = sampling_params.ignore_eos

    def __len__(self):
        return self.num_tokens

    def __getitem__(self, key):
        return self.token_ids[key]

    @property
    def is_finished(self):
        return self.status == SequenceStatus.FINISHED

    @property
    def num_completion_tokens(self):
        return self.num_tokens - self.num_prompt_tokens

    @property
    def prompt_token_ids(self):
        return self.token_ids[:self.num_prompt_tokens]

    @property
    def completion_token_ids(self):
        return self.token_ids[self.num_prompt_tokens:]

    @property
    def num_blocks(self):
        return (self.num_tokens + self.block_size - 1) // self.block_size

    @property
    def last_block_num_tokens(self):
        return self.num_tokens - (self.num_blocks - 1) * self.block_size

    def block(self, i):
        assert 0 <= i < self.num_blocks
        return self.token_ids[i*self.block_size: (i+1)*self.block_size]

    def append_token(self, token_id: int):
        self.token_ids.append(token_id)
        self.last_token = token_id
        self.num_tokens += 1

    def __getstate__(self):
        if self.is_prefill:
            last_state = self.token_ids[self.num_cached_tokens:self.num_cached_tokens + self.num_scheduled_tokens]
        else:
            last_state = tuple(self.token_ids[-2:])    # a draft model scores the last two tokens
        return (self.num_tokens, self.num_prompt_tokens, self.num_cached_tokens, self.num_scheduled_tokens, self.num_inflight, self.inflight_row, self.block_table, self.draft_token_ids, self.draft_cache_lags, last_state)

    def __setstate__(self, state):
        self.num_tokens, self.num_prompt_tokens, self.num_cached_tokens, self.num_scheduled_tokens, self.num_inflight, self.inflight_row, self.block_table, self.draft_token_ids, self.draft_cache_lags, last_state = state
        if isinstance(last_state, list):
            self.token_ids = [0] * self.num_cached_tokens + last_state
        else:
            self.token_ids = [0] * (self.num_tokens - len(last_state)) + list(last_state)
        self.last_token = self.token_ids[-1]
