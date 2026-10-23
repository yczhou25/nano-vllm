import os
from dataclasses import dataclass
from transformers import AutoConfig


@dataclass(slots=True)
class Config:
    model: str
    max_num_batched_tokens: int = 16384
    max_num_seqs: int = 512
    max_model_len: int = 4096
    gpu_memory_utilization: float = 0.9
    tensor_parallel_size: int = 1
    enforce_eager: bool = False
    async_scheduling: bool = False
    num_speculative_tokens: int = 0
    speculative_model: str | None = None    # a smaller model that drafts the speculative tokens; n-gram lookup if None
    hf_config: AutoConfig | None = None
    draft_hf_config: AutoConfig | None = None
    eos: int = -1
    kvcache_block_size: int = 256
    num_kvcache_blocks: int = -1

    def __post_init__(self):
        assert os.path.isdir(self.model)
        assert self.kvcache_block_size % 256 == 0
        assert 1 <= self.tensor_parallel_size <= 8
        assert self.num_speculative_tokens >= 0
        assert not (self.async_scheduling and self.num_speculative_tokens), "speculative decoding does not support async scheduling yet"
        self.hf_config = AutoConfig.from_pretrained(self.model)
        self.max_model_len = min(self.max_model_len, self.hf_config.max_position_embeddings)
        if self.speculative_model is not None:
            assert os.path.isdir(self.speculative_model)
            assert self.num_speculative_tokens > 0, "speculative_model needs num_speculative_tokens"
            self.draft_hf_config = AutoConfig.from_pretrained(self.speculative_model)
            assert self.draft_hf_config.vocab_size == self.hf_config.vocab_size, "the draft model must share the target's tokenizer"
            self.max_model_len = min(self.max_model_len, self.draft_hf_config.max_position_embeddings)
