import pickle
import torch
import torch.distributed as dist
from multiprocessing.synchronize import Event
from multiprocessing.shared_memory import SharedMemory

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence
from nanovllm.models.qwen3 import Qwen3ForCausalLM
from nanovllm.layers.sampler import Sampler, RejectionSampler, accepted_token_ids
from nanovllm.utils.context import set_context, get_context, restore_context, reset_context
from nanovllm.utils.loader import load_model


class AsyncOutput:
    """A step's sampled tokens, copied to the host in the background: one per sequence, or with
    speculative decoding a row per sequence (RejectionSampler's accepted tokens, -1 padded)."""

    def __init__(self, token_ids: torch.Tensor):
        self.token_ids = torch.empty(token_ids.shape, dtype=token_ids.dtype, pin_memory=True)
        self.token_ids.copy_(token_ids, non_blocking=True)
        self.event = torch.cuda.Event()
        self.event.record()

    def result(self) -> list[int] | list[list[int]]:
        self.event.synchronize()
        token_ids = self.token_ids.tolist()
        return accepted_token_ids(token_ids) if token_ids and isinstance(token_ids[0], list) else token_ids


class ModelRunner:

    def __init__(self, config: Config, rank: int, event: Event | list[Event]):
        self.config = config
        hf_config = config.hf_config
        self.block_size = config.kvcache_block_size
        self.enforce_eager = config.enforce_eager
        self.world_size = config.tensor_parallel_size
        self.num_speculative_tokens = config.num_speculative_tokens
        self.rank = rank
        self.event = event
        self.last_token_ids = None    # last run_async step's sampled tokens, still on the GPU (see AsyncOutput)
        self.draft_model = None
        self.draft_inputs = None      # the draft model's decode inputs, prepared with the target's

        dist.init_process_group("nccl", "tcp://localhost:2333", world_size=self.world_size, rank=rank)
        torch.cuda.set_device(rank)
        default_dtype = torch.get_default_dtype()
        torch.set_default_dtype(hf_config.dtype)
        torch.set_default_device("cuda")
        self.model = Qwen3ForCausalLM(hf_config)
        load_model(self.model, config.model)
        if config.speculative_model is not None:
            self.draft_model = Qwen3ForCausalLM(config.draft_hf_config)
            load_model(self.draft_model, config.speculative_model)
        self.sampler = Sampler()
        self.rejection_sampler = RejectionSampler()
        self.warmup_model()
        self.allocate_kv_cache()
        if not self.enforce_eager:
            self.capture_cudagraph()
        torch.set_default_device("cpu")
        torch.set_default_dtype(default_dtype)

        if self.world_size > 1:
            if rank == 0:
                self.shm = SharedMemory(name="nanovllm", create=True, size=2**20)
                dist.barrier()
            else:
                dist.barrier()
                self.shm = SharedMemory(name="nanovllm")
                self.loop()

    def exit(self):
        if self.world_size > 1:
            self.shm.close()
            dist.barrier()
            if self.rank == 0:
                self.shm.unlink()
        if not self.enforce_eager:
            del self.graphs, self.draft_graphs, self.graph_pool
        torch.cuda.synchronize()
        dist.destroy_process_group()

    def loop(self):
        while True:
            method_name, args = self.read_shm()
            self.call(method_name, *args)
            if method_name == "exit":
                break

    def read_shm(self):
        assert self.world_size > 1 and self.rank > 0
        self.event.wait()
        n = int.from_bytes(self.shm.buf[0:4], "little")
        method_name, *args = pickle.loads(self.shm.buf[4:n+4])
        self.event.clear()
        return method_name, args

    def write_shm(self, method_name, *args):
        assert self.world_size > 1 and self.rank == 0
        for event in self.event:
            while event.is_set():    # with async scheduling a worker may not have read the last call yet
                pass
        data = pickle.dumps([method_name, *args])
        n = len(data)
        self.shm.buf[0:4] = n.to_bytes(4, "little")
        self.shm.buf[4:n+4] = data
        for event in self.event:
            event.set()

    def call(self, method_name, *args):
        if self.world_size > 1 and self.rank == 0:
            self.write_shm(method_name, *args)
        method = getattr(self, method_name, None)
        return method(*args)

    def warmup_model(self):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        max_num_batched_tokens, max_model_len = self.config.max_num_batched_tokens, self.config.max_model_len
        seq_len = min(max_num_batched_tokens, max_model_len)
        num_seqs = min(max_num_batched_tokens // seq_len, self.config.max_num_seqs)
        seqs = [Sequence([0] * seq_len) for _ in range(num_seqs)]
        for seq in seqs:
            seq.num_scheduled_tokens = seq_len
        self.run(seqs, True)
        torch.cuda.empty_cache()

    def kv_cache_block_bytes(self, hf_config) -> int:
        num_kv_heads = hf_config.num_key_value_heads // self.world_size
        head_dim = getattr(hf_config, "head_dim", hf_config.hidden_size // hf_config.num_attention_heads)
        return 2 * hf_config.num_hidden_layers * self.block_size * num_kv_heads * head_dim * hf_config.dtype.itemsize

    def bind_kv_cache(self, model: Qwen3ForCausalLM, hf_config, num_blocks: int) -> torch.Tensor:
        num_kv_heads = hf_config.num_key_value_heads // self.world_size
        head_dim = getattr(hf_config, "head_dim", hf_config.hidden_size // hf_config.num_attention_heads)
        kv_cache = torch.empty(2, hf_config.num_hidden_layers, num_blocks, self.block_size, num_kv_heads, head_dim)
        layer_id = 0
        for module in model.modules():
            if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
                module.k_cache = kv_cache[0, layer_id]
                module.v_cache = kv_cache[1, layer_id]
                layer_id += 1
        return kv_cache

    def allocate_kv_cache(self):
        config = self.config
        free, total = torch.cuda.mem_get_info()
        used = total - free
        peak = torch.cuda.memory_stats()["allocated_bytes.all.peak"]
        current = torch.cuda.memory_stats()["allocated_bytes.all.current"]
        # The draft model keeps its own KV cache, with the same block ids as the target's.
        block_bytes = self.kv_cache_block_bytes(config.hf_config)
        if self.draft_model is not None:
            block_bytes += self.kv_cache_block_bytes(config.draft_hf_config)
        config.num_kvcache_blocks = int(total * config.gpu_memory_utilization - used - peak + current) // block_bytes
        assert config.num_kvcache_blocks > 0
        self.kv_cache = self.bind_kv_cache(self.model, config.hf_config, config.num_kvcache_blocks)
        if self.draft_model is not None:
            self.draft_kv_cache = self.bind_kv_cache(self.draft_model, config.draft_hf_config, config.num_kvcache_blocks)

    def prepare_block_tables(self, seqs: list[Sequence]):
        max_len = max(len(seq.block_table) for seq in seqs)
        block_tables = [seq.block_table + [-1] * (max_len - len(seq.block_table)) for seq in seqs]
        block_tables = torch.tensor(block_tables, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        return block_tables

    def prepare_prefill(self, seqs: list[Sequence]):
        input_ids = []
        positions = []
        cu_seqlens_q = [0]
        cu_seqlens_k = [0]
        max_seqlen_q = 0
        max_seqlen_k = 0
        slot_mapping = []
        block_tables = None
        for seq in seqs:
            start = seq.num_cached_tokens
            seqlen_q = seq.num_scheduled_tokens
            end = start + seqlen_q
            seqlen_k = end
            input_ids.extend(seq[start:end])
            positions.extend(range(start, end))
            cu_seqlens_q.append(cu_seqlens_q[-1] + seqlen_q)
            cu_seqlens_k.append(cu_seqlens_k[-1] + seqlen_k)
            max_seqlen_q = max(seqlen_q, max_seqlen_q)
            max_seqlen_k = max(seqlen_k, max_seqlen_k)
            if not seq.block_table:    # warmup
                continue
            start_block = start // self.block_size
            end_block = (end + self.block_size - 1) // self.block_size
            for i in range(start_block, end_block):
                slot_start = seq.block_table[i] * self.block_size
                if i == start_block:
                    slot_start += start % self.block_size
                if i != end_block - 1:
                    slot_end = seq.block_table[i] * self.block_size + self.block_size
                else:
                    slot_end = seq.block_table[i] * self.block_size + end - i * self.block_size
                slot_mapping.extend(range(slot_start, slot_end))
        if cu_seqlens_k[-1] > cu_seqlens_q[-1]:    # prefix cache
            block_tables = self.prepare_block_tables(seqs)
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_q = torch.tensor(cu_seqlens_q, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_k = torch.tensor(cu_seqlens_k, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        set_context(True, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, slot_mapping, None, block_tables)
        return input_ids, positions

    def prepare_decode(self, seqs: list[Sequence]):
        if self.draft_model is not None:
            return self.prepare_decode_with_draft_model(seqs)
        input_ids = []
        inflight_rows = []
        positions = []
        slot_mapping = []
        context_lens = []
        for seq in seqs:
            # With async scheduling the newest token may still be on the GPU, in the last step's output.
            # With speculative decoding the draft tokens follow it, and all of them are scored at once.
            seq_len = len(seq) + seq.num_inflight
            input_ids.append(0 if seq.num_inflight else seq.last_token)
            input_ids.extend(seq.draft_token_ids)
            inflight_rows.append(seq.inflight_row if seq.num_inflight else -1)
            for position in range(seq_len - 1, seq_len + len(seq.draft_token_ids)):
                positions.append(position)
                slot_mapping.append(seq.block_table[position // self.block_size] * self.block_size + position % self.block_size)
            context_lens.append(positions[-1] + 1)
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        if any(row >= 0 for row in inflight_rows):
            inflight_rows = torch.tensor(inflight_rows, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
            input_ids = torch.where(inflight_rows >= 0, self.last_token_ids[inflight_rows.clamp_min(0)], input_ids)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        context_lens = torch.tensor(context_lens, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        block_tables = self.prepare_block_tables(seqs)
        set_context(False, slot_mapping=slot_mapping, context_lens=context_lens, block_tables=block_tables)
        return input_ids, positions

    def prepare_decode_with_draft_model(self, seqs: list[Sequence]):
        """Decode inputs for speculative decoding with a draft model, computed on the GPU: with
        async scheduling the number of tokens the in-flight step added, and so every position, is
        only known there. Also prepares the draft model's inputs (see propose)."""
        k = self.num_speculative_tokens
        block_size = self.block_size

        def to_cuda(values, dtype):
            return torch.tensor(values, dtype=dtype, pin_memory=True).cuda(non_blocking=True)

        num_tokens = to_cuda([len(seq) for seq in seqs], torch.int64)    # as known on the CPU
        last_token = to_cuda([seq.last_token for seq in seqs], torch.int64)
        prev_token = to_cuda([seq.token_ids[-2:][0] for seq in seqs], torch.int64)
        lags = to_cuda([seq.draft_cache_lags for seq in seqs], torch.bool)
        block_tables = self.prepare_block_tables(seqs)
        inflight_rows = [seq.inflight_row if seq.num_inflight else -1 for seq in seqs]
        if any(row >= 0 for row in inflight_rows):
            # The in-flight step's output: per sequence its accepted tokens, then -1.
            inflight_rows = to_cuda(inflight_rows, torch.int64)
            output = self.last_token_ids[inflight_rows.clamp_min(0)]
            num_inflight = torch.where(inflight_rows >= 0, (output >= 0).sum(dim=1), 0)
            num_tokens = num_tokens + num_inflight
            newest = output.gather(1, (num_inflight - 1).clamp_min(0).unsqueeze(1)).squeeze(1)
            second_newest = output.gather(1, (num_inflight - 2).clamp_min(0).unsqueeze(1)).squeeze(1)
            prev_token = torch.where(num_inflight >= 2, second_newest, torch.where(num_inflight == 1, last_token, prev_token))
            last_token = torch.where(num_inflight >= 1, newest, last_token)
            lags = torch.where(inflight_rows >= 0, num_inflight == k + 1, lags)
        # The target scores the newest token and k drafts (propose fills them in) at positions
        # num_tokens - 1 .. num_tokens + k - 1.
        positions = (num_tokens - 1).unsqueeze(1) + torch.arange(k + 1, device=num_tokens.device)
        slot_mapping = (block_tables.gather(1, positions // block_size) * block_size + positions % block_size).to(torch.int32)
        input_ids = torch.cat([last_token.unsqueeze(1), torch.zeros_like(positions[:, 1:])], dim=1)
        # The draft model first scores the last two tokens at positions num_tokens - 2 and
        # num_tokens - 1, then each draft it produces. The second-to-last token is usually in its
        # KV cache already; its slot is masked then so that no hashed block is written.
        draft_positions = positions - 1
        prev_slot = (block_tables.gather(1, draft_positions[:, :1] // block_size) * block_size + draft_positions[:, :1] % block_size).to(torch.int32)
        draft_slot_mapping = torch.cat([torch.where(lags.unsqueeze(1), prev_slot, -1), slot_mapping[:, :k]], dim=1)
        self.draft_inputs = torch.stack([prev_token, last_token], dim=1), draft_positions, draft_slot_mapping
        set_context(False, slot_mapping=slot_mapping.view(-1), context_lens=(num_tokens + k).to(torch.int32), block_tables=block_tables)
        return input_ids.view(-1), positions.view(-1)

    def prepare_sample(self, seqs: list[Sequence]):
        temperatures = [seq.temperature for seq in seqs]
        temperatures = torch.tensor(temperatures, dtype=torch.float32, pin_memory=True).cuda(non_blocking=True)
        return temperatures

    @torch.inference_mode()
    def forward(self, model: Qwen3ForCausalLM, graphs: dict | None, outputs: torch.Tensor | None, input_ids: torch.Tensor, positions: torch.Tensor, is_prefill: bool):
        """Hidden states of input_ids, through a captured CUDA graph when there is one for this
        decode batch size."""
        context = get_context()
        if is_prefill or self.enforce_eager or context.context_lens.size(0) > 512:
            return model(input_ids, positions)
        else:
            bs = context.context_lens.size(0)
            n = input_ids.size(0)    # bs * tokens per sequence
            graph = graphs[next(x for x in self.graph_bs if x >= bs)]
            graph_vars = self.graph_vars
            graph_vars["input_ids"][:n] = input_ids
            graph_vars["positions"][:n] = positions
            graph_vars["slot_mapping"].fill_(-1)
            graph_vars["slot_mapping"][:n] = context.slot_mapping
            graph_vars["context_lens"].zero_()
            graph_vars["context_lens"][:bs] = context.context_lens
            graph_vars["block_tables"][:bs, :context.block_tables.size(1)] = context.block_tables
            graph.replay()
            return outputs[:n]

    def run_model(self, input_ids: torch.Tensor, positions: torch.Tensor, is_prefill: bool):
        graphs, outputs = (self.graphs, self.graph_vars["outputs"]) if not self.enforce_eager else (None, None)
        return self.model.compute_logits(self.forward(self.model, graphs, outputs, input_ids, positions, is_prefill))

    @torch.inference_mode()
    def propose(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Run the draft model num_speculative_tokens times and write its greedy tokens into the
        draft columns of input_ids (one row of batch_size * (k + 1)). The proposals are
        deterministic given the context, so RejectionSampler stays exact."""
        k = self.num_speculative_tokens
        context = get_context()    # the target's
        context_lens, block_tables = context.context_lens, context.block_tables
        draft_input_ids, draft_positions, draft_slot_mapping = self.draft_inputs
        bs = draft_input_ids.size(0)
        input_ids = input_ids.view(bs, k + 1)
        tokens = draft_input_ids.view(-1)
        for i in range(k):
            cols = slice(0, 2) if i == 0 else slice(i + 1, i + 2)
            q = cols.stop - cols.start    # tokens per sequence in this draft step
            set_context(False, slot_mapping=draft_slot_mapping[:, cols].reshape(-1), context_lens=context_lens - k + i, block_tables=block_tables)
            graphs, outputs = (self.draft_graphs[q], self.graph_vars["draft_outputs"]) if not self.enforce_eager else (None, None)
            hidden_states = self.forward(self.draft_model, graphs, outputs, tokens, draft_positions[:, cols].reshape(-1), False)
            logits = self.draft_model.compute_logits(hidden_states[q - 1::q])
            tokens = logits.argmax(dim=-1) if self.rank == 0 else torch.empty(bs, dtype=torch.int64, device="cuda")
            if self.world_size > 1:
                dist.broadcast(tokens, 0)
            input_ids[:, i + 1] = tokens
        restore_context(context)
        return input_ids.view(-1)

    def run(self, seqs: list[Sequence], is_prefill: bool) -> list[int] | list[list[int]]:
        input_ids, positions = self.prepare_prefill(seqs) if is_prefill else self.prepare_decode(seqs)
        temperatures = self.prepare_sample(seqs) if self.rank == 0 else None
        if self.draft_model is not None and not is_prefill:
            input_ids = self.propose(input_ids)
        logits = self.run_model(input_ids, positions, is_prefill)
        if self.draft_model is not None and is_prefill:
            self.forward(self.draft_model, None, None, input_ids, positions, True)    # fills the draft KV cache
        token_ids = None
        if self.rank == 0 and (is_prefill or not self.num_speculative_tokens):
            token_ids = self.sampler(logits, temperatures).tolist()
        elif self.rank == 0:
            draft_token_ids = input_ids.view(len(seqs), -1)[:, 1:]
            token_ids = accepted_token_ids(self.rejection_sampler(logits, temperatures, draft_token_ids).tolist())
        reset_context()
        return token_ids

    def run_async(self, seqs: list[Sequence], is_prefill: bool) -> "AsyncOutput | None":
        """Like run, but returns before the GPU finishes. The sampled tokens stay on the GPU as
        inputs for the next decode step, and are copied to the host in the background."""
        k = self.num_speculative_tokens
        input_ids, positions = self.prepare_prefill(seqs) if is_prefill else self.prepare_decode(seqs)
        temperatures = self.prepare_sample(seqs) if self.rank == 0 else None
        if self.draft_model is not None and not is_prefill:
            input_ids = self.propose(input_ids)
        logits = self.run_model(input_ids, positions, is_prefill)
        if self.draft_model is not None and is_prefill:
            self.forward(self.draft_model, None, None, input_ids, positions, True)    # fills the draft KV cache
        if self.rank == 0 and (is_prefill or not k):
            token_ids = self.sampler(logits, temperatures)
        elif self.rank == 0:
            token_ids = self.rejection_sampler(logits, temperatures, input_ids.view(len(seqs), -1)[:, 1:])
        else:
            token_ids = torch.empty((len(seqs), k + 1) if k and not is_prefill else len(seqs), dtype=torch.int64, device="cuda")
        if self.world_size > 1:
            dist.broadcast(token_ids, 0)    # every rank needs them as next-step inputs
        # With speculative decoding every step's output has a row per sequence (see AsyncOutput).
        self.last_token_ids = token_ids if token_ids.dim() == 2 or not k else torch.nn.functional.pad(token_ids.unsqueeze(1), (0, k), value=-1)
        reset_context()
        return AsyncOutput(token_ids) if self.rank == 0 else None

    @torch.inference_mode()
    def capture_graphs(self, model: Qwen3ForCausalLM, outputs: torch.Tensor, q: int) -> dict[int, torch.cuda.CUDAGraph]:
        """One graph per batch size for decode steps of q tokens per sequence. All graphs share
        the input buffers in graph_vars and the memory pool; they are replayed one at a time."""
        graph_vars = self.graph_vars
        input_ids, positions = graph_vars["input_ids"], graph_vars["positions"]
        graphs = {}
        for bs in reversed(self.graph_bs):
            graph = torch.cuda.CUDAGraph()
            n = bs * q
            set_context(False, slot_mapping=graph_vars["slot_mapping"][:n], context_lens=graph_vars["context_lens"][:bs], block_tables=graph_vars["block_tables"][:bs])
            outputs[:n] = model(input_ids[:n], positions[:n])    # warmup
            with torch.cuda.graph(graph, self.graph_pool):
                outputs[:n] = model(input_ids[:n], positions[:n])    # capture
            if self.graph_pool is None:
                self.graph_pool = graph.pool()
            graphs[bs] = graph
            torch.cuda.synchronize()
            reset_context()
        return graphs

    def capture_cudagraph(self):
        config = self.config
        hf_config = config.hf_config
        max_bs = min(self.config.max_num_seqs, 512)
        q = 1 + self.num_speculative_tokens    # tokens per sequence in a decode step
        max_num_blocks = (config.max_model_len + q - 1 + self.block_size - 1) // self.block_size
        self.graph_bs = [1, 2, 4, 8] + list(range(16, max_bs + 1, 16))
        self.graph_pool = None
        self.graph_vars = dict(
            input_ids=torch.zeros(max_bs * q, dtype=torch.int64),
            positions=torch.zeros(max_bs * q, dtype=torch.int64),
            slot_mapping=torch.zeros(max_bs * q, dtype=torch.int32),
            context_lens=torch.zeros(max_bs, dtype=torch.int32),
            block_tables=torch.zeros(max_bs, max_num_blocks, dtype=torch.int32),
            outputs=torch.zeros(max_bs * q, hf_config.hidden_size),
        )
        self.graphs = self.capture_graphs(self.model, self.graph_vars["outputs"], q)
        self.draft_graphs = {}
        if self.draft_model is not None:
            # The first draft step scores two tokens per sequence, the others one (see propose).
            self.graph_vars["draft_outputs"] = torch.zeros(max_bs * 2, config.draft_hf_config.hidden_size)
            for q in (2, 1):
                self.draft_graphs[q] = self.capture_graphs(self.draft_model, self.graph_vars["draft_outputs"], q)
