import os

import torch
from torch import nn


def _chunks(vocab_size: int) -> int:
    """How many chunks chunked_argmax and chunked_logsumexp cut a vocabulary row into. A chunk is
    the largest power of two up to 1024 that divides the vocabulary (Qwen3's 151936 = 2^7 * 1187:
    128-element chunks, 1187 of them): at most 1024 so that Inductor reduces a chunk as a
    persistent reduction, the chunk in registers and read once for every reduction over it (a max
    and its index, the log-sum-exp's max and sum); A40 job 279567 measured 128-wide chunks of a
    looped reduction (1187 elements, two passes over the logits) no faster than the one-block rows
    at batch 32. A vocabulary that gives chunks under 8 elements is not chunked (1 chunk)."""
    chunk = 1024
    while chunk > 1 and vocab_size % chunk:
        chunk //= 2
    return vocab_size // chunk if chunk >= 8 else 1


def chunked_argmax(x: torch.Tensor) -> torch.Tensor:
    """x.argmax(dim=-1) in two stages: the max and its index per chunk of the last axis, then the
    chunk with the largest max. Same result (ties go to the first index), but a very different
    kernel: Inductor never splits a reduction with an argmax across thread blocks
    (torch/_inductor/ir.py, Reduction.num_splits), so a one-stage argmax over a 151936-wide row
    runs one block per row; a batch of 32 kept 32 of an A40's 84 SMs busy and took 183 us per
    call, a single sequence one SM (A40 job 278958). The first stage here has rows * chunks
    outputs to spread over the GPU and the second reduces the chunk maxima of each row."""
    vocab_size = x.size(-1)
    chunks = _chunks(vocab_size)
    if chunks == 1:
        return x.argmax(dim=-1)
    values, index = x.view(*x.shape[:-1], chunks, vocab_size // chunks).max(dim=-1)
    best = values.argmax(dim=-1, keepdim=True)
    return (index.gather(-1, best) + best * (vocab_size // chunks)).squeeze(-1)


def chunked_logsumexp(x: torch.Tensor) -> torch.Tensor:
    """torch.logsumexp(x, dim=-1) by chunk of the last axis, like chunked_argmax, so that Inductor
    fuses it with the argmax over the same rows in one pass over them."""
    vocab_size = x.size(-1)
    chunks = _chunks(vocab_size)
    if chunks == 1:
        return torch.logsumexp(x, dim=-1)
    return torch.logsumexp(torch.logsumexp(x.view(*x.shape[:-1], chunks, vocab_size // chunks), dim=-1), dim=-1)


def exponential_like(x: torch.Tensor) -> torch.Tensor:
    """Exp(1) samples in the shape of x, as -log(U) of uniform samples: Inductor generates
    torch.rand inside the kernel that consumes it, while exponential_ is one of its fallbacks,
    an eager kernel whose samples go through memory (a 19 MB tensor for 32 rows of 151936)."""
    return torch.rand_like(x).clamp_min_(1e-20).log_().neg_()


try:
    import triton
    import triton.language as tl
except ImportError:    # CPU-only install: the samplers keep their compiled torch path
    triton = None


if triton is not None:

    @triton.jit(do_not_specialize=["seed"])    # Triton specializes integer arguments on == 1 and % 16 == 0: a seed must not recompile
    def _sample_tiles_kernel(logits_ptr, temperatures_ptr, greedy_max_ptr, greedy_index_ptr, scaled_max_ptr,
                             scaled_sum_ptr, gumbel_max_ptr, gumbel_index_ptr, vocab_size, tiles, seed,
                             BLOCK: tl.constexpr):
        """Stage 1 of fused_sample, one program per (row, tile of BLOCK logits): the tile's max and
        its index (the greedy token), the max and the sum of exp of the scaled logits (its log-sum-exp
        as two numbers) and the max and index of the scaled logits plus Gumbel noise (the sample), all
        from one read of the tile; the noise is generated in registers (tl.rand, a Philox counter from
        the element's position and the call's seed) and never stored."""
        row = tl.program_id(0)
        tile = tl.program_id(1)
        offsets = tile * BLOCK + tl.arange(0, BLOCK)
        mask = offsets < vocab_size
        logits = tl.load(logits_ptr + row * vocab_size + offsets, mask=mask, other=float("-inf")).to(tl.float32)
        greedy_max, greedy_index = tl.max(logits, axis=0, return_indices=True)
        temperature = tl.maximum(tl.load(temperatures_ptr + row), 1e-10)
        scaled = logits / temperature
        scaled_max = tl.max(scaled, axis=0)
        scaled_sum = tl.sum(tl.exp(scaled - tl.where(scaled_max == float("-inf"), 0.0, scaled_max)), axis=0)
        # One Philox call gives four uniforms (tl.rand keeps one of them and was ~half of this kernel, A40 job 996212):
        # the tile's four quarters take u0..u3 of the same BLOCK / 4 counters, which no other tile or row uses.
        u0, u1, u2, u3 = tl.rand4x(seed, row * vocab_size + tile * BLOCK + tl.arange(0, BLOCK // 4))
        uniform = tl.reshape(tl.join(tl.join(u0, u1), tl.join(u2, u3)), [BLOCK])
        gumbel = -tl.log(-tl.log(tl.maximum(uniform, 1e-20)))
        gumbel_max, gumbel_index = tl.max(tl.where(mask, scaled + gumbel, float("-inf")), axis=0, return_indices=True)
        partial = row * tiles + tile
        tl.store(greedy_max_ptr + partial, greedy_max)
        tl.store(greedy_index_ptr + partial, greedy_index + tile * BLOCK)
        tl.store(scaled_max_ptr + partial, scaled_max)
        tl.store(scaled_sum_ptr + partial, scaled_sum)
        tl.store(gumbel_max_ptr + partial, gumbel_max)
        tl.store(gumbel_index_ptr + partial, gumbel_index + tile * BLOCK)

    @triton.jit
    def _sample_rows_kernel(temperatures_ptr, greedy_max_ptr, greedy_index_ptr, scaled_max_ptr, scaled_sum_ptr,
                            gumbel_max_ptr, gumbel_index_ptr, tokens_ptr, log_norms_ptr, tiles, BLOCK: tl.constexpr):
        """Stage 2 of fused_sample, one program per row: the greedy token from the first tile with the
        largest max (ties go to the first index, as in torch.argmax), the row's log-sum-exp from the
        tiles' maxima and sums, the sampled token from the tile with the largest noisy max, and the
        greedy token where the temperature is 0."""
        row = tl.program_id(0)
        offsets = tl.arange(0, BLOCK)
        mask = offsets < tiles
        partials = row * tiles + offsets
        greedy_max = tl.load(greedy_max_ptr + partials, mask=mask, other=float("-inf"))
        _, greedy_tile = tl.max(greedy_max, axis=0, return_indices=True)
        greedy_token = tl.load(greedy_index_ptr + row * tiles + greedy_tile)
        scaled_max = tl.load(scaled_max_ptr + partials, mask=mask, other=float("-inf"))
        scaled_sum = tl.load(scaled_sum_ptr + partials, mask=mask, other=0.0)
        row_max = tl.max(scaled_max, axis=0)
        safe_max = tl.where(row_max == float("-inf"), 0.0, row_max)
        log_norm = safe_max + tl.log(tl.sum(scaled_sum * tl.exp(scaled_max - safe_max), axis=0))
        gumbel_max = tl.load(gumbel_max_ptr + partials, mask=mask, other=float("-inf"))
        _, gumbel_tile = tl.max(gumbel_max, axis=0, return_indices=True)
        sample_token = tl.load(gumbel_index_ptr + row * tiles + gumbel_tile)
        token = tl.where(tl.load(temperatures_ptr + row) == 0.0, greedy_token, sample_token)
        tl.store(tokens_ptr + row, token.to(tl.int64))
        tl.store(log_norms_ptr + row, log_norm)


# Tile width and warps per program of fused_sample's first kernel; the environment overrides are for bench_sampler.py.
SAMPLE_BLOCK = int(os.environ.get("NANOVLLM_SAMPLE_BLOCK", "1024"))
SAMPLE_WARPS = int(os.environ.get("NANOVLLM_SAMPLE_WARPS", "4"))


def fused_sample(logits: torch.Tensor, temperatures: torch.Tensor, block: int = SAMPLE_BLOCK, num_warps: int = SAMPLE_WARPS):
    """What Sampler and DraftSampler compute, as two Triton kernels: a sample from
    softmax(logits / temperature) per row by the Gumbel-max trick (the argmax where the temperature
    is 0) and the log-sum-exp of logits / temperature, (batch_size,) int64 tokens and (batch_size,)
    float32 log-norms, for logits (batch_size, vocab_size) of any float dtype.

    The compiled torch version (Sampler/DraftSampler's CPU path) is bound by Inductor's choices:
    it never splits a reduction with an argmax across thread blocks, it materializes every random
    tensor (19 MB of noise for 32 rows of 151936, read back by the argmax), and whether the chunked
    reductions of 62ddc96 came out as one persistent pass or a looped one was not under our control
    (A40 jobs 278958, 279567, 281207: 164-207 us of GPU time per call at batch 32 in all three
    versions). Here every (row, tile of `block` logits) is its own program, the tile is read once
    into registers and reduced three ways there, the noise is generated in registers, and a second
    kernel combines the batch_size * tiles partial results: 6 numbers per tile cross memory instead
    of the logits twice and the noise. A40 job 996212 (1024-wide tiles, 4 warps): 64 us of GPU time
    per call at batch 32 for either sampler, against 167 / 207 us compiled (6.9 us at batch 1); the
    first kernel ran at 160 GB/s, instruction-bound, about half of it the Philox generator."""
    batch_size, vocab_size = logits.shape
    tiles = triton.cdiv(vocab_size, block)
    partials = [torch.empty(batch_size, tiles, device=logits.device, dtype=dtype)
                for dtype in (torch.float32, torch.int32, torch.float32, torch.float32, torch.float32, torch.int32)]
    tokens = torch.empty(batch_size, device=logits.device, dtype=torch.int64)
    log_norms = torch.empty(batch_size, device=logits.device, dtype=torch.float32)
    seed = int(torch.randint(0, 2**31 - 1, (), dtype=torch.int64))    # from torch's CPU generator, so torch.manual_seed fixes the samples
    _sample_tiles_kernel[(batch_size, tiles)](logits.contiguous(), temperatures, *partials, vocab_size, tiles, seed, BLOCK=block, num_warps=num_warps)
    _sample_rows_kernel[(batch_size,)](temperatures, *partials, tokens, log_norms, tiles, BLOCK=triton.next_power_of_2(tiles))
    return tokens, log_norms


class Sampler(nn.Module):
    """A sample from softmax(logits / temperature) per row, by the Gumbel-max trick on the logits
    (argmax(logits / T - log E), E ~ Exp(1): no softmax); the argmax where the temperature is 0.
    On the GPU the Triton kernels of fused_sample; elsewhere the compiled torch version below."""

    def forward(self, logits: torch.Tensor, temperatures: torch.Tensor):
        if logits.is_cuda and triton is not None:
            return fused_sample(logits, temperatures)[0]
        return self.torch_forward(logits, temperatures)

    # Compiled once for symbolic shapes: the batch size changes with every tail batch, and with adaptive
    # speculation the number of tokens per sequence too; a recompile per shape costs seconds each.
    @torch.compile(dynamic=True)
    def torch_forward(self, logits: torch.Tensor, temperatures: torch.Tensor):
        greedy_tokens = chunked_argmax(logits)
        scaled = logits.float() / temperatures.clamp_min(1e-10).unsqueeze(dim=1)
        sample_tokens = chunked_argmax(scaled - exponential_like(scaled).log_())
        return torch.where(temperatures == 0, greedy_tokens, sample_tokens)


class RejectionSampler(nn.Module):
    """Verifies draft tokens against the target model (speculative decoding with deterministic
    proposals, such as n-gram lookup). Draft i is accepted with probability p_i(draft i). At the
    first rejection a token is sampled from p_i without the rejected draft; if every draft is
    accepted, a bonus token is sampled from the last position. The emitted tokens follow exactly
    the distribution that sampling one token at a time would give. Greedy rows (temperature 0)
    keep drafts that match the argmax."""

    @torch.compile(dynamic=True)
    def forward(self, logits: torch.Tensor, temperatures: torch.Tensor, draft_token_ids: torch.Tensor,
                token_index: torch.Tensor | None = None, num_drafts: torch.Tensor | None = None,
                draft_logits: torch.Tensor | None = None, draft_log_norms: torch.Tensor | None = None):
        """logits: (batch_size * (k + 1), vocab_size); draft_token_ids: (batch_size, k).
        Returns (batch_size, k + 1) token ids: the accepted drafts, one sampled token, then -1.

        A ragged batch, whose sequences have different numbers of drafts (see
        ModelRunner.ragged_sampler_inputs), passes logits of sum(1 + k_i) rows, token_index
        (batch_size, K + 1) with the row of each sequence's input token and of its drafts, padded
        by repeating its last row, draft_token_ids padded the same way, and num_drafts (batch_size,)
        with each k_i: the padded positions count as rejected.

        draft_logits (batch_size, k, vocab_size): the draft model's logits each draft was sampled
        from, in the model's dtype, and draft_log_norms (batch_size, k): their log-sum-exp at the
        temperature, both from DraftSampler; q_j = exp(draft_logits_j / temperature - log_norm_j)
        is then a pointwise read of the logits here (no reduction over them in this graph: the
        k=1 graph of a logsumexp over them tripped an Inductor fusion assertion on the GPU, A40
        job 245130), for the exact speculative sampling of Leviathan et al.: draft j is accepted with probability min(1, p(d_j) / q_j(d_j)), and the token after
        a rejected draft is drawn from the residual norm(max(0, p - q_j)). None means one-hot
        drafts (the argmax of the draft model, the n-gram proposer): accept with p(d_j), and the
        residual is p with d_j removed. Greedy rows (temperature 0) accept a draft only if it is
        the target's argmax, either way."""
        batch_size, k = draft_token_ids.shape
        if token_index is None:
            logits = logits.view(batch_size, k + 1, -1)
            num_drafts = k
        else:
            logits = logits[token_index]
        greedy = temperatures == 0
        temperatures = temperatures.clamp_min(1e-10).unsqueeze(dim=1)
        target_at_drafts = logits[:, :k].gather(-1, draft_token_ids.unsqueeze(-1)).squeeze(-1).float()
        log_norms = chunked_logsumexp(logits[:, :k].float() / temperatures.unsqueeze(dim=-1))
        log_target_probs = target_at_drafts / temperatures - log_norms    # log p(d_j)
        if draft_logits is not None:
            draft_at_drafts = draft_logits.gather(-1, draft_token_ids.unsqueeze(-1)).squeeze(-1).float()
            accept_probs = torch.exp(log_target_probs - (draft_at_drafts / temperatures - draft_log_norms))    # p / q
        else:
            accept_probs = torch.exp(log_target_probs)
        accepted = torch.where(greedy.unsqueeze(dim=1), chunked_argmax(logits[:, :k]) == draft_token_ids,
                               torch.rand_like(accept_probs) < accept_probs)
        if token_index is not None:
            accepted &= torch.arange(k, device=logits.device) < num_drafts.unsqueeze(dim=1)
        num_accepted = accepted.int().cumprod(dim=1).sum(dim=1)
        # Sample the token after the accepted drafts: from the residual after a rejected draft (one-hot q: p with the draft removed).
        rows = torch.arange(batch_size, device=logits.device)
        last_logits = logits[rows, num_accepted].float()
        rejected_at = num_accepted.clamp_max(k - 1)
        rejected = draft_token_ids.gather(1, rejected_at.unsqueeze(dim=1))
        was_rejected = (num_accepted < num_drafts).unsqueeze(dim=1)    # false after every draft was accepted (or a padded position)
        keep = last_logits.gather(1, rejected)
        masked_logits = last_logits.scatter(1, rejected, torch.where(was_rejected, float("-inf"), keep))
        greedy_tokens = chunked_argmax(masked_logits)
        if draft_logits is None:
            probs = torch.softmax(masked_logits.div_(temperatures), dim=-1)
        else:
            probs = torch.softmax(last_logits / temperatures, dim=-1)
            q_row = torch.exp(draft_logits[rows, rejected_at].float() / temperatures - draft_log_norms[rows, rejected_at].unsqueeze(dim=1))    # q_j
            residual = (probs - q_row).clamp_min(0)
            norm = residual.sum(dim=-1, keepdim=True)
            probs = torch.where(was_rejected & (norm > 0), residual / norm, probs)
        sample_tokens = chunked_argmax(probs.div_(exponential_like(probs)))
        last_tokens = torch.where(greedy, greedy_tokens, sample_tokens)
        token_ids = torch.cat([draft_token_ids, draft_token_ids[:, :1]], dim=1)
        token_ids.scatter_(1, num_accepted.unsqueeze(dim=1), last_tokens.unsqueeze(dim=1))
        return token_ids.masked_fill_(torch.arange(k + 1, device=logits.device) > num_accepted.unsqueeze(dim=1), -1)


class DraftSampler(nn.Module):
    """A draft model's proposals from its logits (batch_size, vocab_size): a sample from
    softmax(logits / temperature) per sequence, drawn by the Gumbel-max trick on the logits
    themselves (argmax(logits / T - log E), E ~ Exp(1)), so no softmax is computed or stored,
    and the log-sum-exp of logits / T (batch_size,), with which RejectionSampler reads q from
    the same logits. A greedy sequence (temperature 0) drafts the argmax (the sampler does not
    read its q). It runs k times per step, and 4 calls of 183 us each at batch 32 were most of
    what sampled drafts cost per step (A40 job 278958), so on the GPU it is the Triton kernels of
    fused_sample; elsewhere the compiled torch version below, with chunked reductions."""

    def forward(self, logits: torch.Tensor, temperatures: torch.Tensor):
        if logits.is_cuda and triton is not None:
            return fused_sample(logits, temperatures)
        return self.torch_forward(logits, temperatures)

    @torch.compile(dynamic=True)
    def torch_forward(self, logits: torch.Tensor, temperatures: torch.Tensor):
        greedy_tokens = chunked_argmax(logits)
        scaled = logits.float() / temperatures.clamp_min(1e-10).unsqueeze(dim=1)
        log_norms = chunked_logsumexp(scaled)
        sample_tokens = chunked_argmax(scaled - exponential_like(scaled).log_())
        return torch.where(temperatures == 0, greedy_tokens, sample_tokens), log_norms


def accepted_token_ids(token_ids: list[list[int]]) -> list[list[int]]:
    """Strip the -1 padding from RejectionSampler's output (after .tolist())."""
    return [row[:row.index(-1)] if row[-1] == -1 else row for row in token_ids]
