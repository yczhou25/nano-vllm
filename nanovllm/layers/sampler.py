import torch
from torch import nn


def _chunks(vocab_size: int) -> int:
    """How many equal chunks chunked_argmax and chunked_logsumexp cut a vocabulary row into: the
    largest power of two up to 128 that divides it (1 when it is odd)."""
    chunks = 128
    while chunks > 1 and vocab_size % chunks:
        chunks //= 2
    return chunks


def chunked_argmax(x: torch.Tensor) -> torch.Tensor:
    """x.argmax(dim=-1) in two stages: the max and its index per chunk of the last axis, then the
    chunk with the largest max. Same result (ties go to the first index), but a very different
    kernel: Inductor never splits a reduction with an argmax across thread blocks
    (torch/_inductor/ir.py, Reduction.num_splits), so a one-stage argmax over a 151936-wide row
    runs one block per row; a batch of 32 kept 32 of an A40's 84 SMs busy and took 183 us per
    call, a single sequence one SM (A40 job 278958). The first stage here has rows * 128 outputs
    to spread over the GPU and the second reduces 128 values per row."""
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


class Sampler(nn.Module):

    # Compiled once for symbolic shapes: the batch size changes with every tail batch, and with adaptive
    # speculation the number of tokens per sequence too; a recompile per shape costs seconds each.
    @torch.compile(dynamic=True)
    def forward(self, logits: torch.Tensor, temperatures: torch.Tensor):
        """A sample from softmax(logits / temperature) per row, by the Gumbel-max trick on the
        logits (argmax(logits / T - log E), E ~ Exp(1): no softmax); the argmax where the
        temperature is 0."""
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
    read its q). Compiled like the samplers, for the same reason; its reductions are chunked
    (chunked_argmax) since it runs k times per step: 4 calls of 183 us each at batch 32 were most
    of what sampled drafts cost per step (A40 job 278958)."""

    @torch.compile(dynamic=True)
    def forward(self, logits: torch.Tensor, temperatures: torch.Tensor):
        greedy_tokens = chunked_argmax(logits)
        scaled = logits.float() / temperatures.clamp_min(1e-10).unsqueeze(dim=1)
        log_norms = chunked_logsumexp(scaled)
        sample_tokens = chunked_argmax(scaled - exponential_like(scaled).log_())
        return torch.where(temperatures == 0, greedy_tokens, sample_tokens), log_norms


def accepted_token_ids(token_ids: list[list[int]]) -> list[list[int]]:
    """Strip the -1 padding from RejectionSampler's output (after .tolist())."""
    return [row[:row.index(-1)] if row[-1] == -1 else row for row in token_ids]
