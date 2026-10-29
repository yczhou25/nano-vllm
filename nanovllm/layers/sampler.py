import torch
from torch import nn


class Sampler(nn.Module):

    # Compiled once for symbolic shapes: the batch size changes with every tail batch, and with adaptive
    # speculation the number of tokens per sequence too; a recompile per shape costs seconds each.
    @torch.compile(dynamic=True)
    def forward(self, logits: torch.Tensor, temperatures: torch.Tensor):
        greedy_tokens = logits.argmax(dim=-1)
        logits = logits.float().div_(temperatures.clamp_min(1e-10).unsqueeze(dim=1))
        probs = torch.softmax(logits, dim=-1)
        sample_tokens = probs.div_(torch.empty_like(probs).exponential_(1).clamp_min_(1e-10)).argmax(dim=-1)
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
                token_index: torch.Tensor | None = None, num_drafts: torch.Tensor | None = None):
        """logits: (batch_size * (k + 1), vocab_size); draft_token_ids: (batch_size, k).
        Returns (batch_size, k + 1) token ids: the accepted drafts, one sampled token, then -1.

        A ragged batch, whose sequences have different numbers of drafts (see
        ModelRunner.ragged_sampler_inputs), passes logits of sum(1 + k_i) rows, token_index
        (batch_size, K + 1) with the row of each sequence's input token and of its drafts, padded
        by repeating its last row, draft_token_ids padded the same way, and num_drafts (batch_size,)
        with each k_i: the padded positions count as rejected."""
        batch_size, k = draft_token_ids.shape
        if token_index is None:
            logits = logits.view(batch_size, k + 1, -1)
            num_drafts = k
        else:
            logits = logits[token_index]
        greedy = temperatures == 0
        temperatures = temperatures.clamp_min(1e-10).unsqueeze(dim=1)
        draft_logits = logits[:, :k].gather(-1, draft_token_ids.unsqueeze(-1)).squeeze(-1).float()
        log_norms = torch.logsumexp(logits[:, :k].float() / temperatures.unsqueeze(dim=-1), dim=-1)
        draft_probs = torch.exp(draft_logits / temperatures - log_norms)
        accepted = torch.where(greedy.unsqueeze(dim=1), logits[:, :k].argmax(dim=-1) == draft_token_ids,
                               torch.rand_like(draft_probs) < draft_probs)
        if token_index is not None:
            accepted &= torch.arange(k, device=logits.device) < num_drafts.unsqueeze(dim=1)
        num_accepted = accepted.int().cumprod(dim=1).sum(dim=1)
        # Sample the token after the accepted drafts, with the rejected draft (if any) removed.
        rows = torch.arange(batch_size, device=logits.device)
        last_logits = logits[rows, num_accepted].float()
        rejected = draft_token_ids.gather(1, num_accepted.clamp_max(k - 1).unsqueeze(dim=1))
        keep = last_logits.gather(1, rejected)
        last_logits.scatter_(1, rejected, torch.where((num_accepted < num_drafts).unsqueeze(dim=1), float("-inf"), keep))
        greedy_tokens = last_logits.argmax(dim=-1)
        probs = torch.softmax(last_logits.div_(temperatures), dim=-1)
        sample_tokens = probs.div_(torch.empty_like(probs).exponential_(1).clamp_min_(1e-10)).argmax(dim=-1)
        last_tokens = torch.where(greedy, greedy_tokens, sample_tokens)
        token_ids = torch.cat([draft_token_ids, draft_token_ids[:, :1]], dim=1)
        token_ids.scatter_(1, num_accepted.unsqueeze(dim=1), last_tokens.unsqueeze(dim=1))
        return token_ids.masked_fill_(torch.arange(k + 1, device=logits.device) > num_accepted.unsqueeze(dim=1), -1)


def accepted_token_ids(token_ids: list[list[int]]) -> list[list[int]]:
    """Strip the -1 padding from RejectionSampler's output (after .tolist())."""
    return [row[:row.index(-1)] if row[-1] == -1 else row for row in token_ids]
