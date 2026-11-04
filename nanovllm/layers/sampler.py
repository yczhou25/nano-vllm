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
                token_index: torch.Tensor | None = None, num_drafts: torch.Tensor | None = None,
                draft_logits: torch.Tensor | None = None):
        """logits: (batch_size * (k + 1), vocab_size); draft_token_ids: (batch_size, k).
        Returns (batch_size, k + 1) token ids: the accepted drafts, one sampled token, then -1.

        A ragged batch, whose sequences have different numbers of drafts (see
        ModelRunner.ragged_sampler_inputs), passes logits of sum(1 + k_i) rows, token_index
        (batch_size, K + 1) with the row of each sequence's input token and of its drafts, padded
        by repeating its last row, draft_token_ids padded the same way, and num_drafts (batch_size,)
        with each k_i: the padded positions count as rejected.

        draft_logits (batch_size, k, vocab_size): the draft model's logits each draft was sampled
        from (DraftSampler), in the model's dtype; q_j = softmax(draft_logits_j / temperature) is
        computed here, inside the compiled graph, for the exact speculative sampling of Leviathan
        et al.: draft j is accepted with probability min(1, p(d_j) / q_j(d_j)), and the token after
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
        log_norms = torch.logsumexp(logits[:, :k].float() / temperatures.unsqueeze(dim=-1), dim=-1)
        log_target_probs = target_at_drafts / temperatures - log_norms    # log p(d_j)
        if draft_logits is not None:
            draft_at_drafts = draft_logits.gather(-1, draft_token_ids.unsqueeze(-1)).squeeze(-1).float()
            q_log_norms = torch.logsumexp(draft_logits.float() / temperatures.unsqueeze(dim=-1), dim=-1)
            accept_probs = torch.exp(log_target_probs - (draft_at_drafts / temperatures - q_log_norms))    # p / q
        else:
            accept_probs = torch.exp(log_target_probs)
        accepted = torch.where(greedy.unsqueeze(dim=1), logits[:, :k].argmax(dim=-1) == draft_token_ids,
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
        greedy_tokens = masked_logits.argmax(dim=-1)
        if draft_logits is None:
            probs = torch.softmax(masked_logits.div_(temperatures), dim=-1)
        else:
            probs = torch.softmax(last_logits / temperatures, dim=-1)
            residual = (probs - torch.softmax(draft_logits[rows, rejected_at].float() / temperatures, dim=-1)).clamp_min(0)
            norm = residual.sum(dim=-1, keepdim=True)
            probs = torch.where(was_rejected & (norm > 0), residual / norm, probs)
        sample_tokens = probs.div_(torch.empty_like(probs).exponential_(1).clamp_min_(1e-10)).argmax(dim=-1)
        last_tokens = torch.where(greedy, greedy_tokens, sample_tokens)
        token_ids = torch.cat([draft_token_ids, draft_token_ids[:, :1]], dim=1)
        token_ids.scatter_(1, num_accepted.unsqueeze(dim=1), last_tokens.unsqueeze(dim=1))
        return token_ids.masked_fill_(torch.arange(k + 1, device=logits.device) > num_accepted.unsqueeze(dim=1), -1)


class DraftSampler(nn.Module):
    """A draft model's proposals from its logits (batch_size, vocab_size): a sample from
    softmax(logits / temperature) per sequence, drawn by the Gumbel-max trick on the logits
    themselves (argmax(logits / T - log E), E ~ Exp(1)), so no softmax is computed or stored;
    RejectionSampler reads q from the same logits. A greedy sequence (temperature 0) drafts the
    argmax (the sampler does not read its q). Compiled like the samplers, for the same reason."""

    @torch.compile(dynamic=True)
    def forward(self, logits: torch.Tensor, temperatures: torch.Tensor):
        greedy_tokens = logits.argmax(dim=-1)
        scaled = logits.float() / temperatures.clamp_min(1e-10).unsqueeze(dim=1)
        sample_tokens = scaled.sub_(torch.empty_like(scaled).exponential_(1).clamp_min_(1e-10).log_()).argmax(dim=-1)
        return torch.where(temperatures == 0, greedy_tokens, sample_tokens)


def accepted_token_ids(token_ids: list[list[int]]) -> list[list[int]]:
    """Strip the -1 padding from RejectionSampler's output (after .tolist())."""
    return [row[:row.index(-1)] if row[-1] == -1 else row for row in token_ids]
