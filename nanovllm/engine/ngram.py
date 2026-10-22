from nanovllm.engine.sequence import Sequence


class NgramProposer:
    """Draft tokens by prompt lookup: find the latest earlier occurrence of the sequence's last n
    tokens (n from max_n down to 1) and propose the tokens that followed it. If the match runs
    into the end of the sequence, the repeating pattern is continued."""

    def __init__(self, num_tokens: int, max_n: int = 4):
        self.num_tokens = num_tokens
        self.max_n = max_n
        # seq_id -> [number of indexed positions, {n-gram: position right after its latest occurrence}]
        self.index: dict[int, list] = {}

    def propose(self, seq: Sequence) -> list[int]:
        token_ids = seq.token_ids
        num_tokens = len(token_ids)
        state = self.index.setdefault(seq.seq_id, [1, {}])
        num_indexed, table = state
        # Index the n-grams ending before the last token, so a lookup never finds the suffix itself.
        for c in range(num_indexed, num_tokens):
            for n in range(1, min(self.max_n, c) + 1):
                table[tuple(token_ids[c - n:c])] = c
        state[0] = max(num_indexed, num_tokens)
        for n in range(min(self.max_n, num_tokens - 1), 0, -1):
            c = table.get(tuple(token_ids[num_tokens - n:]))
            if c is not None:
                break
        else:
            return [token_ids[-1]] * self.num_tokens
        drafts = token_ids[c:c + self.num_tokens]
        period = num_tokens - c
        while len(drafts) < self.num_tokens:
            drafts.append(drafts[len(drafts) - period])
        return drafts

    def forget(self, seq: Sequence):
        self.index.pop(seq.seq_id, None)
