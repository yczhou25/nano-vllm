from nanovllm.engine.block_manager import BlockManager
from nanovllm.engine.sequence import Sequence

BLOCK_SIZE = 4


def prefill(block_manager, seq):
    num_cached_blocks = block_manager.can_allocate(seq)
    assert num_cached_blocks != -1
    block_manager.allocate(seq, num_cached_blocks)
    block_manager.hash_blocks(seq, seq.num_cached_tokens, len(seq) - seq.num_cached_tokens)
    seq.num_cached_tokens = len(seq)
    return num_cached_blocks


def test_shared_prefix_reuses_blocks():
    Sequence.block_size = BLOCK_SIZE
    bm = BlockManager(8, BLOCK_SIZE)
    a = Sequence(list(range(10)))
    b = Sequence(list(range(8)) + [100, 101, 102])
    assert prefill(bm, a) == 0
    assert prefill(bm, b) == 2
    assert b.block_table[:2] == a.block_table[:2]
    assert b.block_table[2] != a.block_table[2]
    assert [bm.blocks[i].ref_count for i in a.block_table] == [2, 2, 1]
    assert b.num_cached_tokens == len(b)


def test_hashes_chain_on_prefix():
    # Same second block after a different first block must not hit the cache.
    Sequence.block_size = BLOCK_SIZE
    bm = BlockManager(8, BLOCK_SIZE)
    prefill(bm, Sequence([1, 2, 3, 4, 5, 6, 7, 8, 9]))
    assert bm.can_allocate(Sequence([0, 0, 0, 0, 5, 6, 7, 8, 9])) == 0


def test_last_block_is_never_a_cache_hit():
    # At least one prompt token has to be computed to get logits for the first output token.
    Sequence.block_size = BLOCK_SIZE
    bm = BlockManager(8, BLOCK_SIZE)
    prefill(bm, Sequence(list(range(8))))
    assert bm.can_allocate(Sequence(list(range(8)))) == 1


def test_freed_blocks_stay_cached_until_reused():
    Sequence.block_size = BLOCK_SIZE
    bm = BlockManager(3, BLOCK_SIZE)
    a = Sequence(list(range(9)))
    prefill(bm, a)
    cached = a.block_table[:2]
    bm.deallocate(a)
    assert len(bm.free_block_ids) == 3
    b = Sequence(list(range(8)) + [50])
    assert prefill(bm, b) == 2
    assert b.block_table[:2] == cached
    bm.deallocate(b)
    # Reusing every block for unrelated tokens evicts the old hashes.
    c = Sequence([7] * 12)
    assert prefill(bm, c) == 0
    bm.deallocate(c)
    assert bm.can_allocate(Sequence(list(range(8)) + [50])) == 0


def test_cached_free_blocks_count_against_free_space():
    # A cache hit on a free block still takes that block out of the free list.
    Sequence.block_size = BLOCK_SIZE
    bm = BlockManager(3, BLOCK_SIZE)
    a = Sequence(list(range(9)))
    prefill(bm, a)
    bm.deallocate(a)
    seq = Sequence(list(range(8)) + [1] * 4)
    assert bm.can_allocate(seq) == 2
    prefill(bm, Sequence([9]))          # takes the one free block that holds no cached tokens
    assert bm.can_allocate(seq) == -1   # 2 cached + 1 new blocks, but only 2 free


def test_hash_collision_is_not_a_cache_hit(monkeypatch):
    monkeypatch.setattr(BlockManager, "compute_hash", classmethod(lambda cls, token_ids, prefix=-1: 42))
    Sequence.block_size = BLOCK_SIZE
    bm = BlockManager(8, BLOCK_SIZE)
    prefill(bm, Sequence(list(range(9))))
    assert bm.can_allocate(Sequence(list(range(100, 109)))) == 0
