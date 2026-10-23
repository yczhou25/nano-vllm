import os

import pytest

from engine_sim import run_random_case, run_random_case_block_256

NUM_CASES = int(os.environ.get("NANOVLLM_SIM_CASES", "300"))
# (async_scheduling, num_speculative_tokens, draft_model)
MODES = {"sync": (False, 0, False), "async": (True, 0, False), "spec1": (False, 1, False), "spec4": (False, 4, False),
         "draft1": (False, 1, True), "draft4": (False, 4, True)}


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("seed", range(NUM_CASES))
def test_random_engine(seed, mode):
    run_random_case(seed, *MODES[mode])


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("seed", range(max(1, NUM_CASES // 10)))
def test_random_engine_block_256(seed, mode):
    run_random_case_block_256(seed, *MODES[mode])
