import os

import pytest

from engine_sim import run_random_case, run_random_case_block_256

NUM_CASES = int(os.environ.get("NANOVLLM_SIM_CASES", "300"))
# (async_scheduling, num_speculative_tokens)
MODES = {"sync": (False, 0), "async": (True, 0), "spec1": (False, 1), "spec4": (False, 4)}


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("seed", range(NUM_CASES))
def test_random_engine(seed, mode):
    run_random_case(seed, *MODES[mode])


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("seed", range(max(1, NUM_CASES // 10)))
def test_random_engine_block_256(seed, mode):
    run_random_case_block_256(seed, *MODES[mode])
