import os

import pytest

from engine_sim import run_random_case, run_random_case_block_256

NUM_CASES = int(os.environ.get("NANOVLLM_SIM_CASES", "300"))


@pytest.mark.parametrize("seed", range(NUM_CASES))
def test_random_engine(seed):
    run_random_case(seed)


@pytest.mark.parametrize("seed", range(max(1, NUM_CASES // 10)))
def test_random_engine_block_256(seed):
    run_random_case_block_256(seed)
