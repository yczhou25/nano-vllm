import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402

if not torch.cuda.is_available():
    # Run the Triton kernels (fused_sample) in Triton's CPU interpreter. Read when triton.language
    # is imported, so before cpu_shims and nanovllm import it.
    os.environ.setdefault("TRITON_INTERPRET", "1")

from cpu_shims import HAS_CUDA  # noqa: E402  (must run before nanovllm is imported)


def pytest_configure(config):
    config.addinivalue_line("markers", "gpu: needs a CUDA GPU and a model checkpoint")


@pytest.fixture(autouse=True)
def restore_block_size():
    # Sequence.block_size is a class attribute that LLMEngine and the tests overwrite.
    from nanovllm.engine.sequence import Sequence
    block_size = Sequence.block_size
    yield
    Sequence.block_size = block_size


def pytest_collection_modifyitems(config, items):
    if HAS_CUDA:
        return
    skip = pytest.mark.skip(reason="needs a CUDA GPU")
    for item in items:
        if "gpu" in item.keywords:
            item.add_marker(skip)
