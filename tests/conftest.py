import os
import sys
import types

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

HAS_CUDA = torch.cuda.is_available()


def _stub_module(name, **attrs):
    module = types.ModuleType(name)
    module.__dict__.update(attrs)
    sys.modules[name] = module


# The scheduler, block manager and input preparation are plain Python. Without a GPU, make the
# GPU-only imports and host-to-device copies no-ops so those parts can still be tested on CPU.
if not HAS_CUDA:
    import torch._dynamo
    torch._dynamo.config.disable = True

    try:
        import flash_attn, triton  # noqa: F401
    except ImportError:
        # layers/attention.py is the only module that needs them, for the GPU kernels.
        _stub_module("nanovllm.layers.attention", Attention=None)

    _tensor = torch.tensor

    def _tensor_no_pin(*args, pin_memory=False, **kwargs):
        return _tensor(*args, **kwargs)

    torch.tensor = _tensor_no_pin
    torch.Tensor.cuda = lambda self, *args, **kwargs: self


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
