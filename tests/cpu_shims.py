"""Without a GPU, stub the GPU-only imports and the host-to-device copies so that the scheduler,
block manager and input preparation can still run. Imported by conftest.py, and first by any test
module whose functions run in a spawned process."""
import sys
import types

import torch

HAS_CUDA = torch.cuda.is_available()


def _stub_module(name, **attrs):
    module = types.ModuleType(name)
    module.__dict__.update(attrs)
    sys.modules[name] = module


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
