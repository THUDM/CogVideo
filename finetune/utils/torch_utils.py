from typing import Dict, List, Optional, Union

import torch
from accelerate import Accelerator
from diffusers.utils.torch_utils import is_compiled_module


def unwrap_model(accelerator: Accelerator, model):
    model = accelerator.unwrap_model(model)
    model = model._orig_mod if is_compiled_module(model) else model
    return model


def align_device_and_dtype(
    x: Union[torch.Tensor, Dict[str, torch.Tensor]],
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
):
    if isinstance(x, torch.Tensor):
        if device is not None:
            x = x.to(device)
        if dtype is not None:
            x = x.to(dtype)
    elif isinstance(x, dict):
        if device is not None:
            x = {k: align_device_and_dtype(v, device, dtype) for k, v in x.items()}
        if dtype is not None:
            x = {k: align_device_and_dtype(v, device, dtype) for k, v in x.items()}
    return x


def expand_tensor_to_dims(tensor, ndim):
    while len(tensor.shape) < ndim:
        tensor = tensor.unsqueeze(-1)
    return tensor


def cast_training_params(model: Union[torch.nn.Module, List[torch.nn.Module]], dtype=torch.float32):
    """
    Casts the training parameters of the model to the specified data type.

    Args:
        model: The PyTorch model whose parameters will be cast.
        dtype: The data type to which the model parameters will be cast.
    """
    if not isinstance(model, list):
        model = [model]
    for m in model:
        for param in m.parameters():
            # only upcast trainable parameters into fp32
            if param.requires_grad:
                param.data = param.to(dtype)
