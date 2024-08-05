# Copyright 2023 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Callable, Dict, List, Optional, Set, Tuple, Type, Union

import torch
import torch.nn.functional as F
from torch import nn


class LoRALinearLayer(nn.Module):
    def __init__(self, in_features, out_features, rank=4, network_alpha=None, device=None, dtype=None):
        super().__init__()

        self.down = nn.Linear(in_features, rank, bias=False, device=device, dtype=dtype)
        self.up = nn.Linear(rank, out_features, bias=False, device=device, dtype=dtype)
        # This value has the same meaning as the `--network_alpha` option in the kohya-ss trainer script.
        # See https://github.com/darkstorm2150/sd-scripts/blob/main/docs/train_network_README-en.md#execute-learning
        self.network_alpha = network_alpha
        self.rank = rank
        self.out_features = out_features
        self.in_features = in_features

        nn.init.normal_(self.down.weight, std=1 / rank)
        nn.init.zeros_(self.up.weight)

    def forward(self, hidden_states):
        orig_dtype = hidden_states.dtype
        dtype = self.down.weight.dtype

        down_hidden_states = self.down(hidden_states.to(dtype))
        up_hidden_states = self.up(down_hidden_states)

        if self.network_alpha is not None:
            up_hidden_states *= self.network_alpha / self.rank

        return up_hidden_states.to(orig_dtype)


class LoRAConv2dLayer(nn.Module):
    def __init__(
        self, in_features, out_features, rank=4, kernel_size=(1, 1), stride=(1, 1), padding=0, network_alpha=None
    ):
        super().__init__()

        self.down = nn.Conv2d(in_features, rank, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        # according to the official kohya_ss trainer kernel_size are always fixed for the up layer
        # # see: https://github.com/bmaltais/kohya_ss/blob/2accb1305979ba62f5077a23aabac23b4c37e935/networks/lora_diffusers.py#L129
        self.up = nn.Conv2d(rank, out_features, kernel_size=(1, 1), stride=(1, 1), bias=False)

        # This value has the same meaning as the `--network_alpha` option in the kohya-ss trainer script.
        # See https://github.com/darkstorm2150/sd-scripts/blob/main/docs/train_network_README-en.md#execute-learning
        self.network_alpha = network_alpha
        self.rank = rank

        nn.init.normal_(self.down.weight, std=1 / rank)
        nn.init.zeros_(self.up.weight)

    def forward(self, hidden_states):
        orig_dtype = hidden_states.dtype
        dtype = self.down.weight.dtype

        down_hidden_states = self.down(hidden_states.to(dtype))
        up_hidden_states = self.up(down_hidden_states)

        if self.network_alpha is not None:
            up_hidden_states *= self.network_alpha / self.rank

        return up_hidden_states.to(orig_dtype)


class LoRACompatibleConv(nn.Conv2d):
    """
    A convolutional layer that can be used with LoRA.
    """

    def __init__(self, *args, lora_layer: Optional[LoRAConv2dLayer] = None, scale: float = 1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.lora_layer = lora_layer
        self.scale = scale

    def set_lora_layer(self, lora_layer: Optional[LoRAConv2dLayer]):
        self.lora_layer = lora_layer

    def _fuse_lora(self, lora_scale=1.0):
        if self.lora_layer is None:
            return

        dtype, device = self.weight.data.dtype, self.weight.data.device

        w_orig = self.weight.data.float()
        w_up = self.lora_layer.up.weight.data.float()
        w_down = self.lora_layer.down.weight.data.float()

        if self.lora_layer.network_alpha is not None:
            w_up = w_up * self.lora_layer.network_alpha / self.lora_layer.rank

        fusion = torch.mm(w_up.flatten(start_dim=1), w_down.flatten(start_dim=1))
        fusion = fusion.reshape((w_orig.shape))
        fused_weight = w_orig + (lora_scale * fusion)
        self.weight.data = fused_weight.to(device=device, dtype=dtype)

        # we can drop the lora layer now
        self.lora_layer = None

        # offload the up and down matrices to CPU to not blow the memory
        self.w_up = w_up.cpu()
        self.w_down = w_down.cpu()
        self._lora_scale = lora_scale

    def _unfuse_lora(self):
        if not (hasattr(self, "w_up") and hasattr(self, "w_down")):
            return

        fused_weight = self.weight.data
        dtype, device = fused_weight.data.dtype, fused_weight.data.device

        self.w_up = self.w_up.to(device=device).float()
        self.w_down = self.w_down.to(device).float()

        fusion = torch.mm(self.w_up.flatten(start_dim=1), self.w_down.flatten(start_dim=1))
        fusion = fusion.reshape((fused_weight.shape))
        unfused_weight = fused_weight.float() - (self._lora_scale * fusion)
        self.weight.data = unfused_weight.to(device=device, dtype=dtype)

        self.w_up = None
        self.w_down = None

    def forward(self, hidden_states, scale: float = None):
        if scale is None:
            scale = self.scale
        if self.lora_layer is None:
            # make sure to the functional Conv2D function as otherwise torch.compile's graph will break
            # see: https://github.com/huggingface/diffusers/pull/4315
            return F.conv2d(
                hidden_states, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups
            )
        else:
            return super().forward(hidden_states) + (scale * self.lora_layer(hidden_states))


class LoRACompatibleLinear(nn.Linear):
    """
    A Linear layer that can be used with LoRA.
    """

    def __init__(self, *args, lora_layer: Optional[LoRALinearLayer] = None, scale: float = 1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.lora_layer = lora_layer
        self.scale = scale

    def set_lora_layer(self, lora_layer: Optional[LoRALinearLayer]):
        self.lora_layer = lora_layer

    def _fuse_lora(self, lora_scale=1.0):
        if self.lora_layer is None:
            return

        dtype, device = self.weight.data.dtype, self.weight.data.device

        w_orig = self.weight.data.float()
        w_up = self.lora_layer.up.weight.data.float()
        w_down = self.lora_layer.down.weight.data.float()

        if self.lora_layer.network_alpha is not None:
            w_up = w_up * self.lora_layer.network_alpha / self.lora_layer.rank

        fused_weight = w_orig + (lora_scale * torch.bmm(w_up[None, :], w_down[None, :])[0])
        self.weight.data = fused_weight.to(device=device, dtype=dtype)

        # we can drop the lora layer now
        self.lora_layer = None

        # offload the up and down matrices to CPU to not blow the memory
        self.w_up = w_up.cpu()
        self.w_down = w_down.cpu()
        self._lora_scale = lora_scale

    def _unfuse_lora(self):
        if not (hasattr(self, "w_up") and hasattr(self, "w_down")):
            return

        fused_weight = self.weight.data
        dtype, device = fused_weight.dtype, fused_weight.device

        w_up = self.w_up.to(device=device).float()
        w_down = self.w_down.to(device).float()

        unfused_weight = fused_weight.float() - (self._lora_scale * torch.bmm(w_up[None, :], w_down[None, :])[0])
        self.weight.data = unfused_weight.to(device=device, dtype=dtype)

        self.w_up = None
        self.w_down = None

    def forward(self, hidden_states, scale: float = None):
        if scale is None:
            scale = self.scale
        if self.lora_layer is None:
            out = super().forward(hidden_states)
            return out
        else:
            out = super().forward(hidden_states) + (scale * self.lora_layer(hidden_states))
            return out


def _find_children(
    model,
    search_class: List[Type[nn.Module]] = [nn.Linear],
):
    """
    Find all modules of a certain class (or union of classes).

    Returns all matching modules, along with the parent of those moduless and the
    names they are referenced by.
    """
    # For each target find every linear_class module that isn't a child of a LoraInjectedLinear
    for parent in model.modules():
        for name, module in parent.named_children():
            if any([isinstance(module, _class) for _class in search_class]):
                yield parent, name, module


def _find_modules_v2(
    model,
    ancestor_class: Optional[Set[str]] = None,
    search_class: List[Type[nn.Module]] = [nn.Linear],
    exclude_children_of: Optional[List[Type[nn.Module]]] = [
        LoRACompatibleLinear,
        LoRACompatibleConv,
        LoRALinearLayer,
        LoRAConv2dLayer,
    ],
):
    """
    Find all modules of a certain class (or union of classes) that are direct or
    indirect descendants of other modules of a certain class (or union of classes).

    Returns all matching modules, along with the parent of those moduless and the
    names they are referenced by.
    """

    # Get the targets we should replace all linears under
    if ancestor_class is not None:
        ancestors = (module for module in model.modules() if module.__class__.__name__ in ancestor_class)
    else:
        # this, incase you want to naively iterate over all modules.
        ancestors = [module for module in model.modules()]

    # For each target find every linear_class module that isn't a child of a LoraInjectedLinear
    for ancestor in ancestors:
        for fullname, module in ancestor.named_modules():
            if any([isinstance(module, _class) for _class in search_class]):
                # Find the direct parent if this is a descendant, not a child, of target
                *path, name = fullname.split(".")
                parent = ancestor
                flag = False
                while path:
                    try:
                        parent = parent.get_submodule(path.pop(0))
                    except:
                        flag = True
                        break
                if flag:
                    continue
                # Skip this linear if it's a child of a LoraInjectedLinear
                if exclude_children_of and any([isinstance(parent, _class) for _class in exclude_children_of]):
                    continue
                # Otherwise, yield it
                yield parent, name, module


_find_modules = _find_modules_v2


def inject_trainable_lora_extended(
    model: nn.Module,
    target_replace_module: Set[str] = None,
    rank: int = 4,
    scale: float = 1.0,
):
    for _module, name, _child_module in _find_modules(
        model, target_replace_module, search_class=[nn.Linear, nn.Conv2d]
    ):
        if _child_module.__class__ == nn.Linear:
            weight = _child_module.weight
            bias = _child_module.bias
            lora_layer = LoRALinearLayer(
                in_features=_child_module.in_features,
                out_features=_child_module.out_features,
                rank=rank,
            )
            _tmp = (
                LoRACompatibleLinear(
                    _child_module.in_features,
                    _child_module.out_features,
                    lora_layer=lora_layer,
                    scale=scale,
                )
                .to(weight.dtype)
                .to(weight.device)
            )
            _tmp.weight = weight
            if bias is not None:
                _tmp.bias = bias
        elif _child_module.__class__ == nn.Conv2d:
            weight = _child_module.weight
            bias = _child_module.bias
            lora_layer = LoRAConv2dLayer(
                in_features=_child_module.in_channels,
                out_features=_child_module.out_channels,
                rank=rank,
                kernel_size=_child_module.kernel_size,
                stride=_child_module.stride,
                padding=_child_module.padding,
            )
            _tmp = (
                LoRACompatibleConv(
                    _child_module.in_channels,
                    _child_module.out_channels,
                    kernel_size=_child_module.kernel_size,
                    stride=_child_module.stride,
                    padding=_child_module.padding,
                    lora_layer=lora_layer,
                    scale=scale,
                )
                .to(weight.dtype)
                .to(weight.device)
            )
            _tmp.weight = weight
            if bias is not None:
                _tmp.bias = bias
        else:
            continue

        _module._modules[name] = _tmp
        # print('injecting lora layer to', _module, name)

    return


def update_lora_scale(
    model: nn.Module,
    target_module: Set[str] = None,
    scale: float = 1.0,
):
    for _module, name, _child_module in _find_modules(
        model, target_module, search_class=[LoRACompatibleLinear, LoRACompatibleConv]
    ):
        _child_module.scale = scale

    return
