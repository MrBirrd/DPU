# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.
"""
adaptive group norm 
"""
import torch.nn as nn
from torch.nn.init import _calculate_fan_in_and_fan_out
import math
import torch

def _calculate_correct_fan(tensor, mode):
    """
    copied and modified from https://github.com/pytorch/pytorch/blob/master/torch/nn/init.py#L337
    """
    mode = mode.lower()
    valid_modes = ["fan_in", "fan_out", "fan_avg"]
    if mode not in valid_modes:
        raise ValueError("Mode {} not supported, please use one of {}".format(mode, valid_modes))

    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    return fan_in if mode == "fan_in" else fan_out

def kaiming_uniform_(tensor, gain=1.0, mode="fan_in"):
    r"""Fills the input `Tensor` with values according to the method
    described in `Delving deep into rectifiers: Surpassing human-level
    performance on ImageNet classification` - He, K. et al. (2015), using a
    uniform distribution. The resulting tensor will have values sampled from
    :math:`\mathcal{U}(-\text{bound}, \text{bound})` where
    .. math::
        \text{bound} = \text{gain} \times \sqrt{\frac{3}{\text{fan\_mode}}}
    Also known as He initialization.
    Args:
        tensor: an n-dimensional `torch.Tensor`
        gain: multiplier to the dispersion
        mode: either ``'fan_in'`` (default) or ``'fan_out'``. Choosing ``'fan_in'``
            preserves the magnitude of the variance of the weights in the
            forward pass. Choosing ``'fan_out'`` preserves the magnitudes in the
            backwards pass.
    Examples:
        >>> w = torch.empty(3, 5)
        >>> nn.init.kaiming_uniform_(w, mode='fan_in')
    """
    fan = _calculate_correct_fan(tensor, mode)
    # gain = calculate_gain(nonlinearity, a)
    var = gain / max(1.0, fan)
    bound = math.sqrt(3.0 * var)  # Calculate uniform bounds from standard deviation
    with torch.no_grad():
        return tensor.uniform_(-bound, bound)

def variance_scaling_init_(tensor, scale):
    return kaiming_uniform_(tensor, gain=1e-10 if scale == 0 else scale, mode="fan_avg")

def dense(in_channels, out_channels, init_scale=1.0):
    lin = nn.Linear(in_channels, out_channels)
    variance_scaling_init_(lin.weight, scale=init_scale)
    nn.init.zeros_(lin.bias)
    return lin

class AdaGN(nn.Module):
    """
    adaptive group normalization
    """

    def __init__(self, ndim, cfg, n_channel):
        """
        ndim: dim of the input features
        n_channel: number of channels of the inputs
        ndim_style: channel of the style features
        """
        super().__init__()
        style_dim = cfg.latent_pts.style_dim
        init_scale = cfg.latent_pts.ada_mlp_init_scale
        self.ndim = ndim
        self.n_channel = n_channel
        self.style_dim = style_dim
        self.out_dim = n_channel * 2
        self.norm = nn.GroupNorm(8, n_channel)
        in_channel = n_channel
        self.emd = dense(style_dim, n_channel * 2, init_scale=init_scale)
        self.emd.bias.data[:in_channel] = 1
        self.emd.bias.data[in_channel:] = 0

    def __repr__(self):
        return f"AdaGN(GN(8, {self.n_channel}), Linear({self.style_dim}, {self.out_dim}))"

    def forward(self, image, style):
        # style: B,D
        # image: B,D,N,1
        CHECK2D(style)
        style = self.emd(style)
        if self.ndim == 3:  # B,D,V,V,V
            CHECK5D(image)
            style = style.view(style.shape[0], -1, 1, 1, 1)  # 5D
        elif self.ndim == 2:  # B,D,N,1
            CHECK4D(image)
            style = style.view(style.shape[0], -1, 1, 1)  # 4D
        elif self.ndim == 1:  # B,D,N
            CHECK3D(image)
            style = style.view(style.shape[0], -1, 1)  # 4D
        else:
            raise NotImplementedError

        factor, bias = style.chunk(2, 1)
        result = self.norm(image)
        result = result * factor + bias
        return result
