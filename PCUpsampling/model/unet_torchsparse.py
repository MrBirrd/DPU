import argparse
import random
from functools import partial
from typing import Any, Dict

import numpy as np
import torch
import torch.utils.data
from einops import rearrange
from torch import nn, einsum
from torch.cuda import amp
from torch.nn import functional as F
from torchsparse import SparseTensor
from torchsparse import nn as spnn
from torchsparse.nn.utils import fapply
from torchsparse.utils.collate import sparse_collate
from torchsparse.utils.quantize import sparse_quantize
import math


def silu(input: SparseTensor, inplace: bool = True) -> SparseTensor:
    return fapply(input, F.silu, inplace=inplace)


class BatchNorm(nn.BatchNorm1d):
    def forward(self, input: SparseTensor) -> SparseTensor:
        return fapply(input, super().forward)


class GroupNorm(nn.GroupNorm):
    def forward(self, input: SparseTensor) -> SparseTensor:
        coords, feats, stride = input.coords, input.feats, input.stride

        batch_size = torch.max(coords[:, 0]).item() + 1
        num_channels = feats.shape[1]

        # PyTorch's GroupNorm function expects the input to be in (N, C, *)
        # format where N is batch size, and C is number of channels. "feats"
        # is not in that format. So, we extract the feats corresponding to
        # each sample, bring it to the format expected by PyTorch's GroupNorm
        # function, and invoke it.

        nfeats = torch.zeros_like(feats)
        for k in range(batch_size):
            indices = coords[:, 0] == k
            bfeats = feats[indices]
            bfeats = bfeats.transpose(0, 1).reshape(1, num_channels, -1)
            bfeats = super().forward(bfeats)
            bfeats = bfeats.reshape(num_channels, -1).transpose(0, 1)
            nfeats[indices] = bfeats

        input.F = nfeats
        return input


def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d


class ModuleAdapter(nn.Module):
    def __init__(self, module: nn.Module) -> None:
        super().__init__()
        self.module = module

    def forward(self, input: SparseTensor) -> SparseTensor:
        feats = input.F
        feats = self.module(feats)
        input.F = feats
        return input


class Identity(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, input: SparseTensor) -> SparseTensor:
        return input


class FunctionToModule(nn.Module):
    def __init__(self, fn: Any) -> None:
        super().__init__()
        self.fn = fn

    def forward(self, input: SparseTensor) -> SparseTensor:
        feats = input.F
        feats = self.fn(feats)
        input.F = feats
        return input


class ModuleWrapper(nn.Module):
    def __init__(self, func):
        super().__init__()
        self.func = func

    def forward(self, input: SparseTensor, *args, **kwargs) -> SparseTensor:
        return self.func(input, *args, **kwargs)


def Upsample(dim, dim_out):
    return spnn.Conv3d(dim, dim_out, kernel_size=2, stride=2, dilation=1, bias=False, transposed=True)


def Downsample(dim, dim_out):
    return spnn.Conv3d(dim, dim_out, kernel_size=2, stride=2, dilation=1, bias=False)


def DownsampleOld(stride=2, kernel_size=2):
    return ModuleWrapper(partial(spnn.functional.spdownsample, stride=stride, kernel_size=kernel_size))


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x


class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))

    def forward(self, x):
        return F.normalize(x, dim=1) * self.g * (x.shape[1] ** 0.5)


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = RMSNorm(dim)

    def forward(self, x):
        x = self.norm(x)
        return self.fn(x)


class Block(nn.Module):
    def __init__(self, dim, dim_out, groups=8):
        super().__init__()
        self.proj = spnn.Conv3d(dim, dim_out, 3)
        # self.norm = GroupNorm(groups, dim_out)
        self.norm = BatchNorm(dim_out)
        # self.act = silu

    def forward(self, x, scale_shift=None):
        x = self.proj(x)
        x = self.norm(x)

        if exists(scale_shift):
            x_coords = x.C
            x_feats = x.F
            batch_size = torch.max(x_coords[:, 0]).item() + 1
            scale, shift = scale_shift

            scaled_feats = torch.zeros_like(x_feats, device=x_feats.device)
            for k in range(batch_size):
                indices = x_coords[:, 0] == k
                bfeats = x_feats[indices]
                bfeats = bfeats * (scale[k] + 1) + shift[k]
                scaled_feats[indices] = bfeats

            x.F = scaled_feats

        silu(x)

        return x


class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, *, time_emb_dim=None, groups=8):
        super().__init__()
        self.mlp = nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, dim_out * 2)) if exists(time_emb_dim) else None

        self.block1 = Block(dim, dim_out, groups=groups)
        self.block2 = Block(dim_out, dim_out, groups=groups)
        self.res_conv = spnn.Conv3d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb=None):
        scale_shift = None
        if exists(self.mlp) and exists(time_emb):
            time_emb = self.mlp(time_emb)
            # time_emb = rearrange(time_emb, 'b c -> b c 1')
            scale_shift = time_emb.chunk(2, dim=1)

        h = self.block1(x, scale_shift=scale_shift)

        h = self.block2(h)

        return h + self.res_conv(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        hidden_dim = dim_head * heads

        self.to_qkv = nn.Conv1d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv1d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, n = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(lambda t: rearrange(t, "b (h c) n -> b h c n", h=self.heads), qkv)

        q = q * self.scale

        sim = einsum("b h d i, b h d j -> b h i j", q, k)
        attn = sim.softmax(dim=-1)
        out = einsum("b h i j, b h d j -> b h i d", attn, v)

        out = rearrange(out, "b h n d -> b (h d) n")
        return self.to_out(out)


class MiddleAttention(nn.Module):
    def __init__(self, mid_dim, attn_dim_head, attn_heads) -> None:
        super().__init__()
        self.attend = Residual(PreNorm(mid_dim, Attention(mid_dim, dim_head=attn_dim_head, heads=attn_heads)))
        print("Mid dim:", mid_dim)

    def forward(self, x):
        x_coords = x.C
        x_feats = x.F
        batch_size = torch.max(x_coords[:, 0]).item() + 1
        n_channels = x_feats.shape[1]

        feats_tensor = torch.tensor([], device=x_feats.device)

        for k in range(batch_size):
            indices = x_coords[:, 0] == k
            bfeats = x_feats[indices]
            print(bfeats.shape)
            feats_tensor = torch.cat((feats_tensor, bfeats.unsqueeze(0)), dim=0)
        print(feats_tensor.shape)
        exit(0)

        return self.attend(x)


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim, theta=10000):
        super().__init__()
        self.dim = dim
        self.theta = theta

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(self.theta) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class RandomOrLearnedSinusoidalPosEmb(nn.Module):
    """following @crowsonkb 's lead with random (learned optional) sinusoidal pos emb"""

    """ https://github.com/crowsonkb/v-diffusion-jax/blob/master/diffusion/models/danbooru_128.py#L8 """

    def __init__(self, dim, is_random=False):
        super().__init__()
        assert (dim % 2) == 0
        half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(half_dim), requires_grad=not is_random)

    def forward(self, x):
        x = rearrange(x, "b -> b 1")
        freqs = x * rearrange(self.weights, "d -> 1 d") * 2 * math.pi
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim=-1)
        fouriered = torch.cat((x, fouriered), dim=-1)
        return fouriered


def batched_tensors_to_sparse(tensors, voxel_size=1e-3):
    device = tensors.device
    sparse_tensors = []
    for item in tensors:
        item = rearrange(item, "c n -> n c")
        coords, indices = sparse_quantize(item[:, :3].cpu().numpy(), voxel_size, return_index=True)
        coords = torch.tensor(coords, dtype=torch.int, device=device)
        feats = torch.tensor(item[indices], dtype=torch.float, device=device)
        tensor = SparseTensor(coords=coords, feats=feats)
        sparse_tensors.append(tensor)
    return sparse_collate(sparse_tensors)


class TSUnet(nn.Module):
    def __init__(
        self,
        dim,
        init_dim=None,
        out_dim=None,
        dim_mults=(1, 2, 4, 8),
        channels=3,
        self_condition=False,
        resnet_block_groups=8,
        learned_variance=False,
        learned_sinusoidal_cond=False,
        random_fourier_features=False,
        learned_sinusoidal_dim=16,
        sinusoidal_pos_emb_theta=10000,
        attn_dim_head=32,
        attn_heads=4,
    ):
        super().__init__()

        # determine dimensions

        self.channels = channels
        self.self_condition = self_condition
        input_channels = channels * (2 if self_condition else 1)

        init_dim = default(init_dim, dim)
        self.init_conv = spnn.Conv3d(input_channels, init_dim, kernel_size=7, stride=1, dilation=1, bias=False)

        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        block_klass = partial(ResnetBlock, groups=resnet_block_groups)

        # to sparse
        self.to_sparse = batched_tensors_to_sparse
        # time embeddings

        time_dim = dim * 4

        self.random_or_learned_sinusoidal_cond = learned_sinusoidal_cond or random_fourier_features

        if self.random_or_learned_sinusoidal_cond:
            sinu_pos_emb = RandomOrLearnedSinusoidalPosEmb(learned_sinusoidal_dim, random_fourier_features)
            fourier_dim = learned_sinusoidal_dim + 1
        else:
            sinu_pos_emb = SinusoidalPosEmb(dim, theta=sinusoidal_pos_emb_theta)
            fourier_dim = dim

        self.time_mlp = nn.Sequential(
            sinu_pos_emb, nn.Linear(fourier_dim, time_dim), nn.GELU(), nn.Linear(time_dim, time_dim)
        )

        # layers

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(
                nn.ModuleList(
                    [
                        block_klass(dim_in, dim_in, time_emb_dim=time_dim),
                        block_klass(dim_in, dim_in, time_emb_dim=time_dim),
                        # Residual(PreNorm(dim_in, LinearAttention(dim_in))),
                        Identity(),
                        Downsample(dim_in, dim_out) if not is_last else spnn.Conv3d(dim_in, dim_out, kernel_size=3),
                    ]
                )
            )

        mid_dim = dims[-1]
        self.mid_block1 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)
        self.mid_attn = MiddleAttention(mid_dim, attn_dim_head, attn_heads)
        self.mid_block2 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind == (len(in_out) - 1)

            self.ups.append(
                nn.ModuleList(
                    [
                        block_klass(dim_out + dim_in, dim_out, time_emb_dim=time_dim),
                        block_klass(dim_out + dim_in, dim_out, time_emb_dim=time_dim),
                        # Residual(PreNorm(dim_out, LinearAttention(dim_out))),
                        Identity(),
                        Upsample(dim_out, dim_in) if not is_last else spnn.Conv3d(dim_out, dim_in, kernel_size=3),
                    ]
                )
            )

        default_out_dim = channels * (1 if not learned_variance else 2)
        self.out_dim = default(out_dim, default_out_dim)

        self.final_res_block = block_klass(dim * 2, dim, time_emb_dim=time_dim)
        self.final_conv = spnn.Conv3d(dim, self.out_dim, kernel_size=1)

    def forward(self, x_coords, time, cond=None, x_self_cond=None):
        # handle conditioning
        if cond is not None:
            # stacked_feats = self.voxel_feat_cat(x1_features=x_coords, x2_features=cond, x1_coords=x_coords, x2_coords=cond)
            stacked_coords = torch.cat((x_coords, cond), dim=1)

        # generate sparse tensor
        x_sparse = self.to_sparse(x_coords if cond is None else stacked_coords, voxel_size=0.001)

        if self.self_condition:
            raise NotImplementedError
            x_self_cond = default(x_self_cond, lambda: torch.zeros_like(x))
            x = torch.cat((x_self_cond, x), dim=1)

        x = self.init_conv(x_sparse)
        r = x

        t = self.time_mlp(time)

        h = []

        for block1, block2, attn, downsample in self.downs:
            x = block1(x, t)
            h.append(x)

            x = block2(x, t)
            x = attn(x)
            h.append(x)

            x = downsample(x)

        x = self.mid_block1(x, t)
        # x = self.mid_attn(x)
        x = self.mid_block2(x, t)

        for block1, block2, attn, upsample in self.ups:
            x = spnn.cat((x, h.pop()), dim=1)
            x = block1(x, t)

            x = spnn.cat((x, h.pop()), dim=1)
            x = block2(x, t)
            x = attn(x)

            x = upsample(x)

        x = spnn.cat((x, r), dim=1)

        x = self.final_res_block(x, t)
        return self.final_conv(x)
