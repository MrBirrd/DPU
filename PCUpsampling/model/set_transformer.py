"""
Implements the Set Transformer from https://arxiv.org/abs/1810.00825 with
some modifications to "inject" the diffusion noise level into the network.
"""
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor, nn


class AdaNorm(nn.Module):
    def forward(self, x: Tensor, ctx: Tensor) -> Tensor:
        raise NotImplementedError()


class AdaGN(nn.Module):
    def __init__(
        self,
        num_channels: int,
        ctx_dim: int,
        num_groups: int = 32,
    ):
        super().__init__()
        self.gn = nn.GroupNorm(
            num_groups=num_groups,
            num_channels=num_channels,
            affine=False,
        )
        self.bias = nn.Linear(ctx_dim, num_channels)
        self.scale = nn.Linear(ctx_dim, num_channels)

        with torch.no_grad():
            self.bias.weight.fill_(0.0)
            self.bias.bias.fill_(0.0)
            self.scale.weight.fill_(0.0)
            self.scale.bias.fill_(1.0)

    def forward(self, x: Tensor, ctx: Tensor) -> Tensor:
        x_bcn = rearrange(x, "b n c -> b c n")
        normed_bcn = self.gn(x_bcn)
        normed = rearrange(normed_bcn, "b c n -> b n c")
        bias = self.bias(ctx)
        scale = self.scale(ctx)

        shape = (x.shape[0], *((1,) * (x.ndim - 2)), x.shape[-1])
        return scale.reshape(shape) * normed + bias.reshape(shape)


class MLP(nn.Sequential):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        width_size: int,
        depth: int = 1,
        activation: Callable = nn.ReLU,
    ):
        layers = []
        layers.append(
            nn.Linear(
                in_features,
                width_size,
            )
        )
        layers.append(activation())

        for _ in range(depth - 1):
            layers.append(
                nn.Linear(
                    width_size,
                    width_size,
                )
            )
            layers.append(activation())

        layers.append(
            nn.Linear(
                width_size,
                out_features,
            )
        )

        super().__init__(*layers)


class AttentionPool(nn.Module):
    """
    Uses attention to pool a set of features into a smaller set of features.
    The queries are defined by the `inducers` learnable parameter.
    """

    def __init__(
        self,
        feature_dim: int,
        num_heads: int,
        num_inducers: int,
    ):
        super().__init__()

        assert feature_dim % num_heads == 0, (feature_dim, num_heads)
        dims_per_head = feature_dim // num_heads

        self.inducers = nn.Parameter(
            torch.randn(
                1,
                num_heads,
                num_inducers,
                dims_per_head,
            )
        )

        self.kv_proj = nn.Linear(feature_dim, feature_dim * 2, bias=False)
        self.out_proj = nn.Linear(feature_dim, feature_dim, bias=False)

        self.num_heads = num_heads
        self.feature_dim = feature_dim
        self.dims_per_head = dims_per_head

    def forward(self, kv: Tensor) -> Tensor:
        k_heads, v_heads = rearrange(
            self.kv_proj(kv),
            "b n (t h d) -> t b h n d",
            t=2,
            h=self.num_heads,
            d=self.dims_per_head,
        )

        queries = self.inducers.repeat(kv.shape[0], 1, 1, 1)
        attn = F.scaled_dot_product_attention(
            queries,
            k_heads,
            v_heads,
        )  # (batch_size, num_heads, num_inducers, feature_dim // num_heads)

        attn = rearrange(attn, "b h i d -> b i (h d)")

        return self.out_proj(attn)


class Broadcast(nn.Module):
    """
    A module that implements a pool -> mlp -> unpool sequence to return an updated
    version of the input tokens.
    """

    def __init__(
        self,
        feature_dim: int,
        num_inducers: int,
        t_embed_dim: int,
        num_heads: int = 8,
        mlp_blowup: int = 2,
        activation: nn.Module = nn.ReLU,
    ):
        super().__init__()
        self.pool = AttentionPool(feature_dim, num_heads, num_inducers)
        self.norm_1 = AdaGN(feature_dim, t_embed_dim)
        self.mlp = MLP(feature_dim, feature_dim, mlp_blowup * feature_dim, activation=activation)
        self.norm_2 = AdaGN(feature_dim, t_embed_dim)
        self.unpool = nn.MultiheadAttention(feature_dim, num_heads, batch_first=True)

    def forward(
        self,
        x: Tensor,
        t_embed: Tensor,
        return_h: bool = False,
        h: Tensor | None = None,
    ) -> tuple[Tensor, Tensor | None]:
        """
        Args:
            `x` - the input features
            `t_embed` - the diffusion noise level
            `return_h` - whether to return the inducer states for efficient point cloud upsampling
            `h` - the inducer states if using cached values
        """
        if h is None:
            h = self.pool(x)
            h = self.norm_1(h, t_embed)
            h = self.mlp(h)
            h = self.norm_2(h, t_embed)

        attn, _weights = self.unpool(x, h, h, need_weights=False)

        if return_h:
            return attn, h
        else:
            return attn, None


class BroadcastingLayer(nn.Module):
    """
    A module equivalent to a standard transformer layer which uses a
    broadcast -> mlp sequence, with skip connections in a pre-norm fashion.
    """

    def __init__(
        self,
        feature_dim: int,
        num_inducers: int,
        embed_dim: int,
        num_heads: int = 8,
        mlp_blowup: int = 2,
        activation: nn.Module = nn.ReLU,
        num_groups: int = 32,
    ):
        super().__init__()
        self.broadcast_norm = AdaGN(feature_dim, embed_dim, num_groups)
        self.broadcast = Broadcast(
            feature_dim,
            num_inducers,
            embed_dim,
            num_heads,
            mlp_blowup=mlp_blowup,
            activation=activation,
        )
        self.mlp_norm = AdaGN(feature_dim, embed_dim)
        self.mlp = MLP(feature_dim, feature_dim, mlp_blowup * feature_dim, activation=activation)

        with torch.no_grad():
            # scale down the skip connection weights
            self.broadcast.unpool.out_proj.weight *= 0.1
            self.mlp[-1].weight *= 0.1

    def forward(
        self,
        x: Tensor,
        t_embed: Tensor,
        return_h: bool = False,
        h: Tensor | None = None,
    ) -> tuple[Tensor, Tensor | None]:
        y = self.broadcast_norm(x, t_embed)
        x_b, h = self.broadcast(y, t_embed, return_h, h)
        x = x + x_b
        y = self.mlp_norm(x, t_embed)
        x = x + self.mlp(y)

        return x, h


class SetTransformer(nn.Module):
    """
    A set transformer is just a sequence of broadcasting layers.
    """

    def __init__(
        self,
        n_layers: int,
        feature_dim: int,
        num_inducers: int,
        t_embed_dim: int,
        num_groups: int = 32,
        **kwargs,
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                BroadcastingLayer(
                    feature_dim=feature_dim,
                    num_inducers=num_inducers,
                    embed_dim=t_embed_dim,
                    num_groups=num_groups,
                    **kwargs,
                )
                for _ in range(n_layers)
            ]
        )
        self.feature_dim = feature_dim

    def forward(
        self,
        features: Tensor,
        t_embed: Tensor,
        return_h: bool = False,
        hs: list[Tensor] | None = None,
    ) -> tuple[Tensor, list[Tensor] | None]:
        if hs is None:
            hs = [None] * len(self.layers)

        stored_h = []
        for layer, h in zip(self.layers, hs):
            features, h = layer(features, t_embed, return_h=return_h, h=h)
            stored_h.append(h)

        if return_h:
            return features, stored_h
        else:
            return features, None
