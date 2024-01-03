# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.

from functools import partial
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
from einops import rearrange

import model.utils as utils
from model.set_transformer import SetTransformer

from model.modules import Attention
from .pvcnn import (LinearAttention, SharedMLP, create_mlp_components,
                    create_pointnet2_fp_modules,
                    create_pointnet2_sa_components, create_pvc_layer_params)


class PVCNN2Unet(nn.Module):
    """
    copied and modified from https://github.com/alexzhou907/PVD/blob/9747265a5f141e5546fd4f862bfa66aa59f1bd33/model/pvcnn_generation.py#L172
    """

    def __init__(
        self,
        cfg: Dict,
    ):
        super().__init__()
        
        model_cfg = cfg.model
        pvd_cfg = model_cfg.PVD
        
        # initialize class variables
        self.input_dim = utils.default(model_cfg.in_dim, 3)
        self.extra_feature_channels = utils.default(model_cfg.extra_feature_channels, 3)
        self.embed_dim = utils.default(model_cfg.time_embed_dim, 64)
        
        out_dim = utils.default(model_cfg.out_dim, 3)
        st_params = utils.default(model_cfg.ST, {"layers": 6, "fdim": 512, "inducers": 32})
        dropout = utils.default(model_cfg.dropout, 0.1)
        attn_type = utils.default(pvd_cfg.attention_type, "linear")

        self.embedf = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(self.embed_dim, self.embed_dim),
        )

        sa_blocks, fp_blocks = create_pvc_layer_params(cfg)

        # prepare attention
        if attn_type.lower() == "settransformer":
            attention_fn = partial(SetTransformer,
                           n_layers=st_params["layers"],
                           num_inducers=st_params["inducers"],
                           t_embed_dim=1,
                           num_groups=st_params["gn_groups"]
                           )
        elif attn_type.lower() == "linear":
            attention_fn = partial(LinearAttention, heads=cfg.model.PVD.attention_heads)
        elif attn_type.lower() == "flash":
            attention_fn = partial(Attention, norm=False, flash=True, heads=cfg.model.PVD.attention_heads)
        else:
            attention_fn = None
            
        # create set abstraction layers
        sa_layers, sa_in_channels, channels_sa_features, *_ = create_pointnet2_sa_components(
            input_dim=self.input_dim,
            sa_blocks=sa_blocks,
            extra_feature_channels=self.extra_feature_channels,
            with_se=True,
            embed_dim=self.embed_dim,  # time embedding dim
            attention_fn=attention_fn,
            attention_layers=cfg.model.PVD.attentions,
            dropout=dropout,
            cfg=cfg,
        )
        self.sa_layers = nn.ModuleList(sa_layers)

        if attention_fn is not None:
            self.global_att = attention_fn(dim=channels_sa_features)

        # create feature propagation layers
        # only use extra features in the last fp module WHY ACTUALLY??
        sa_in_channels[0] = self.extra_feature_channels + self.input_dim

        fp_layers, channels_fp_features = create_pointnet2_fp_modules(
            fp_blocks=fp_blocks,
            in_channels=channels_sa_features,
            sa_in_channels=sa_in_channels,
            with_se=True,
            embed_dim=self.embed_dim,
            attention_layers=cfg.model.PVD.attentions,
            attention_fn=attention_fn,
            dropout=dropout,
            cfg=cfg,
        )

        self.fp_layers = nn.ModuleList(fp_layers)

        layers, *_ = create_mlp_components(
            in_channels=channels_fp_features,
            out_channels=[128, dropout, out_dim],
            classifier=True,
            dim=2,
            cfg=cfg,
        )
        self.classifier = nn.ModuleList(layers)

    def get_timestep_embedding(self, timesteps, device):
        if len(timesteps.shape) == 2 and timesteps.shape[1] == 1:
            timesteps = timesteps[:, 0]
        assert len(timesteps.shape) == 1, f"get shape: {timesteps.shape}"

        half_dim = self.embed_dim // 2
        emb = np.log(10000) / (half_dim - 1)
        emb = torch.from_numpy(np.exp(np.arange(0, half_dim) * -emb)).float().to(device)
        emb = timesteps[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        if self.embed_dim % 2 == 1:  # zero pad
            emb = nn.functional.pad(emb, (0, 1), "constant", 0)
        assert emb.shape == torch.Size([timesteps.shape[0], self.embed_dim])
        return emb

    def forward(self, x, t):
        (B, C, N), device = x.shape, x.device
        assert (
            C == self.input_dim + self.extra_feature_channels
        ), f"input dim: {C}, expected: {self.input_dim + self.extra_feature_channels}"

        coords = x[:, : self.input_dim, :].contiguous()
        # take coords + extra features as the feature input to the model
        features = x.clone().contiguous()

        # initialize lists
        coords_list, in_features_list = [], []
        in_features_list.append(features)

        temb = None
        if t is not None:
            if t.ndim == 0 and not len(t.shape) == 1:
                t = t.view(1).expand(B)
            temb = self.embedf(self.get_timestep_embedding(t, device))[:, :, None].expand(-1, -1, N)

        for i, sa_blocks in enumerate(self.sa_layers):
            in_features_list.append(features)
            coords_list.append(coords)
            if i > 0 and temb is not None:
                features = torch.cat([features, temb], dim=1)
                features, coords, temb, _ = sa_blocks((features, coords, temb, None))
            else:  # i == 0 or temb is None
                features, coords, temb, _ = sa_blocks((features, coords, temb, None))

        # remove first added feature in feature list
        in_features_list.pop(1)

        if self.global_att is not None:
            if isinstance(self.global_att, LinearAttention):
                features = self.global_att(features)
            elif isinstance(self.global_att, Attention):
                features = rearrange(features, "b n c -> b c n")
                features = self.global_att(features)
                features = rearrange(features, "b c n -> b n c")
            elif isinstance(self.global_att, SetTransformer):
                features = rearrange(features, "b n c -> b c n")
                temb_st = rearrange(t, "b -> b 1 1").float()
                features, _ = self.global_att(features=features, t_embed=temb_st)
                features = rearrange(features, "b c n -> b n c")

        for fp_idx, fp_blocks in enumerate(self.fp_layers):
            if temb is not None:
                features, coords, temb, _ = fp_blocks(
                    (
                        coords_list[-1 - fp_idx],
                        coords,
                        torch.cat([features, temb], dim=1),
                        in_features_list[-1 - fp_idx],
                        temb,
                        None,
                    )
                )
            else:
                features, coords, temb, _ = fp_blocks(
                    (
                        coords_list[-1 - fp_idx],
                        coords,
                        features,
                        in_features_list[-1 - fp_idx],
                        temb,
                        None,
                    )
                )

        for l in self.classifier:
            if isinstance(l, SharedMLP):
                features = l(features, None)
            else:
                features = l(features)
        return features


class PVCLionSmall(PVCNN2Unet):
    def __init__(
        self,
        out_dim: int = 3,
        input_dim: int = 3,
        embed_dim: int = 64,
        npoints: int = 2048,
        use_att: bool = True,
        use_st: bool = False,
        dropout: float = 0.1,
        extra_feature_channels: int = 3,
        width_multiplier: int = 1,
        voxel_resolution_multiplier: int = 1,
        self_cond: bool = False,
    ):
        sa_blocks = [
            # conv vfg  , sa config
            # out channels, num blocks, voxel resolution | num_centers, radius, num_neighbors, out_channels
            (
                (32, 2, 32),
                (1024, 0.1, 32, (32, 64)),
            ),
            ((64, 3, 16), (256, 0.2, 32, (64, 128))),
            ((128, 3, 8), (64, 0.4, 32, (128, 256))),
            (None, (16, 0.8, 32, (256, 256, 512))),
        ]

        # in_channels, out_channels X | out_channels, num_blocks, voxel_resolution
        fp_blocks = [
            (
                (256, 256),
                (256, 3, 8),
            ),
            ((256, 256), (256, 3, 8)),
            ((256, 128), (128, 2, 16)),
            ((128, 128, 64), (64, 2, 32)),
        ]

        super().__init__(
            out_dim=out_dim,
            input_dim=input_dim,
            embed_dim=embed_dim,
            use_att=use_att,
            use_st=use_st,
            dropout=dropout,
            sa_blocks=sa_blocks,
            fp_blocks=fp_blocks,
            extra_feature_channels=extra_feature_channels,
            width_multiplier=width_multiplier,
            voxel_resolution_multiplier=voxel_resolution_multiplier,
            self_cond=self_cond,
            flash=False,
        )
