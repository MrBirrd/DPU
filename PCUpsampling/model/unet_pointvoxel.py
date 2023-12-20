# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.

import numpy as np
import torch
import torch.nn as nn
from einops import rearrange

from model.set_transformer import SetTransformer

from .pvcnn2_ada import (
    LinearAttention,
    SharedMLP,
    create_mlp_components,
    create_pointnet2_fp_modules,
    create_pointnet2_sa_components,
)
from .pvcnn_generation import PVCNN2Base
from .attention import Attention


class PVCNN2Unet(nn.Module):
    """
    copied and modified from https://github.com/alexzhou907/PVD/blob/9747265a5f141e5546fd4f862bfa66aa59f1bd33/model/pvcnn_generation.py#L172
    """

    def __init__(
        self,
        out_dim=3,
        embed_dim=64,
        use_att=True,
        use_st=False,
        dropout=0.1,
        extra_feature_channels=3,
        input_dim=3,
        width_multiplier=1,
        voxel_resolution_multiplier=1,
        time_emb_scales=1.0,
        verbose=True,
        condition_input=False,
        self_cond=False,
        flash=False,
        point_as_feat=1,
        cfg={},
        sa_blocks={},
        fp_blocks={},
        st_params={},
    ):
        super().__init__()
        self.input_dim = input_dim

        self.sa_blocks = sa_blocks
        self.fp_blocks = fp_blocks
        self.point_as_feat = point_as_feat
        self.condition_input = condition_input
        assert extra_feature_channels >= 0
        self.extra_feature_channels = extra_feature_channels
        self.time_emb_scales = time_emb_scales
        self.embed_dim = embed_dim
        self.self_condition = self_cond

        if self.embed_dim > 0:  # has time embedding
            # for prior model, we have time embedding, for VAE model, no time embedding
            self.embedf = nn.Sequential(
                nn.Linear(embed_dim, embed_dim),
                nn.LeakyReLU(0.1, inplace=True),
                nn.Linear(embed_dim, embed_dim),
            )

        (
            sa_layers,
            sa_in_channels,
            channels_sa_features,
            _,
        ) = create_pointnet2_sa_components(
            input_dim=input_dim,
            sa_blocks=self.sa_blocks,
            extra_feature_channels=extra_feature_channels,
            with_se=True,
            embed_dim=embed_dim,  # time embedding dim
            use_att=use_att,
            dropout=dropout,
            width_multiplier=width_multiplier,
            voxel_resolution_multiplier=voxel_resolution_multiplier,
            verbose=verbose,
            cfg=cfg,
        )
        self.sa_layers = nn.ModuleList(sa_layers)

        if use_att:
            if use_st:
                self.global_att = SetTransformer(
                    n_layers=st_params["layers"],
                    feature_dim=channels_sa_features,
                    num_inducers=st_params["inducers"],
                    t_embed_dim=1,
                    num_groups=st_params["gn_groups"],
                )
            else:
                if flash:
                    self.global_att = Attention(
                        dim=channels_sa_features,
                        heads=8,
                        norm=False,
                        flash=True,
                        # time_cond_dim=embed_dim,
                    )
                else:
                    self.global_att = LinearAttention(channels_sa_features, 8, verbose=verbose)
        else:
            self.global_att = None

        # only use extra features in the last fp module
        sa_in_channels[0] = extra_feature_channels + input_dim

        fp_layers, channels_fp_features = create_pointnet2_fp_modules(
            fp_blocks=self.fp_blocks,
            in_channels=channels_sa_features,
            sa_in_channels=sa_in_channels,
            with_se=True,
            embed_dim=embed_dim,
            use_att=use_att,
            dropout=dropout,
            width_multiplier=width_multiplier,
            voxel_resolution_multiplier=voxel_resolution_multiplier,
            verbose=verbose,
            cfg=cfg,
        )

        self.fp_layers = nn.ModuleList(fp_layers)

        layers, *_ = create_mlp_components(
            in_channels=channels_fp_features,
            out_channels=[128, dropout, out_dim],  # was 0.5
            classifier=True,
            dim=2,
            width_multiplier=width_multiplier,
            cfg=cfg,
        )
        self.classifier = nn.ModuleList(layers)

    def get_timestep_embedding(self, timesteps, device):
        if len(timesteps.shape) == 2 and timesteps.shape[1] == 1:
            timesteps = timesteps[:, 0]
        assert len(timesteps.shape) == 1, f"get shape: {timesteps.shape}"
        timesteps = timesteps * self.time_emb_scales

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


class PVCAdaptive(PVCNN2Unet):
    def __init__(
        self,
        out_dim: int = 3,
        input_dim: int = 3,
        embed_dim: int = 64,
        channels: list = [32, 64, 128, 256, 512],
        npoints: int = 2048,
        use_att: bool = True,
        use_st: bool = False,
        dropout: float = 0.1,
        extra_feature_channels: int = 3,
        width_multiplier: int = 1,
        voxel_resolution_multiplier: int = 1,
        self_cond: bool = False,
        st_params: dict = {},
    ):
        voxel_resolutions = [32, 16, 8, 8]
        n_sa_blocks = [2, 2, 3, 4]
        n_fp_blocks = [2, 2, 3, 4]
        n_centers = [npoints // 4, npoints // 4**2, npoints // 4**3, npoints // 4**4]
        radius = [0.1, 0.2, 0.4, 0.8]

        sa_blocks = [
            # conv vfg  , sa config
            # out channels, num blocks, voxel resolution | num_centers, radius, num_neighbors, out_channels
            (
                (channels[0], n_sa_blocks[0], voxel_resolutions[0]),
                (n_centers[0], radius[0], 32, (channels[0], channels[1])),
            ),
            (
                (channels[1], n_sa_blocks[1], voxel_resolutions[1]),
                (n_centers[1], radius[1], 32, (channels[1], channels[2])),
            ),
            (
                (channels[2], n_sa_blocks[2], voxel_resolutions[2]),
                (n_centers[2], radius[2], 32, (channels[2], channels[3])),
            ),
            (None, (n_centers[3], radius[3], 32, (channels[3], channels[3], channels[4]))),
        ]

        # in_channels, out_channels X | out_channels, num_blocks, voxel_resolution
        fp_blocks = [
            ((channels[3], channels[3]), (channels[3], n_fp_blocks[3], voxel_resolutions[3])),
            ((channels[3], channels[3]), (channels[3], n_fp_blocks[2], voxel_resolutions[2])),
            ((channels[3], channels[2]), (channels[2], n_fp_blocks[1], voxel_resolutions[1])),
            ((channels[2], channels[2], channels[1]), (channels[1], n_fp_blocks[0], voxel_resolutions[0])),
        ]

        super().__init__(
            out_dim=out_dim,
            input_dim=input_dim,
            embed_dim=embed_dim,
            use_att=use_att,
            use_st=use_st,
            st_params=st_params,
            dropout=dropout,
            sa_blocks=sa_blocks,
            fp_blocks=fp_blocks,
            extra_feature_channels=extra_feature_channels,
            width_multiplier=width_multiplier,
            voxel_resolution_multiplier=voxel_resolution_multiplier,
            self_cond=self_cond,
            flash=False,
        )


class PVCNN2(PVCNN2Base):
    sa_blocks = [
        # conv vfg  , sa config
        (
            (32, 2, 32),
            (1024, 0.1, 32, (32, 64)),
        ),  # out channels, num blocks, voxel resolution | num_centers, radius, num_neighbors, out_channels
        ((64, 3, 16), (256, 0.2, 32, (64, 128))),
        ((128, 3, 8), (64, 0.4, 32, (128, 256))),
        (None, (16, 0.8, 32, (256, 256, 512))),
    ]
    fp_blocks = [
        (
            (256, 256),
            (256, 3, 8),
        ),  # in_channels, out_channels X | out_channels, num_blocks, voxel_resolution
        ((256, 256), (256, 3, 8)),
        ((256, 128), (128, 2, 16)),
        ((128, 128, 64), (64, 2, 32)),
    ]

    def __init__(
        self,
        out_dim,
        input_dim,
        embed_dim,
        use_att,
        dropout,
        extra_feature_channels=3,
        width_multiplier=1,
        voxel_resolution_multiplier=1,
    ):
        super().__init__(
            out_dim=out_dim,
            input_dim=input_dim,
            embed_dim=embed_dim,
            use_att=use_att,
            dropout=dropout,
            extra_feature_channels=extra_feature_channels,
            width_multiplier=width_multiplier,
            voxel_resolution_multiplier=voxel_resolution_multiplier,
        )


if __name__ == "__main__":
    net = PVCLionSmall(input_dim=3, extra_feature_channels=0)
    net = net.cuda()

    test = torch.rand(1, 3, 2048).cuda()
    t = torch.rand(1).cuda()
    test = net(test, t)
    print(test.shape)
