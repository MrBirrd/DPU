import math
from functools import partial
from typing import Any

import MinkowskiEngine as ME
import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from MinkowskiEngine.MinkowskiOps import to_sparse_all
from torch import einsum, nn
from torch.nn import Module

from data.utils import FeatureVoxelConcatenation


class MinowskiIdentity(nn.Module):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__()

    def forward(self, input):
        return input


class MinkowskiGroupNorm(Module):
    r"""A group normalization layer for a sparse tensor.

    See the pytorch :attr:`torch.nn.GroupNorm` for more details.
    """

    def __init__(
        self,
        num_groups,
        num_channels,
        eps=1e-5,
        affine=True,
    ):
        super(MinkowskiGroupNorm, self).__init__()
        self.gn = torch.nn.GroupNorm(
            num_groups,
            num_channels,
            eps=eps,
            affine=affine,
        )

    def forward(self, input):
        x = input.decomposed_features
        x = torch.cat([item.unsqueeze(0) for item in x])
        # reshape to (B, C, N)
        x = rearrange(x, "B N C -> B C N")

        # do group norm
        output = self.gn(x)

        # stack features
        output = rearrange(output, "B C N -> (B N) C)")

        if isinstance(input, ME.TensorField):
            return ME.TensorField(
                output,
                coordinate_field_map_key=input.coordinate_field_map_key,
                coordinate_manager=input.coordinate_manager,
                quantization_mode=input.quantization_mode,
            )
        else:
            return ME.SparseTensor(
                output,
                coordinate_map_key=input.coordinate_map_key,
                coordinate_manager=input.coordinate_manager,
            )

    def __repr__(self):
        s = "(num_groups={}, eps={}, num_channels={}, affine={})".format(
            self.gn.num_groups,
            self.gn.eps,
            self.gn.num_channels,
            self.gn.affine,
        )
        return self.__class__.__name__ + s


def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d


def mink_conv(
    dim,
    dim_out=None,
    kernel_size=3,
    stride=1,
    dimension=-1,
    bias=True,
    new_coords=False,
):
    return ME.MinkowskiConvolution(
        dim,
        default(dim_out, dim),
        kernel_size=kernel_size,
        stride=stride,
        bias=bias,
        dimension=dimension,
        expand_coordinates=new_coords,
    )


def mink_convtTranspose(
    dim,
    dim_out=None,
    stride=1,
    kernel_size=3,
    dimension=-1,
    bias=True,
    new_coords=False,
):
    if new_coords:
        return ME.MinkowskiGenerativeConvolutionTranspose(
            dim,
            default(dim_out, dim),
            kernel_size=kernel_size,
            stride=stride,
            bias=bias,
            dimension=dimension,
        )
    else:
        return ME.MinkowskiConvolutionTranspose(
            dim,
            default(dim_out, dim),
            kernel_size=kernel_size,
            stride=stride,
            bias=bias,
            dimension=dimension,
        )


def DownsampleME(in_planes, out_planes, ds_factor=2, D=-1, batchnorm=False):
    """
    Downsamples the input tensor using Minkowski Engine.

    Args:
        in_planes (int): Number of input channels.
        out_planes (int): Number of output channels.
        stride (int, optional): Stride of the convolution. Defaults to 2.
        D (int, optional): Number of spatial dimensions. Defaults to -1.
        ks_p_down (int, optional): Kernel size of the convolution. Defaults to 2.
        batchnorm (bool, optional): Whether to use batch normalization. Defaults to False.

    Returns:
        nn.Sequential: A sequential module containing the convolution and batch normalization layers (if specified).
    """
    module_list = []
    # first upsample the channel planes
    module_list.append(
        mink_conv(
            in_planes,
            default(out_planes, in_planes),
            kernel_size=3,
            stride=1,
            dimension=D,
        )
    )
    # then downsample the spatial dimensions
    module_list.append(ME.MinkowskiMaxPooling(kernel_size=ds_factor, stride=ds_factor, dimension=D))
    if batchnorm:
        module_list.append(ME.MinkowskiBatchNorm(default(out_planes, in_planes)))
    return nn.Sequential(*module_list)


def UpsampleME(in_planes, out_planes=None, upsample_factor=2, D=-1, batchnorm=False):
    """
    Upsamples the input tensor by a factor of `stride` using transposed convolution.

    Args:
        in_planes (int): Number of input channels.
        out_planes (int, optional): Number of output channels. If not provided, defaults to `in_planes`.
        stride (int, optional): Upsampling factor. Defaults to 2.
        D (int, optional): Number of spatial dimensions. Defaults to -1.
        kernel_size (int, optional): Size of the convolution kernel. Defaults to 2.
        batchnorm (bool, optional): Whether to apply batch normalization. Defaults to False.

    Returns:
        nn.Sequential: Sequential module containing the transposed convolution and batch normalization (if applied).
    """
    module_list = []
    # first upsample the points
    module_list.append(ME.MinkowskiPoolingTranspose(kernel_size=upsample_factor, stride=upsample_factor, dimension=D))
    # then reduce the channels
    module_list.append(
        mink_convtTranspose(
            in_planes,
            out_planes,
            kernel_size=3,
            stride=1,
            dimension=D,
            new_coords=False,
        )
    )  # upsample the number of channels
    if batchnorm:
        module_list.append(ME.MinkowskiBatchNorm(default(out_planes, in_planes)))
    return nn.Sequential(*module_list)


# sinusoidal positional embeds
class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
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


# building block modules
class BlockMEScaleShift(nn.Module):
    def __init__(self, dim, dim_out, groups=8, D=-1):
        super().__init__()
        self.proj = mink_conv(dim, dim_out, kernel_size=3, dimension=D)
        # self.norm = MinkowskiGroupNorm(groups, dim_out)
        # self.norm = ME.MinkowskiFunctional.group_norm(num_groups=groups)
        self.act = ME.MinkowskiSiLU()

    def forward(self, x, scale_shift=None, time_emb=None):
        x = self.proj(x)
        # x = self.norm(x)

        if exists(scale_shift):
            scale, shift = scale_shift
            b = scale.shape[0]
            scale = scale + 1  # shift the scale to be unit scale by default
            n_pts_per_batch = [
                x.features_at(idx).shape[0] for idx in range(b)
            ]  # calculate npoint per batch as these are not the same for each batch

            scale_sparse = ME.SparseTensor(
                features=torch.cat(
                    [repeat(item, "f -> c f", c=n_pts_per_batch[idx]) for idx, item in enumerate(scale)]
                ),
                device=x.device,
                coordinate_manager=x.coordinate_manager,  # must share the same coordinate manager
                coordinate_map_key=x.coordinate_map_key,  # must share the same coordinate map key
            )
            shift_sparse = ME.SparseTensor(
                features=torch.cat(
                    [repeat(item, "f -> c f", c=n_pts_per_batch[idx]) for idx, item in enumerate(shift)]
                ),
                device=x.device,
                coordinate_manager=x.coordinate_manager,  # must share the same coordinate manager
                coordinate_map_key=x.coordinate_map_key,  # must share the same coordinate map key
            )

            x = x * scale_sparse + shift_sparse

        x = self.act(x)

        return x


class BlockMEConcat(nn.Module):
    def __init__(self, dim, dim_out, groups=8, D=-1, use_norm=True):
        super().__init__()
        self.proj = mink_conv(dim, dim_out, kernel_size=3, dimension=D)
        # self.norm = MinkowskiGroupNorm(groups, dim_out)
        self.norm = ME.MinkowskiBatchNorm(dim_out) if use_norm else MinowskiIdentity()
        # self.norm = ME.MinkowskiFunctional.group_norm(num_groups=groups)
        self.act = ME.MinkowskiSiLU()

    def forward(self, x, scale_shift=None, time_emb=None):
        # if we got a time_emb but no scale shift we should do time feature concatenation
        if exists(time_emb) and not exists(scale_shift):
            # extract batch size and the number of points per batch in sparse tensor x
            b = time_emb.shape[0]
            n_pts_per_batch = [x.features_at(idx).shape[0] for idx in range(b)]

            time_emb_sparse = ME.SparseTensor(
                features=torch.cat(
                    [repeat(item, "f -> c f", c=n_pts_per_batch[idx]) for idx, item in enumerate(time_emb)]
                ),
                device=x.device,
                coordinate_manager=x.coordinate_manager,  # must share the same coordinate manager
                coordinate_map_key=x.coordinate_map_key,  # must share the same coordinate map key
            )

            x = ME.cat((x, time_emb_sparse))

        x = self.proj(x)
        x = self.norm(x)
        x = self.act(x)

        return x


class ResnetBlockME(nn.Module):
    def __init__(self, dim, dim_out, *, D=-1, time_emb_dim=None, groups=8, concat_time_emb=True):
        super().__init__()
        self.mlp = (
            nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, dim_out * 2))
            if exists(time_emb_dim) and not concat_time_emb
            else None
        )

        self.time_emb_concat = concat_time_emb

        if concat_time_emb:
            self.block1 = BlockMEConcat(dim + time_emb_dim, dim_out, groups=groups, D=D)
            self.block2 = BlockMEConcat(dim_out, dim_out, groups=groups, D=D)

        else:
            self.block1 = BlockMEScaleShift(dim, dim_out, groups=groups, D=D)
            self.block2 = BlockMEScaleShift(dim_out, dim_out, groups=groups, D=D)

        self.res_conv = mink_conv(dim, dim_out, kernel_size=1, dimension=D) if dim != dim_out else MinowskiIdentity()

    def forward(self, x, time_emb=None):
        # calculate scale shift parameters if needed
        scale_shift = None
        if exists(self.mlp) and exists(time_emb) and not self.time_emb_concat:
            time_emb = self.mlp(time_emb)
            scale_shift = time_emb.chunk(2, dim=1)

        h = self.block1(x, scale_shift=scale_shift, time_emb=time_emb)

        h = self.block2(h)

        return h + self.res_conv(x)


def scaled_dot_product(q, k, v, mask=None):
    d_k = q.size()[-1]
    attn_logits = torch.matmul(q, k.transpose(-2, -1))
    attn_logits = attn_logits / math.sqrt(d_k)
    if mask is not None:
        attn_logits = attn_logits.masked_fill(mask == 0, -9e15)
    attention = F.softmax(attn_logits, dim=-1)
    values = torch.matmul(attention, v)
    return values, attention


class MinkAttention(nn.Module):
    def __init__(self, input_dim, embed_dim, num_heads):
        super().__init__()
        assert embed_dim % num_heads == 0, "Embedding dimension must be 0 modulo number of heads."

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # Stack all weight matrices 1...h together for efficiency
        # Note that in many implementations you see "bias=False" which is optional
        self.qkv_proj = nn.Linear(input_dim, 3 * embed_dim)
        self.o_proj = nn.Linear(embed_dim, embed_dim)

        self._reset_parameters()

    def _reset_parameters(self):
        # Original Transformer initialization, see PyTorch documentation
        nn.init.xavier_uniform_(self.qkv_proj.weight)
        self.qkv_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.o_proj.weight)
        self.o_proj.bias.data.fill_(0)

    def forward(self, input, mask=None, return_attention=False):
        x = input.decomposed_features
        x = torch.cat([item.unsqueeze(0) for item in x])

        batch_size, seq_length, embed_dim = x.size()
        qkv = self.qkv_proj(x)

        # Separate Q, K, V from linear output
        qkv = qkv.reshape(batch_size, seq_length, self.num_heads, 3 * self.head_dim)
        qkv = qkv.permute(0, 2, 1, 3)  # [Batch, Head, SeqLen, Dims]
        q, k, v = qkv.chunk(3, dim=-1)

        # Determine value outputs
        values, attention = scaled_dot_product(q, k, v, mask=mask)
        values = values.permute(0, 2, 1, 3)  # [Batch, SeqLen, Head, Dims]
        values = values.reshape(batch_size, seq_length, -1)
        output = self.o_proj(values)
        output = output.view(-1, embed_dim)

        if isinstance(input, ME.TensorField):
            return ME.TensorField(
                output,
                coordinate_field_map_key=input.coordinate_field_map_key,
                coordinate_manager=input.coordinate_manager,
                quantization_mode=input.quantization_mode,
            )
        else:
            return ME.SparseTensor(
                output,
                coordinate_map_key=input.coordinate_map_key,
                coordinate_manager=input.coordinate_manager,
            )


# model
class MinkUnet(nn.Module):
    def __init__(
        self,
        dim,
        D=1,
        init_dim=None,
        init_ds_factor=1,
        out_dim=None,
        dim_mults=(1, 1, 2, 2, 4, 4),
        downsampfactors=(4, 1, 4, 1, 4, 1),
        in_channels=3,
        in_shape=None,
        self_condition=False,
        resnet_block_groups=8,
        learned_variance=False,
        learned_sinusoidal_cond=False,
        random_fourier_features=False,
        learned_sinusoidal_dim=16,
        use_attention=False,
        voxel_feature_concat=False,  # concatenate the conditional pointcloud by per voxel featuress
        concat_time_emb=True,  # concatenate the time embeddings to the input features, otherwise use scale and shift
    ):
        super().__init__()

        # setup functions to generate sparse tensors and handle caching
        if in_shape is not None:
            coordinates = ME.dense_coordinates(in_shape)
            coordinates_b1 = ME.dense_coordinates(torch.Size([1, *in_shape[1:]]))

            self.to_sparse = ME.MinkowskiToSparseTensor(coordinates=coordinates)
            self.to_sparse_single = ME.MinkowskiToSparseTensor(coordinates=coordinates_b1)

            self.to_dense = ME.MinkowskiToDenseTensor(torch.Size(in_shape))
            self.to_dense_single = ME.MinkowskiToDenseTensor(torch.Size([1, *in_shape[1:]]))
        else:
            self.to_sparse = ME.MinkowskiToSparseTensor()
            self.to_dense = ME.MinkowskiToFeature()

        # setup voxelwise feature concatenatino
        self.voxel_feat_cat = (
            FeatureVoxelConcatenation(resolution=64, normalize=False) if voxel_feature_concat else None
        )

        self.channels = in_channels
        self.self_condition = self_condition
        input_channels = in_channels * (2 if self_condition else 1)  # TODO maybe it would be only 3

        init_dim = default(init_dim, dim)
        self.init_conv = mink_conv(input_channels, init_dim, kernel_size=5, stride=init_ds_factor, dimension=D)

        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out_down = list(zip(dims[:-1], dims[1:], downsampfactors))
        in_out_up = list(zip(dims[:-1], dims[1:], reversed(downsampfactors)))

        block_klass = partial(
            ResnetBlockME,
            D=D,
            groups=resnet_block_groups,
            concat_time_emb=concat_time_emb,
        )

        # time embeddings

        time_dim = dim * 4 if not concat_time_emb else dim

        self.random_or_learned_sinusoidal_cond = learned_sinusoidal_cond or random_fourier_features

        if self.random_or_learned_sinusoidal_cond:
            sinu_pos_emb = RandomOrLearnedSinusoidalPosEmb(learned_sinusoidal_dim, random_fourier_features)
            fourier_dim = learned_sinusoidal_dim + 1
        else:
            sinu_pos_emb = SinusoidalPosEmb(dim)
            fourier_dim = dim

        self.time_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(fourier_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim),
        )

        # layers
        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out_down)

        for ind, (dim_in, dim_out, factor) in enumerate(in_out_down):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(
                nn.ModuleList(
                    [
                        block_klass(dim_in, dim_in, time_emb_dim=time_dim),
                        block_klass(dim_in, dim_in, time_emb_dim=time_dim),
                        DownsampleME(dim_in, dim_out, ds_factor=factor, D=D)
                        if not is_last
                        else mink_conv(dim_in, dim_out, kernel_size=3, dimension=D),
                    ]
                )
            )

        # middle block which can be used as a bottleneck with atttention
        mid_dim = dims[-1]
        self.mid_block1 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)
        self.mid_attn = MinkAttention(mid_dim, embed_dim=mid_dim, num_heads=8) if use_attention else MinowskiIdentity()
        self.mid_block2 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)

        # upsampling layers
        for ind, (dim_in, dim_out, factor) in enumerate(reversed(in_out_up)):
            is_last = ind == (len(in_out_up) - 1)

            self.ups.append(
                nn.ModuleList(
                    [
                        block_klass(dim_out + dim_in, dim_out, time_emb_dim=time_dim),
                        block_klass(dim_out + dim_in, dim_out, time_emb_dim=time_dim),
                        UpsampleME(dim_out, dim_in, D=D, upsample_factor=factor)
                        if not is_last
                        else mink_conv(dim_in, dim_out, kernel_size=3, dimension=D),
                    ]
                )
            )

        default_out_dim = in_channels * (1 if not learned_variance else 2)
        self.out_dim = default(out_dim, default_out_dim)

        self.final_res_block = block_klass(dim * 2, dim, time_emb_dim=time_dim)

        if init_ds_factor == 1:
            # if no initial downsampling factor, use standard convolution to compress the channels
            self.final_conv = mink_conv(dim_in, self.out_dim, kernel_size=1, dimension=D)
        elif init_ds_factor > 1:
            # if there is an initial downsampling factor, use transposed convolution to upsample the points too
            self.final_conv = mink_convtTranspose(dim, self.out_dim, kernel_size=2, stride=init_ds_factor, dimension=D)
        else:
            raise ValueError(f"initial downsampling factor must be positive, but is {init_ds_factor}")

    def forward(self, x_coords, time, cond=None, x_self_cond=None):
        # handle conditioning
        if cond is not None:
            if self.voxel_feat_cat is not None:
                stacked_feats = self.voxel_feat_cat(
                    x1_features=x_coords,
                    x2_features=cond,
                    x1_coords=x_coords,
                    x2_coords=cond,
                )
            else:
                stacked_feats = torch.cat((x_coords, cond), dim=1)

        # generate sparse tensor
        if x_coords.shape[0] == 1:
            x_sparse = self.to_sparse_single(x_coords)
        else:
            x_sparse = self.to_sparse(x_coords)

        x = self.init_conv(x_sparse)

        r = x

        t = self.time_mlp(time)

        h = []

        for block1, block2, downsample in self.downs:
            x = block1(x, t)
            h.append(x)

            x = block2(x, t)
            h.append(x)

            x = downsample(x)

        x = self.mid_block1(x, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t)

        for block1, block2, upsample in self.ups:
            skip_x1 = h.pop()
            skip_x2 = h.pop()

            x = ME.cat((x, skip_x1))
            x = block1(x, t)

            x = ME.cat((x, skip_x2))
            x = block2(x, t)

            x = upsample(x)

        x = ME.cat((x, r))

        x = self.final_res_block(x, t)
        x = self.final_conv(x)

        if x_coords.shape[0] == 1:
            xd = self.to_dense_single(x)
        else:
            xd = self.to_dense(x)

        return xd


def to_parse_tensor(coords, feats=None, voxel_size=0.001):
    """
    Converts the input coordinates and features to a sparse tensor.

    Args:
        coords (torch.Tensor): The input coordinates.
        feats (torch.Tensor): The input features.
        voxel_size (float): The voxel size for scaling the coordinates.

    Returns:
        ME.SparseTensor: The sparse tensor with the converted coordinates and features.
    """

    # switch from channel first to channel last
    coords = coords.permute(0, 2, 1)
    b, n, c = coords.shape

    # scale for discrete coordinates
    coords = coords / voxel_size

    if feats is None:
        feats = torch.zeros((b * n, 3)).to(coords.device)
    if feats.ndim == 3:
        assert feats.shape[-1] == n, "feats must be of shape (b, c, n)"
        feats = rearrange(feats, "b c n -> (b n) c")

    stensor = ME.SparseTensor(
        features=feats,  # Convert to a tensor
        coordinates=ME.utils.batched_coordinates(
            [c for c in coords], dtype=torch.float32, device=coords.device
        ),  # coordinates must be defined in a integer grid. If the scale
        quantization_mode=ME.SparseTensorQuantizationMode.UNWEIGHTED_AVERAGE,  # when used with continuous coordinates, average features in the same coordinate
    )
    return stensor


if __name__ == "__main__":
    from torch.optim import Adam
    from tqdm import tqdm

    B = 1

    net = MinkUnet(dim=64, D=3, use_attention=True).cuda()
    in_test = torch.randn(B, 2048, 3).cuda()
    time = torch.rand(B).cuda()
    epsilon = torch.randn_like(in_test).cuda()

    optimizer = Adam(net.parameters(), lr=1e-3)
    voxel_size = 0.005

    # create fake mixing inputs
    alphas = torch.rand(B).cuda()
    betas = 1 - alphas

    model_in = torch.tensor([]).cuda()
    for idx in range(B):
        mix = alphas[idx] * in_test[idx] + betas[idx] * epsilon[idx]
        model_in = torch.cat((model_in, mix.unsqueeze(0)))
    sin = to_parse_tensor(coords=model_in, feats=model_in.view(-1, 3), voxel_size=voxel_size)

    print("Testing overfitting on a single noisy prediction")
    pbar = tqdm(range(500), desc="Overfitting")
    for _ in pbar:
        out = net(sin, time)
        loss = F.mse_loss(out, epsilon)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        pbar.set_description(f"Overfitting: {loss.item():.6f}")
