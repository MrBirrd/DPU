import math
from collections import namedtuple
from functools import partial, wraps
from random import random
import torch
import torch.nn.functional as F
from beartype import beartype
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange
from ema_pytorch import EMA
from packaging import version
from PIL import Image
from torch import einsum, nn
from torch.special import expm1
from torchvision import transforms as T
from torchvision import utils
from tqdm.auto import tqdm

# constants
FlashAttentionConfig = namedtuple('FlashAttentionConfig', ['enable_flash', 'enable_math', 'enable_mem_efficient'])

# helpers

def exists(val):
    return val is not None

def once(fn):
    called = False
    @wraps(fn)
    def inner(x):
        nonlocal called
        if called:
            return
        called = True
        return fn(x)
    return inner

print_once = once(print)

# main class

class Attend(nn.Module):
    def __init__(
        self,
        dropout = 0.,
        flash = False
    ):
        super().__init__()
        self.dropout = dropout
        self.attn_dropout = nn.Dropout(dropout)

        self.flash = flash
        assert not (flash and version.parse(torch.__version__) < version.parse('2.0.0')), 'in order to use flash attention, you must be using pytorch 2.0 or above'

        # determine efficient attention configs for cuda and cpu

        self.cpu_config = FlashAttentionConfig(True, True, True)
        self.cuda_config = None

        if not torch.cuda.is_available() or not flash:
            return

        device_properties = torch.cuda.get_device_properties(torch.device('cuda'))

        if device_properties.major == 8 and device_properties.minor == 0:
            print_once('A100 GPU detected, using flash attention if input tensor is on cuda')
            self.cuda_config = FlashAttentionConfig(True, False, False)
        else:
            print_once('Non-A100 GPU detected, using math or mem efficient attention if input tensor is on cuda')
            self.cuda_config = FlashAttentionConfig(False, True, True)

    def flash_attn(self, q, k, v, mask = None):
        _, heads, q_len, _, k_len, is_cuda, device = *q.shape, k.shape[-2], q.is_cuda, q.device

        # Check if mask exists and expand to compatible shape
        # The mask is B L, so it would have to be expanded to B H N L

        if exists(mask):
            mask = mask.expand(-1, heads, q_len, -1)

        # Check if there is a compatible device for flash attention

        config = self.cuda_config if is_cuda else self.cpu_config

        # pytorch 2.0 flash attn: q, k, v, mask, dropout, softmax_scale

        with torch.backends.cuda.sdp_kernel(**config._asdict()):
            out = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask = mask,
                dropout_p = self.dropout if self.training else 0.
            )

        return out

    def forward(self, q, k, v, mask = None):
        """
        einstein notation
        b - batch
        h - heads
        n, i, j - sequence length (base sequence length, source, target)
        d - feature dimension
        """

        q_len, k_len, device = q.shape[-2], k.shape[-2], q.device

        scale = q.shape[-1] ** -0.5

        if exists(mask) and mask.ndim != 4:
            mask = rearrange(mask, 'b j -> b 1 1 j')

        if self.flash:
            return self.flash_attn(q, k, v, mask = mask)

        # similarity

        sim = einsum(f"b h i d, b h j d -> b h i j", q, k) * scale

        # key padding mask

        if exists(mask):
            sim = sim.masked_fill(~mask, -torch.finfo(sim.dtype).max)

        # attention

        attn = sim.softmax(dim=-1)
        attn = self.attn_dropout(attn)

        # aggregate values

        out = einsum(f"b h i j, b h j d -> b h i d", attn, v)

        return out

# helpers functions

def exists(x):
    return x is not None

def identity(x):
    return x

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

def divisible_by(numer, denom):
    return (numer % denom) == 0

def safe_div(numer, denom, eps = 1e-10):
    return numer / denom.clamp(min = eps)

def cycle(dl):
    while True:
        for data in dl:
            yield data

def has_int_squareroot(num):
    num_sqrt = math.sqrt(num)
    return int(num_sqrt) == num_sqrt

def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr

def convert_image_to(img_type, image):
    if image.mode != img_type:
        return image.convert(img_type)
    return image

def Sequential(*mods):
    return nn.Sequential(*filter(exists, mods))

# use layernorm without bias, more stable

class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))
        self.register_buffer("beta", torch.zeros(dim))

    def forward(self, x):
        return F.layer_norm(x, x.shape[-1:], self.gamma, self.beta)

class MultiHeadedRMSNorm(nn.Module):
    def __init__(self, dim, heads = 1):
        super().__init__()
        self.scale = dim ** 0.5
        self.gamma = nn.Parameter(torch.ones(heads, 1, dim))

    def forward(self, x):
        return F.normalize(x, dim = -1) * self.scale * self.gamma

# positional embeds

class LearnedSinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        assert (dim % 2) == 0
        half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(half_dim))

    def forward(self, x):
        x = rearrange(x, 'b -> b 1')
        freqs = x * rearrange(self.weights, 'd -> 1 d') * 2 * math.pi
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim = -1)
        fouriered = torch.cat((x, fouriered), dim = -1)
        return fouriered

# used for self attention
class LinearAttention(nn.Module):
    def __init__(
        self,
        dim,
        heads = 4,
        dim_head = 32,
        norm = False,
        qk_norm = False,
        time_cond_dim = None
    ):
        super().__init__()
        hidden_dim = dim_head * heads
        self.scale = dim_head ** -0.5
        self.heads = heads

        self.time_cond = None

        if exists(time_cond_dim):
            self.time_cond = nn.Sequential(
                nn.SiLU(),
                nn.Linear(time_cond_dim, dim * 2),
                Rearrange('b d -> b 1 d')
            )

            nn.init.zeros_(self.time_cond[-2].weight)
            nn.init.zeros_(self.time_cond[-2].bias)

        self.norm = LayerNorm(dim) if norm else nn.Identity()

        self.to_qkv = nn.Linear(dim, hidden_dim * 3, bias = False)

        self.qk_norm = qk_norm
        if qk_norm:
            self.q_norm = MultiHeadedRMSNorm(dim_head, heads)
            self.k_norm = MultiHeadedRMSNorm(dim_head, heads)

        self.to_out = nn.Sequential(
            nn.Linear(hidden_dim, dim, bias = False),
            LayerNorm(dim)
        )

    def forward(
        self,
        x,
        time = None
    ):
        h = self.heads
        x = self.norm(x)

        if exists(self.time_cond):
            assert exists(time)
            scale, shift = self.time_cond(time).chunk(2, dim = -1)
            x = (x * (scale + 1)) + shift

        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)

        if self.qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        q = q.softmax(dim = -1)
        k = k.softmax(dim = -2)

        q = q * self.scale

        context = torch.einsum('b h n d, b h n e -> b h d e', k, v)

        out = torch.einsum('b h d e, b h n d -> b h n e', context, q)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

# cross attention
class Attention(nn.Module):
    def __init__(
        self,
        dim,
        dim_context = None,
        heads = 4,
        dim_head = 32,
        norm = False,
        norm_context = False,
        time_cond_dim = None,
        flash = False,
        qk_norm = False
    ):
        super().__init__()
        hidden_dim = dim_head * heads
        dim_context = default(dim_context, dim)

        self.time_cond = None

        if exists(time_cond_dim):
            self.time_cond = nn.Sequential(
                nn.SiLU(),
                nn.Linear(time_cond_dim, dim * 2),
                Rearrange('b d -> b 1 d')
            )

            nn.init.zeros_(self.time_cond[-2].weight)
            nn.init.zeros_(self.time_cond[-2].bias)

        self.scale = dim_head ** -0.5
        self.heads = heads

        self.norm = LayerNorm(dim) if norm else nn.Identity()
        self.norm_context = LayerNorm(dim_context) if norm_context else nn.Identity()

        self.to_q = nn.Linear(dim, hidden_dim, bias = False)
        self.to_kv = nn.Linear(dim_context, hidden_dim * 2, bias = False)
        self.to_out = nn.Linear(hidden_dim, dim, bias = False)

        self.qk_norm = qk_norm
        if qk_norm:
            self.q_norm = MultiHeadedRMSNorm(dim_head, heads)
            self.k_norm = MultiHeadedRMSNorm(dim_head, heads)

        self.attend = Attend(flash = flash)

    def forward(
        self,
        x,
        context = None,
        time = None
    ):
        h = self.heads

        if exists(context):
            context = self.norm_context(context)

        x = self.norm(x)

        context = default(context, x)

        if exists(self.time_cond):
            assert exists(time)
            scale, shift = self.time_cond(time).chunk(2, dim = -1)
            x = (x * (scale + 1)) + shift

        qkv = (self.to_q(x), *self.to_kv(context).chunk(2, dim = -1))
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)

        if self.qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        out = self.attend(q, k, v)

        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class FeedForward(nn.Module):
    def __init__(self, dim, mult = 4, time_cond_dim = None):
        super().__init__()
        self.norm = LayerNorm(dim)

        self.time_cond = None

        if exists(time_cond_dim):
            self.time_cond = nn.Sequential(
                nn.SiLU(),
                nn.Linear(time_cond_dim, dim * 2),
                Rearrange('b d -> b 1 d')
            )

            nn.init.zeros_(self.time_cond[-2].weight)
            nn.init.zeros_(self.time_cond[-2].bias)

        inner_dim = int(dim * mult)
        self.net = nn.Sequential(
            nn.Linear(dim, inner_dim),
            nn.GELU(),
            nn.Linear(inner_dim, dim)
        )

    def forward(self, x, time = None):
        x = self.norm(x)

        if exists(self.time_cond):
            assert exists(time)
            scale, shift = self.time_cond(time).chunk(2, dim = -1)
            x = (x * (scale + 1)) + shift

        return self.net(x)

# model
class RINBlock(nn.Module):
    def __init__(
        self,
        pc_dim,
        latent_self_attn_depth,
        dim_latent = None,
        final_norm = True,
        **attn_kwargs
    ):
        super().__init__()
        dim_latent = default(dim_latent, pc_dim)

        # reading operation by attending to the input and then MLP by ff
        self.latents_attend_to_pc = Attention(dim_latent, dim_context = pc_dim, norm = True, norm_context = True, **attn_kwargs)
        self.latents_cross_attn_ff = FeedForward(dim_latent)

        # K times self attention + MLP
        self.latent_self_attns = nn.ModuleList([])
        for _ in range(latent_self_attn_depth):
            self.latent_self_attns.append(nn.ModuleList([
                Attention(dim_latent, norm = True, **attn_kwargs),
                FeedForward(dim_latent)
            ]))

        # final norm for latent K bloks
        self.latent_final_norm = LayerNorm(dim_latent) if final_norm else nn.Identity()

        # back to the series space by cross attention and MLP
        self.pc_attend_to_latents = Attention(pc_dim, dim_context = dim_latent, norm = True, norm_context = True, **attn_kwargs)
        self.pc_cross_attn_ff = FeedForward(pc_dim)

    def forward(self, patches, latents, t):

        # latents extract or cluster information from the patches
        latents = self.latents_attend_to_pc(latents, patches, time = t) + latents

        latents = self.latents_cross_attn_ff(latents, time = t) + latents

        # latent self attention
        for attn, ff in self.latent_self_attns:
            latents = attn(latents, time = t) + latents
            latents = ff(latents, time = t) + latents

        # patches attend to the latents
        patches = self.pc_attend_to_latents(patches, latents, time = t) + patches
        patches = self.pc_cross_attn_ff(patches, time = t) + patches

        latents = self.latent_final_norm(latents)
        return patches, latents

class RIN(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        lift_dimension=0,
        depth = 6,                      # number of RIN blocks
        latent_self_attn_depth = 2,     # how many self attentions for the latent per each round of cross attending from pixel space to latents and back
        dim_latent = 768,              # will default to image dim (dim)
        num_latents = 256,              # they still had to use a fair amount of latents for good results (256), in line with the Perceiver line of papers from Deepmind
        learned_sinusoidal_dim = 16,
        latent_token_time_cond = False, # whether to use 1 latent token as time conditioning, or do it the adaptive layernorm way (which is highly effective as shown by some other papers "Paella" - Dominic Rampas et al.)
        **attn_kwargs
    ):
        super().__init__()

        self.lift_dimension = lift_dimension
        
        if self.lift_dimension > 0:
            self.uplift = nn.Linear(in_dim, self.lift_dimension)
            in_dim = lift_dimension
        
        dim_latent = default(dim_latent, in_dim)
        
        self.in_dim = in_dim    
        self.out_dim = out_dim
        self.lift_dimension = lift_dimension

        # time conditioning
        sinu_pos_emb = LearnedSinusoidalPosEmb(learned_sinusoidal_dim)
        time_dim = in_dim * 4
        fourier_dim = learned_sinusoidal_dim + 1
    
        self.latent_token_time_cond = latent_token_time_cond
        time_output_dim = dim_latent if latent_token_time_cond else time_dim

        self.time_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(fourier_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_output_dim)
        )
        
        self.to_points = nn.Sequential(
            LayerNorm(in_dim),
            nn.Linear(in_dim, self.out_dim),
        )
            
        self.latents = nn.Parameter(torch.randn(num_latents, dim_latent))
        
        nn.init.normal_(self.latents, std = 0.02)

        self.init_self_cond_latents = nn.Sequential(
            FeedForward(dim_latent),
            LayerNorm(dim_latent)
        )

        nn.init.zeros_(self.init_self_cond_latents[-1].gamma)

        # the main RIN body parameters  - another attention is all you need moment
        if not latent_token_time_cond:
            attn_kwargs = {**attn_kwargs, 'time_cond_dim': time_dim}

        self.blocks = nn.ModuleList([RINBlock(in_dim, dim_latent = dim_latent, latent_self_attn_depth = latent_self_attn_depth, **attn_kwargs) for _ in range(depth)])

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(
        self,
        x,
        time,
        cond = None,
        x_self_cond = None,
        latent_self_cond = None,
        return_latents = False
    ):
        batch, n, c = x.shape
        
        if c > n:
            x = x.transpose(1, 2)
            cond = cond.transpose(1, 2) if exists(cond) else None
            x_self_cond = x_self_cond.transpose(1, 2) if exists(x_self_cond) else None
            transposed = True
        else:
            transposed = False
        
        x_self_cond = default(x_self_cond, lambda: torch.zeros_like(x))
        
        x = torch.cat((x_self_cond, x), dim = -1)

        if cond is not None:
            x = torch.cat((x, cond), dim = -1)
        
        # prepare time conditioning
        t = self.time_mlp(time)

        # prepare latents
        latents = repeat(self.latents, 'n d -> b n d', b = batch)

        # the warm starting of latents as in the paper

        if exists(latent_self_cond):
            latents = latents + self.init_self_cond_latents(latent_self_cond)

        # whether the time conditioning is to be treated as one latent token or for projecting into scale and shift for adaptive layernorm

        if self.latent_token_time_cond:
            t = rearrange(t, 'b d -> b 1 d')
            latents = torch.cat((latents, t), dim = -2)

        # the recurrent interface network body
        patches = x

        patches = self.uplift(patches) if self.lift_dimension > 0 else patches
        
        for block in self.blocks:
            patches, latents = block(patches, latents, t)

        points = self.to_points(patches)

        if not return_latents:
            return points

        # remove time conditioning token, if that is the settings
        if self.latent_token_time_cond:
            latents = latents[:, :-1]

        if transposed:
            points = points.transpose(1, 2)
        
        return points, latents


if __name__ == "__main__":
    npoints = 2048
    B = 16

    model = RIN(
        in_dim=6,
        out_dim=3,
        depth=4,
        lift_dimension=128,
        dim_latent=256,
        latent_self_attn_depth=2,
        latent_token_time_cond=True,
        num_latents=256,
        flash=True,
    )
    
    model.to('cuda')

    loss = F.mse_loss
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)

    test_cloud = torch.randn(B, npoints, 3).to('cuda')
    times = torch.zeros((B,), device = "cuda").float().uniform_(0, 1.)

    tqdm_bar = tqdm(range(10000))
    
    for _ in tqdm_bar:
        out = model(x=test_cloud, time=times)
        loss = F.mse_loss(out, test_cloud)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        tqdm_bar.set_description(f"Loss: {loss.item():.4f}")
