from typing import Any

from einops import rearrange
from torch import Tensor, cat, nn

from third_party.gecco_torch.models.set_transformer import SetTransformer


class LinearLift(nn.Module):
    """
    Embeds the 3d geometry (xyz points) in a higher dimensional space, passes it through
    the SetTransformer, and then maps it back to 3d. "Lift" refers to the embedding action.
    This class is used in the unconditional ShapeNet experiments.
    """

    def __init__(
        self,
        inner: SetTransformer,
        feature_dim: int,
        in_dim: int = 3,
        out_dim: int = 3,
        do_norm: bool = True,
        self_conditioning = False,
    ):
        super().__init__()
        self.lift = nn.Linear(in_dim, feature_dim)
        self.inner = inner
        self.self_condition = self_conditioning
        self.in_dim = in_dim

        if do_norm:
            self.lower = nn.Sequential(
                nn.LayerNorm(feature_dim, elementwise_affine=False),
                nn.Linear(feature_dim, out_dim),
            )
        else:
            self.lower = nn.Linear(feature_dim, out_dim)

    def forward(
        self,
        geometry: Tensor,
        embed: Tensor,
        cond = None,
        x_self_cond = None,
        do_cache: bool = False,
        cache: list[Tensor] | None = None,
    ) -> tuple[Tensor, list[Tensor] | None]:

        n = geometry.shape[-1]

        if cond is not None and self.in_dim == 3:
            geometry = cat([geometry, cond], dim=-1)
        elif cond is not None and self.in_dim == 6:
            geometry = cat([geometry, cond], dim=1)

        geometry = rearrange(geometry, "b d n -> b n d")
        embed = rearrange(embed, "b -> b 1 1").float()

        features = self.lift(geometry)
        features, out_cache = self.inner(features, embed, do_cache, cache)
        points = self.lower(features)
        points = rearrange(points, "b n d -> b d n")[..., :n]

        return points
