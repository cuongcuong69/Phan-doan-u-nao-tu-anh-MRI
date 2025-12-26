# -*- coding: utf-8 -*-
"""
models/swin_unet.py

Swin-Unet (pure transformer decoder) aligned with:
  https://github.com/HuCaoFighting/Swin-Unet

Key choices for this project:
- Input image_size fixed to 224 (best match for ImageNet pretrained Swin-T)
- in_chans=4 (MRI modalities), num_classes=4 (0..3)
- Optional pretrained init for ENCODER ONLY via timm:
    use_imagenet_pretrained_encoder=True/False
  (weights are NOT frozen; encoder still updates during training)

Run quick test:
  python -m models.swin_unet
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Tuple, List, Dict, Any, Optional

import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from einops import rearrange

# timm new import path (avoid FutureWarning)
from timm.layers import DropPath, to_2tuple, trunc_normal_


# =============================================================================
# Small utils
# =============================================================================

def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def _safe_print(msg: str):
    try:
        print(msg)
    except Exception:
        pass


# =============================================================================
# MLP + Window partition helpers (same as author style)
# =============================================================================

class Mlp(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer=nn.GELU,
        drop: float = 0.0
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def window_partition(x: torch.Tensor, window_size: int) -> torch.Tensor:
    """
    Args:
        x: (B, H, W, C)
        window_size: int
    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows: torch.Tensor, window_size: int, H: int, W: int) -> torch.Tensor:
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


# =============================================================================
# Window Attention (W-MSA / SW-MSA)
# =============================================================================

class WindowAttention(nn.Module):
    """
    Window based multi-head self attention (W-MSA) module with relative position bias.
    """

    def __init__(
        self,
        dim: int,
        window_size: Tuple[int, int],
        num_heads: int,
        qkv_bias: bool = True,
        qk_scale: Optional[float] = None,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        super().__init__()
        self.dim = dim
        self.window_size = window_size  # (Wh, Ww)
        self.num_heads = num_heads

        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # relative position bias table: (2*Wh-1 * 2*Ww-1, nH)
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads)
        )

        # pair-wise relative position index for each token inside the window
        coords_h = torch.arange(window_size[0])
        coords_w = torch.arange(window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing="ij"))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += window_size[0] - 1
        relative_coords[:, :, 1] += window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index, persistent=False)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=0.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        x: (num_windows*B, N, C)
        mask: (num_windows, N, N) or None
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # each: (B_, nH, N, head_dim)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))  # (B_, nH, N, N)

        # relative bias
        rel = self.relative_position_bias_table[self.relative_position_index.view(-1)]
        rel = rel.view(N, N, -1).permute(2, 0, 1).contiguous()  # (nH, N, N)
        attn = attn + rel.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N)
            attn = attn + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)

        attn = self.softmax(attn)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


# =============================================================================
# Swin Transformer Block
# =============================================================================

class SwinTransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        input_resolution: Tuple[int, int],
        num_heads: int,
        window_size: int = 7,
        shift_size: int = 0,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        qk_scale: Optional[float] = None,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads

        self.window_size = window_size
        self.shift_size = shift_size

        if min(input_resolution) <= window_size:
            self.shift_size = 0
            self.window_size = min(input_resolution)

        assert 0 <= self.shift_size < self.window_size

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim=dim,
            window_size=to_2tuple(self.window_size),
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)

        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        # attention mask for shifted windows
        if self.shift_size > 0:
            H, W = input_resolution
            img_mask = torch.zeros((1, H, W, 1))
            h_slices = (
                slice(0, -self.window_size),
                slice(-self.window_size, -self.shift_size),
                slice(-self.shift_size, None),
            )
            w_slices = (
                slice(0, -self.window_size),
                slice(-self.window_size, -self.shift_size),
                slice(-self.shift_size, None),
            )
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            mask_windows = window_partition(img_mask, self.window_size)  # (nW, ws, ws, 1)
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, 0.0)
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, H*W, C)
        """
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, f"Wrong token length: {L} vs {H*W}"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # (nW*B, ws, ws, C)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)

        # attention
        attn_windows = self.attn(x_windows, mask=self.attn_mask)

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)

        # reverse shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x

        x = x.view(B, H * W, C)

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


# =============================================================================
# Patch Merging / Expand
# =============================================================================

class PatchMerging(nn.Module):
    def __init__(self, input_resolution: Tuple[int, int], dim: int, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, H*W, C)
        """
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W
        assert H % 2 == 0 and W % 2 == 0

        x = x.view(B, H, W, C)
        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3], dim=-1)  # (B, H/2, W/2, 4C)
        x = x.view(B, -1, 4 * C)

        x = self.norm(x)
        x = self.reduction(x)
        return x


class PatchExpand(nn.Module):
    def __init__(self, input_resolution: Tuple[int, int], dim: int, dim_scale: int = 2, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.expand = nn.Linear(dim, 2 * dim, bias=False) if dim_scale == 2 else nn.Identity()
        self.norm = norm_layer(dim // dim_scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, H*W, C)
        """
        H, W = self.input_resolution
        x = self.expand(x)
        B, L, C = x.shape
        assert L == H * W

        x = x.view(B, H, W, C)
        x = rearrange(x, "b h w (p1 p2 c) -> b (h p1) (w p2) c", p1=2, p2=2, c=C // 4)
        x = x.view(B, -1, C // 4)
        x = self.norm(x)
        return x


class FinalPatchExpand_X4(nn.Module):
    def __init__(self, input_resolution: Tuple[int, int], dim: int, dim_scale: int = 4, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.dim_scale = dim_scale

        self.expand = nn.Linear(dim, (dim_scale ** 2) * dim, bias=False)
        self.output_dim = dim
        self.norm = norm_layer(self.output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, H*W, C)
        """
        H, W = self.input_resolution
        x = self.expand(x)
        B, L, C = x.shape
        assert L == H * W

        x = x.view(B, H, W, C)
        x = rearrange(
            x,
            "b h w (p1 p2 c) -> b (h p1) (w p2) c",
            p1=self.dim_scale,
            p2=self.dim_scale,
            c=C // (self.dim_scale ** 2),
        )
        x = x.view(B, -1, self.output_dim)
        x = self.norm(x)
        return x


# =============================================================================
# Basic layers (encoder/decoder)
# =============================================================================

class BasicLayer(nn.Module):
    def __init__(
        self,
        dim: int,
        input_resolution: Tuple[int, int],
        depth: int,
        num_heads: int,
        window_size: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        qk_scale: Optional[float] = None,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float | List[float] = 0.0,
        norm_layer=nn.LayerNorm,
        downsample=None,
        use_checkpoint: bool = False,
    ):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        self.blocks = nn.ModuleList()
        for i in range(depth):
            blk = SwinTransformerBlock(
                dim=dim,
                input_resolution=input_resolution,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=0 if (i % 2 == 0) else window_size // 2,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer,
            )
            self.blocks.append(blk)

        self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer) if downsample is not None else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        if self.downsample is not None:
            x = self.downsample(x)
        return x


class BasicLayer_up(nn.Module):
    def __init__(
        self,
        dim: int,
        input_resolution: Tuple[int, int],
        depth: int,
        num_heads: int,
        window_size: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        qk_scale: Optional[float] = None,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float | List[float] = 0.0,
        norm_layer=nn.LayerNorm,
        upsample: Optional[Any] = None,
        use_checkpoint: bool = False,
    ):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        self.blocks = nn.ModuleList()
        for i in range(depth):
            blk = SwinTransformerBlock(
                dim=dim,
                input_resolution=input_resolution,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=0 if (i % 2 == 0) else window_size // 2,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer,
            )
            self.blocks.append(blk)

        self.upsample = PatchExpand(input_resolution, dim=dim, dim_scale=2, norm_layer=norm_layer) if upsample is not None else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        if self.upsample is not None:
            x = self.upsample(x)
        return x


# =============================================================================
# Patch Embedding
# =============================================================================

class PatchEmbed(nn.Module):
    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 4,
        in_chans: int = 3,
        embed_dim: int = 96,
        norm_layer: Optional[Any] = None,
    ):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)

        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer is not None else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        assert (H, W) == (self.img_size[0], self.img_size[1]), (
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        )
        x = self.proj(x).flatten(2).transpose(1, 2)  # (B, Ph*Pw, embed_dim)
        if self.norm is not None:
            x = self.norm(x)
        return x


# =============================================================================
# Swin-Unet (System)
# =============================================================================

@dataclass
class SwinUNetConfig:
    # fixed for best pretrained compatibility
    img_size: int = 224
    patch_size: int = 4
    window_size: int = 7

    in_chans: int = 4
    num_classes: int = 4

    embed_dim: int = 96
    depths: Tuple[int, int, int, int] = (2, 2, 6, 2)  # Swin-T
    num_heads: Tuple[int, int, int, int] = (3, 6, 12, 24)

    mlp_ratio: float = 4.0
    qkv_bias: bool = True
    drop_rate: float = 0.0
    attn_drop_rate: float = 0.0
    drop_path_rate: float = 0.1

    ape: bool = False
    patch_norm: bool = True
    use_checkpoint: bool = False

    final_upsample: str = "expand_first"


class SwinUNet(nn.Module):
    """
    Swin-Unet: encoder Swin + transformer decoder with skip connections.
    Pretrained init: encoder-only from timm Swin-T.
    """

    def __init__(self, cfg: SwinUNetConfig, use_imagenet_pretrained_encoder: bool = False,
                 timm_name: str = "swin_tiny_patch4_window7_224.ms_in1k"):
        super().__init__()
        self.cfg = cfg
        self.use_imagenet_pretrained_encoder = bool(use_imagenet_pretrained_encoder)
        self.timm_name = timm_name

        img_size = cfg.img_size
        patch_size = cfg.patch_size
        window_size = cfg.window_size

        self.num_classes = cfg.num_classes
        self.num_layers = len(cfg.depths)
        self.embed_dim = cfg.embed_dim
        self.ape = cfg.ape
        self.patch_norm = cfg.patch_norm
        self.num_features = int(cfg.embed_dim * 2 ** (self.num_layers - 1))
        self.mlp_ratio = cfg.mlp_ratio
        self.final_upsample = cfg.final_upsample

        # patch embed
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=cfg.in_chans,
            embed_dim=cfg.embed_dim,
            norm_layer=nn.LayerNorm if self.patch_norm else None,
        )
        num_patches = self.patch_embed.num_patches
        self.patches_resolution = self.patch_embed.patches_resolution

        # absolute pos embed (not used in Swin-T by default)
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, cfg.embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=0.02)
        else:
            self.absolute_pos_embed = None

        self.pos_drop = nn.Dropout(p=cfg.drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, cfg.drop_path_rate, sum(cfg.depths))]

        # encoder
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(
                dim=int(cfg.embed_dim * 2 ** i_layer),
                input_resolution=(
                    self.patches_resolution[0] // (2 ** i_layer),
                    self.patches_resolution[1] // (2 ** i_layer),
                ),
                depth=cfg.depths[i_layer],
                num_heads=cfg.num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=cfg.mlp_ratio,
                qkv_bias=cfg.qkv_bias,
                qk_scale=None,
                drop=cfg.drop_rate,
                attn_drop=cfg.attn_drop_rate,
                drop_path=dpr[sum(cfg.depths[:i_layer]):sum(cfg.depths[:i_layer + 1])],
                norm_layer=nn.LayerNorm,
                downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                use_checkpoint=cfg.use_checkpoint,
            )
            self.layers.append(layer)

        # decoder
        self.layers_up = nn.ModuleList()
        self.concat_back_dim = nn.ModuleList()

        for i_layer in range(self.num_layers):
            # concat then reduce dim (except first up stage)
            concat_linear = (
                nn.Linear(2 * int(cfg.embed_dim * 2 ** (self.num_layers - 1 - i_layer)),
                          int(cfg.embed_dim * 2 ** (self.num_layers - 1 - i_layer)))
                if i_layer > 0 else nn.Identity()
            )

            if i_layer == 0:
                layer_up = PatchExpand(
                    input_resolution=(
                        self.patches_resolution[0] // (2 ** (self.num_layers - 1 - i_layer)),
                        self.patches_resolution[1] // (2 ** (self.num_layers - 1 - i_layer)),
                    ),
                    dim=int(cfg.embed_dim * 2 ** (self.num_layers - 1 - i_layer)),
                    dim_scale=2,
                    norm_layer=nn.LayerNorm,
                )
            else:
                stage = (self.num_layers - 1 - i_layer)
                layer_up = BasicLayer_up(
                    dim=int(cfg.embed_dim * 2 ** stage),
                    input_resolution=(
                        self.patches_resolution[0] // (2 ** stage),
                        self.patches_resolution[1] // (2 ** stage),
                    ),
                    depth=cfg.depths[stage],
                    num_heads=cfg.num_heads[stage],
                    window_size=window_size,
                    mlp_ratio=cfg.mlp_ratio,
                    qkv_bias=cfg.qkv_bias,
                    qk_scale=None,
                    drop=cfg.drop_rate,
                    attn_drop=cfg.attn_drop_rate,
                    drop_path=dpr[sum(cfg.depths[:stage]):sum(cfg.depths[:stage + 1])],
                    norm_layer=nn.LayerNorm,
                    upsample=PatchExpand if (i_layer < self.num_layers - 1) else None,
                    use_checkpoint=cfg.use_checkpoint,
                )

            self.layers_up.append(layer_up)
            self.concat_back_dim.append(concat_linear)

        self.norm = nn.LayerNorm(self.num_features)
        self.norm_up = nn.LayerNorm(cfg.embed_dim)

        if self.final_upsample == "expand_first":
            # resolution at patch grid: img_size/patch_size = 56 for 224/4
            self.up = FinalPatchExpand_X4(
                input_resolution=(img_size // patch_size, img_size // patch_size),
                dim=cfg.embed_dim,
                dim_scale=4,
                norm_layer=nn.LayerNorm,
            )
            self.output = nn.Conv2d(cfg.embed_dim, cfg.num_classes, kernel_size=1, bias=False)
        else:
            raise ValueError(f"Unsupported final_upsample: {self.final_upsample}")

        self.apply(self._init_weights)

        # optional pretrained init for encoder only
        if self.use_imagenet_pretrained_encoder:
            self._load_pretrained_encoder_from_timm()

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.no_grad()
    def _load_pretrained_encoder_from_timm(self):
        """
        Load ONLY encoder-related weights from timm Swin.
        We do NOT freeze encoder; this only initializes weights.
        """
        try:
            import timm
        except Exception as e:
            raise RuntimeError("timm is required for pretrained init. Please pip install timm.") from e

        # create timm model with same in_chans so timm adapts first conv weight (3->4)
        timm_model = timm.create_model(
            self.timm_name,
            pretrained=True,
            in_chans=self.cfg.in_chans,
            num_classes=0,  # no head
        )

        src_sd = timm_model.state_dict()
        dst_sd = self.state_dict()

        # map rule:
        # - patch_embed.*  -> patch_embed.*
        # - layers.*       -> layers.*
        # - norm.* (final encoder norm) exists in our model as self.norm (but our norm is applied after last stage);
        #   timm has 'norm' too, we can load it.
        allowed_prefixes = (
            "patch_embed.",
            "layers.",
            "norm.",
            "absolute_pos_embed",  # only if ape True (we default False)
        )

        load_sd = {}
        for k, v in src_sd.items():
            if k.startswith(allowed_prefixes):
                if k in dst_sd and dst_sd[k].shape == v.shape:
                    load_sd[k] = v

        missing, unexpected = self.load_state_dict(load_sd, strict=False)

        _safe_print(f"[PRETRAIN-ENC] timm={self.timm_name} | loaded_tensors={len(load_sd)}")
        # keep these logs short but useful
        if len(unexpected) > 0:
            _safe_print(f"[PRETRAIN-ENC] unexpected_keys={len(unexpected)} (ignored)")
        if len(missing) > 0:
            # missing includes decoder/head keys + any mismatch keys
            _safe_print(f"[PRETRAIN-ENC] missing_keys={len(missing)} (expected for decoder/head)")

    # ------------------- forward -------------------
    def forward_features(self, x: torch.Tensor):
        x = self.patch_embed(x)  # (B, L, C)
        if self.absolute_pos_embed is not None:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        x_downsample = []
        for layer in self.layers:
            x_downsample.append(x)
            x = layer(x)

        x = self.norm(x)  # (B, L, C_last)
        return x, x_downsample

    def forward_up_features(self, x: torch.Tensor, x_downsample: List[torch.Tensor]):
        for i, layer_up in enumerate(self.layers_up):
            if i == 0:
                x = layer_up(x)
            else:
                # skip connection: from encoder stage (reverse order)
                x = torch.cat([x, x_downsample[(self.num_layers - 1) - i]], dim=-1)
                x = self.concat_back_dim[i](x)
                x = layer_up(x)
        x = self.norm_up(x)  # (B, L, embed_dim)
        return x

    def up_x4(self, x: torch.Tensor) -> torch.Tensor:
        H, W = self.patches_resolution  # 56,56
        B, L, C = x.shape
        assert L == H * W
        x = self.up(x)  # (B, (4H)*(4W), embed_dim) token space
        x = x.view(B, 4 * H, 4 * W, -1).permute(0, 3, 1, 2).contiguous()  # (B, C, 224, 224)
        x = self.output(x)  # (B, num_classes, 224, 224)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, x_down = self.forward_features(x)
        x = self.forward_up_features(x, x_down)
        x = self.up_x4(x)
        return x


# =============================================================================
# Convenience builder
# =============================================================================

def build_swin_unet_tiny_224(
    in_chans: int = 4,
    num_classes: int = 4,
    use_imagenet_pretrained_encoder: bool = False,
) -> SwinUNet:
    cfg = SwinUNetConfig(
        img_size=224,
        patch_size=4,
        window_size=7,
        in_chans=in_chans,
        num_classes=num_classes,
        embed_dim=96,
        depths=(2, 2, 6, 2),
        num_heads=(3, 6, 12, 24),
        drop_path_rate=0.1,
        ape=False,
        patch_norm=True,
        use_checkpoint=False,
        final_upsample="expand_first",
    )
    return SwinUNet(cfg, use_imagenet_pretrained_encoder=use_imagenet_pretrained_encoder)


# =============================================================================
# Main test
# =============================================================================

def _run_summary(model: nn.Module, device: torch.device):
    # torchsummary uses input_size without batch
    try:
        from torchsummary import summary
        _safe_print("\n[torchsummary]")
        summary(model, input_size=(4, 224, 224), device=str(device))
        return
    except Exception as e:
        _safe_print(f"[torchsummary] failed: {repr(e)}")

    # fallback torchinfo
    try:
        from torchinfo import summary as info_summary
        _safe_print("\n[torchinfo]")
        info_summary(model, input_size=(2, 4, 224, 224), device=str(device))
    except Exception as e:
        _safe_print(f"[torchinfo] failed: {repr(e)}")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _safe_print(f"Device: {device}")

    x = torch.randn(2, 4, 224, 224, device=device)

    # pretrained encoder
    _safe_print("\n=== SwinUNet (pretrained encoder only) ===")
    model_pt = build_swin_unet_tiny_224(in_chans=4, num_classes=4, use_imagenet_pretrained_encoder=True).to(device)
    with torch.no_grad():
        y = model_pt(x)
    _safe_print(f"Output: {tuple(y.shape)}")
    _safe_print(f"Params: {count_parameters(model_pt):,}")
    _run_summary(model_pt, device)

    # scratch
    _safe_print("\n=== SwinUNet (scratch) ===")
    model_sc = build_swin_unet_tiny_224(in_chans=4, num_classes=4, use_imagenet_pretrained_encoder=False).to(device)
    with torch.no_grad():
        y2 = model_sc(x)
    _safe_print(f"Output: {tuple(y2.shape)}")
    _safe_print(f"Params: {count_parameters(model_sc):,}")
    _run_summary(model_sc, device)


if __name__ == "__main__":
    main()
