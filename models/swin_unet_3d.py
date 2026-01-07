# -*- coding: utf-8 -*-
"""
models/swin_unet_3d.py

SwinUnet3D (paper-style) for BraTS-like 3D segmentation with FIXED patch size 128x128x128.

Design choices for patch=128^3:
- Downsampling factors: 4,2,2,2  => spatial: 128 -> 32 -> 16 -> 8 -> 4
- Window size must divide ALL token resolutions => choose window_size=(4,4,4)
  (works cleanly for 32/16/8/4, avoids any window-padding)
- Shifted window uses shift=(2,2,2) on odd blocks.
- Patch Embedding (DownStage12): Conv3d(k=4,s=4) + LayerNorm
- Next downsamples: PatchMerging3D: Conv3d(k=2,s=2) + LayerNorm
- Feature extraction per stage: Hybrid(Swin blocks + ConvBlock3D) then fuse by addition
- Upsample: PatchExpanding3D: ConvTranspose3d(k=s) + LayerNorm + PReLU
- Skip connections like UNet, fuse via concat + 1x1 conv + LN
- Final up x4 to return to 128 resolution, then 1x1 conv for logits.

Input : x [B, in_channels, 128, 128, 128]
Output: y [B, num_classes, 128, 128, 128]

Main:
- forward test for (1,4,128,128,128)
- torchinfo.summary shows parameter count + shapes

Requirements:
- torch
- torchinfo (pip install torchinfo)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# Helpers
# ============================================================

def _to_3tuple(x) -> Tuple[int, int, int]:
    if isinstance(x, (tuple, list)):
        assert len(x) == 3
        return int(x[0]), int(x[1]), int(x[2])
    return int(x), int(x), int(x)


def _channel_last_ln(x: torch.Tensor, ln: nn.LayerNorm) -> torch.Tensor:
    """Apply LN over channel C for x: [B,C,D,H,W] via channel-last."""
    b, c, d, h, w = x.shape
    x = x.permute(0, 2, 3, 4, 1).contiguous()  # [B,D,H,W,C]
    x = ln(x)
    x = x.permute(0, 4, 1, 2, 3).contiguous()  # [B,C,D,H,W]
    return x


# ============================================================
# MLP
# ============================================================

class Mlp(nn.Module):
    def __init__(self, dim: int, mlp_ratio: float = 4.0, drop: float = 0.0):
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


# ============================================================
# Window partition / reverse (3D)
# ============================================================

def window_partition_3d(x: torch.Tensor, window_size: Tuple[int, int, int]) -> torch.Tensor:
    """
    x: [B, D, H, W, C]
    return: [B*nW, wd*wh*ww, C]
    """
    wd, wh, ww = window_size
    b, d, h, w, c = x.shape
    assert d % wd == 0 and h % wh == 0 and w % ww == 0, \
        f"(D,H,W)=({d},{h},{w}) must be divisible by window={window_size}"

    x = x.view(b, d // wd, wd, h // wh, wh, w // ww, ww, c)
    x = x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous()  # [B, d//wd, h//wh, w//ww, wd,wh,ww,C]
    windows = x.view(-1, wd * wh * ww, c)
    return windows


def window_reverse_3d(windows: torch.Tensor, window_size: Tuple[int, int, int],
                      b: int, d: int, h: int, w: int) -> torch.Tensor:
    """
    windows: [B*nW, wd*wh*ww, C]
    return: [B, D, H, W, C]
    """
    wd, wh, ww = window_size
    c = windows.shape[-1]
    x = windows.view(b, d // wd, h // wh, w // ww, wd, wh, ww, c)
    x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous()
    x = x.view(b, d, h, w, c)
    return x


# ============================================================
# Window Attention 3D (with relative position bias)
# ============================================================

class WindowAttention3D(nn.Module):
    def __init__(
        self,
        dim: int,
        window_size: Tuple[int, int, int],
        num_heads: int,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        super().__init__()
        self.dim = dim
        self.window_size = _to_3tuple(window_size)
        self.num_heads = num_heads
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        wd, wh, ww = self.window_size
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * wd - 1) * (2 * wh - 1) * (2 * ww - 1), num_heads)
        )

        coords_d = torch.arange(wd)
        coords_h = torch.arange(wh)
        coords_w = torch.arange(ww)
        coords = torch.stack(torch.meshgrid(coords_d, coords_h, coords_w, indexing="ij"))  # [3,wd,wh,ww]
        coords_flatten = torch.flatten(coords, 1)  # [3,N]
        rel = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # [3,N,N]
        rel = rel.permute(1, 2, 0).contiguous()  # [N,N,3]
        rel[:, :, 0] += wd - 1
        rel[:, :, 1] += wh - 1
        rel[:, :, 2] += ww - 1
        rel[:, :, 0] *= (2 * wh - 1) * (2 * ww - 1)
        rel[:, :, 1] *= (2 * ww - 1)
        relative_position_index = rel.sum(-1)  # [N,N]
        self.register_buffer("relative_position_index", relative_position_index)

        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        x: [B*nW, N, C]
        mask: [nW, N, N] or None
        """
        b_, n, c = x.shape
        qkv = self.qkv(x).reshape(b_, n, 3, self.num_heads, c // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4).contiguous()
        q, k, v = qkv[0], qkv[1], qkv[2]  # [b_,heads,N,head_dim]

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))  # [b_,heads,N,N]

        rpb = self.relative_position_bias_table[self.relative_position_index.view(-1)]
        rpb = rpb.view(n, n, -1).permute(2, 0, 1).contiguous()  # [heads,N,N]
        attn = attn + rpb.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(b_ // nW, nW, self.num_heads, n, n)
            attn = attn + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, n, n)

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = (attn @ v).transpose(1, 2).reshape(b_, n, c)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out


# ============================================================
# Swin Transformer Block 3D
# ============================================================

class SwinTransformerBlock3D(nn.Module):
    """
    LN -> (W-MSA or SW-MSA) -> residual -> LN -> MLP -> residual
    Input/Output in channel-last tokens: [B, D, H, W, C]
    """
    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: Tuple[int, int, int],
        shift_size: Tuple[int, int, int],
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
    ):
        super().__init__()
        self.dim = dim
        self.window_size = _to_3tuple(window_size)
        self.shift_size = _to_3tuple(shift_size)

        wd, wh, ww = self.window_size
        sd, sh, sw = self.shift_size
        assert 0 <= sd < wd and 0 <= sh < wh and 0 <= sw < ww, "invalid shift_size"

        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention3D(
            dim=dim,
            window_size=self.window_size,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(dim=dim, mlp_ratio=mlp_ratio, drop=drop)

        self.register_buffer("attn_mask", None, persistent=False)

    def _build_mask(self, d: int, h: int, w: int, device) -> Optional[torch.Tensor]:
        sd, sh, sw = self.shift_size
        if sd == 0 and sh == 0 and sw == 0:
            return None

        wd, wh, ww = self.window_size
        img_mask = torch.zeros((1, d, h, w, 1), device=device)
        cnt = 0

        d_slices = (slice(0, -wd), slice(-wd, -sd), slice(-sd, None))
        h_slices = (slice(0, -wh), slice(-wh, -sh), slice(-sh, None))
        w_slices = (slice(0, -ww), slice(-ww, -sw), slice(-sw, None))

        for ds in d_slices:
            for hs in h_slices:
                for ws in w_slices:
                    img_mask[:, ds, hs, ws, :] = cnt
                    cnt += 1

        mask_windows = window_partition_3d(img_mask, self.window_size)  # [nW, N, 1]
        mask_windows = mask_windows.view(-1, wd * wh * ww)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float("-inf")).masked_fill(attn_mask == 0, 0.0)
        return attn_mask

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, D, H, W, C]
        """
        b, d, h, w, c = x.shape
        shortcut = x

        x = self.norm1(x)

        sd, sh, sw = self.shift_size
        if sd or sh or sw:
            x = torch.roll(x, shifts=(-sd, -sh, -sw), dims=(1, 2, 3))
            if self.attn_mask is None or self.attn_mask.device != x.device:
                self.attn_mask = self._build_mask(d, h, w, x.device)

        x_windows = window_partition_3d(x, self.window_size)      # [B*nW, N, C]
        attn_windows = self.attn(x_windows, mask=self.attn_mask)  # [B*nW, N, C]
        x = window_reverse_3d(attn_windows, self.window_size, b, d, h, w)

        if sd or sh or sw:
            x = torch.roll(x, shifts=(sd, sh, sw), dims=(1, 2, 3))

        x = shortcut + x
        x = x + self.mlp(self.norm2(x))
        return x


# ============================================================
# Conv Block3D (local dependency)
# ============================================================

class DepthwiseSeparableConv3d(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 3, padding: int = 1):
        super().__init__()
        self.dw = nn.Conv3d(channels, channels, kernel_size=kernel_size, padding=padding, groups=channels, bias=False)
        self.pw = nn.Conv3d(channels, channels, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pw(self.dw(x))


class ConvBlock3D(nn.Module):
    """
    Conv Block3D (paper-style):
      Y = PReLU(LN(Conv(I)))
      Y = PReLU(LN(Conv(Y)))
      O = Y * I
    """
    def __init__(self, dim: int):
        super().__init__()
        self.conv1 = DepthwiseSeparableConv3d(dim, 3, 1)
        self.ln1 = nn.LayerNorm(dim)
        self.act1 = nn.PReLU()

        self.conv2 = DepthwiseSeparableConv3d(dim, 3, 1)
        self.ln2 = nn.LayerNorm(dim)
        self.act2 = nn.PReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv1(x)
        y = _channel_last_ln(y, self.ln1)
        y = self.act1(y)

        y = self.conv2(y)
        y = _channel_last_ln(y, self.ln2)
        y = self.act2(y)

        return y * x


# ============================================================
# Patch Embedding / Merging / Expanding
# ============================================================

class PatchEmbed3D(nn.Module):
    """DownStage12: Conv3d(k=4,s=4) + LN"""
    def __init__(self, in_chans: int, embed_dim: int):
        super().__init__()
        self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=4, stride=4)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)                 # [B,C,32,32,32] when input 128^3
        x = _channel_last_ln(x, self.norm)
        return x


class PatchMerging3D(nn.Module):
    """Downsample by 2: Conv3d(k=2,s=2) + LN"""
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.reduction = nn.Conv3d(in_dim, out_dim, kernel_size=2, stride=2)
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.reduction(x)
        x = _channel_last_ln(x, self.norm)
        return x


class PatchExpanding3D(nn.Module):
    """Upsample by factor k: ConvTranspose3d(k,s=k) + LN + PReLU"""
    def __init__(self, in_dim: int, out_dim: int, up_factor: int):
        super().__init__()
        k = int(up_factor)
        self.up = nn.ConvTranspose3d(in_dim, out_dim, kernel_size=k, stride=k, padding=0)
        self.norm = nn.LayerNorm(out_dim)
        self.act = nn.PReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        x = _channel_last_ln(x, self.norm)
        x = self.act(x)
        return x


# ============================================================
# Hybrid block: Swin blocks (long-range) + Conv block (local)
# ============================================================

class HybridSwinConv3D(nn.Module):
    """
    Parallel branches:
      - Swin blocks over tokens
      - ConvBlock3D over feature map
    Fuse by addition.
    """
    def __init__(
        self,
        dim: int,
        depth: int,
        num_heads: int,
        window_size: Tuple[int, int, int],
        mlp_ratio: float = 4.0,
        drop: float = 0.0,
        attn_drop: float = 0.0,
    ):
        super().__init__()
        ws = _to_3tuple(window_size)
        shift = tuple(w // 2 for w in ws)

        self.swin_blocks = nn.ModuleList([
            SwinTransformerBlock3D(
                dim=dim,
                num_heads=num_heads,
                window_size=ws,
                shift_size=(0, 0, 0) if (i % 2 == 0) else shift,
                mlp_ratio=mlp_ratio,
                drop=drop,
                attn_drop=attn_drop,
            )
            for i in range(depth)
        ])
        self.conv = ConvBlock3D(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Swin branch
        xt = x.permute(0, 2, 3, 4, 1).contiguous()  # [B,D,H,W,C]
        for blk in self.swin_blocks:
            xt = blk(xt)
        xswin = xt.permute(0, 4, 1, 2, 3).contiguous()

        # Conv branch
        xconv = self.conv(x)

        return xswin + xconv


# ============================================================
# UpStage with skip fusion
# ============================================================

class UpStage(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        depth: int,
        num_heads: int,
        window_size: Tuple[int, int, int],
        up_factor: int = 2,
        mlp_ratio: float = 4.0,
    ):
        super().__init__()
        self.up = PatchExpanding3D(in_dim, out_dim, up_factor=up_factor)
        self.fuse = nn.Conv3d(out_dim * 2, out_dim, kernel_size=1, bias=False)
        self.fuse_ln = nn.LayerNorm(out_dim)

        self.hybrid = HybridSwinConv3D(
            dim=out_dim,
            depth=depth,
            num_heads=num_heads,
            window_size=window_size,
            mlp_ratio=mlp_ratio,
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        # shapes must match exactly (for patch=128, window=4, no padding needed)
        if x.shape[2:] != skip.shape[2:]:
            raise RuntimeError(f"Skip shape mismatch: up={x.shape} vs skip={skip.shape}")

        x = torch.cat([x, skip], dim=1)
        x = self.fuse(x)
        x = _channel_last_ln(x, self.fuse_ln)

        x = self.hybrid(x)
        return x


# ============================================================
# Config + Model
# ============================================================

@dataclass
class SwinUnet3DConfig:
    in_channels: int = 4
    num_classes: int = 4

    embed_dim: int = 96
    depths: Tuple[int, int, int, int] = (2, 2, 2, 2)     # for stages at 32/16/8/4
    num_heads: Tuple[int, int, int, int] = (3, 6, 12, 24)

    window_size: Tuple[int, int, int] = (4, 4, 4)        # IMPORTANT for patch=128
    mlp_ratio: float = 4.0


class SwinUnet3D(nn.Module):
    """
    For patch 128x128x128:
      enc1: 128 -> 32  (embed_dim)
      enc2: 32  -> 16  (x2)
      enc3: 16  -> 8   (x2)
      enc4: 8   -> 4   (x2)
      dec3: 4   -> 8
      dec2: 8   -> 16
      dec1: 16  -> 32
      final_up: 32 -> 128 (x4)
    """
    def __init__(self, cfg: SwinUnet3DConfig):
        super().__init__()
        self.cfg = cfg
        ws = _to_3tuple(cfg.window_size)

        c1 = cfg.embed_dim
        c2 = c1 * 2
        c3 = c2 * 2
        c4 = c3 * 2

        # Encoder
        self.patch_embed = PatchEmbed3D(cfg.in_channels, c1)
        self.enc1 = HybridSwinConv3D(c1, cfg.depths[0], cfg.num_heads[0], ws, cfg.mlp_ratio)

        self.down2 = PatchMerging3D(c1, c2)
        self.enc2 = HybridSwinConv3D(c2, cfg.depths[1], cfg.num_heads[1], ws, cfg.mlp_ratio)

        self.down3 = PatchMerging3D(c2, c3)
        self.enc3 = HybridSwinConv3D(c3, cfg.depths[2], cfg.num_heads[2], ws, cfg.mlp_ratio)

        self.down4 = PatchMerging3D(c3, c4)
        self.enc4 = HybridSwinConv3D(c4, cfg.depths[3], cfg.num_heads[3], ws, cfg.mlp_ratio)

        # Decoder
        self.dec3 = UpStage(c4, c3, cfg.depths[2], cfg.num_heads[2], ws, up_factor=2, mlp_ratio=cfg.mlp_ratio)
        self.dec2 = UpStage(c3, c2, cfg.depths[1], cfg.num_heads[1], ws, up_factor=2, mlp_ratio=cfg.mlp_ratio)
        self.dec1 = UpStage(c2, c1, cfg.depths[0], cfg.num_heads[0], ws, up_factor=2, mlp_ratio=cfg.mlp_ratio)

        # Final up (x4) + head
        self.final_up = PatchExpanding3D(c1, c1, up_factor=4)
        self.head = nn.Conv3d(c1, cfg.num_classes, kernel_size=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, C, 128, 128, 128]
        """
        if x.shape[2:] != (128, 128, 128):
            raise ValueError(f"This implementation is configured for patch=128^3, got {x.shape[2:]}")

        # Encoder
        x1 = self.patch_embed(x)   # [B, c1, 32,32,32]
        x1 = self.enc1(x1)

        x2 = self.down2(x1)        # [B, c2, 16,16,16]
        x2 = self.enc2(x2)

        x3 = self.down3(x2)        # [B, c3,  8, 8, 8]
        x3 = self.enc3(x3)

        x4 = self.down4(x3)        # [B, c4,  4, 4, 4]
        x4 = self.enc4(x4)

        # Decoder + skips
        y3 = self.dec3(x4, x3)     # [B, c3,  8, 8, 8]
        y2 = self.dec2(y3, x2)     # [B, c2, 16,16,16]
        y1 = self.dec1(y2, x1)     # [B, c1, 32,32,32]

        # Back to 128
        y = self.final_up(y1)      # [B, c1, 128,128,128]
        y = self.head(y)           # [B, num_classes, 128,128,128]
        return y


# ============================================================
# Main test + torchinfo
# ============================================================

def main():
    from torchinfo import summary

    cfg = SwinUnet3DConfig(
        in_channels=4,
        num_classes=4,
        embed_dim=96,
        depths=(2, 2, 2, 2),
        num_heads=(3, 6, 12, 24),
        window_size=(4, 4, 4),
        mlp_ratio=4.0,
    )
    model = SwinUnet3D(cfg)

    x = torch.randn(1, cfg.in_channels, 128, 128, 128)
    with torch.no_grad():
        y = model(x)
    print(f"[OK] x={tuple(x.shape)} -> y={tuple(y.shape)}")

    # torchinfo: params + shapes
    summary(model, input_size=tuple(x.shape), depth=5, verbose=1)
    # Tính toán số lượng tham số
    total_params = sum(p.numel() for p in model.parameters())
    print(f'Total number of parameters: {total_params}')


if __name__ == "__main__":
    main()
