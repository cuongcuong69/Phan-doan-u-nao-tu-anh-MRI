import torch
from torch import nn


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class ViTBlock(nn.Module):
    def __init__(self, dim: int, heads: int, mlp_dim: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        return x + self.mlp(self.norm2(x))


class TransUNet(nn.Module):
    def __init__(self, in_channels: int = 4, num_classes: int = 3, img_dim: int = 256):
        super().__init__()
        if img_dim % 8 != 0:
            raise ValueError("img_dim must be divisible by 8.")
        self.img_dim = img_dim

        # CNN encoder
        self.conv1 = ConvBlock(in_channels, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = ConvBlock(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.conv3 = ConvBlock(128, 256)
        self.pool3 = nn.MaxPool2d(2)

        # Transformer bottleneck
        self.patch_embed = nn.Conv2d(256, 512, kernel_size=1)
        grid = (img_dim // 8) * (img_dim // 8)
        self.pos_embed = nn.Parameter(torch.zeros(1, grid, 512))
        self.blocks = nn.ModuleList([ViTBlock(512, 8, 1024) for _ in range(6)])
        self.norm = nn.LayerNorm(512)

        # Decoder
        self.up1 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec1 = ConvBlock(512, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = ConvBlock(256, 128)
        self.up3 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec3 = ConvBlock(128, 64)
        self.out_conv = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.conv1(x)
        p1 = self.pool1(x1)
        x2 = self.conv2(p1)
        p2 = self.pool2(x2)
        x3 = self.conv3(p2)
        p3 = self.pool3(x3)

        emb = self.patch_embed(p3)
        b, c, h, w = emb.shape
        emb = emb.flatten(2).transpose(1, 2)
        if emb.shape[1] != self.pos_embed.shape[1]:
            raise ValueError("Input size does not match position embedding.")
        emb = emb + self.pos_embed
        for blk in self.blocks:
            emb = blk(emb)
        emb = self.norm(emb)
        emb = emb.transpose(1, 2).reshape(b, c, h, w)

        u1 = self.up1(emb)
        u1 = torch.cat([x3, u1], dim=1)
        d1 = self.dec1(u1)

        u2 = self.up2(d1)
        u2 = torch.cat([x2, u2], dim=1)
        d2 = self.dec2(u2)

        u3 = self.up3(d2)
        u3 = torch.cat([x1, u3], dim=1)
        d3 = self.dec3(u3)

        return self.out_conv(d3)
