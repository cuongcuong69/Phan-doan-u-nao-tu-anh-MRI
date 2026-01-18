import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List

class PatchEmbedding3D(nn.Module):
    def __init__(self, img_size=(128, 128, 128), patch_size=16, in_channels=4, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = tuple(s // patch_size for s in img_size)
        self.num_patches = self.grid_size[0] * self.grid_size[1] * self.grid_size[2]
        self.proj = nn.Conv3d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        
    def forward(self, x):
        x = self.proj(x)  # (B, embed_dim, D/16, H/16, W/16)
        x = x.flatten(2).transpose(1, 2)  # (B, num_patches, embed_dim)
        return x

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim=768, num_heads=12, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, int(embed_dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(embed_dim * mlp_ratio), embed_dim),
            nn.Dropout(dropout),
        )
        
    def forward(self, x):
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + self.mlp(self.norm2(x))
        return x

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, skip_channels):
        super().__init__()
        self.upsample = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = nn.Sequential(
            nn.Conv3d(out_channels + skip_channels, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm3d(out_channels),
            nn.ReLU(inplace=True),
        )
        
    def forward(self, x, skip):
        x = self.upsample(x)
        x = torch.cat([x, skip], dim=1)
        x = self.conv(x)
        return x

class UNETR(nn.Module):
    def __init__(
        self,
        n_channels: int = 4,
        n_classes: int = 4,
        img_size: Tuple[int, int, int] = (128, 128, 128),
        patch_size: int = 16,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.grid_size = tuple(s // patch_size for s in img_size)

        # 1. Embedding & Transformer
        self.patch_embed = PatchEmbedding3D(img_size, patch_size, n_channels, embed_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.patch_embed.num_patches, embed_dim))
        self.pos_drop = nn.Dropout(dropout)
        
        self.transformer = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])
        self.transformer_norm = nn.LayerNorm(embed_dim)

        # 2. Projection Layers (Sửa lỗi chính ở đây)
        # Nhánh skip_0: Xử lý ảnh gốc (128^3)
        self.encoder0 = nn.Sequential(
            nn.Conv3d(n_channels, 32, kernel_size=3, padding=1),
            nn.InstanceNorm3d(32),
            nn.ReLU(inplace=True),
            nn.Conv3d(32, 64, kernel_size=3, padding=1),
            nn.InstanceNorm3d(64),
            nn.ReLU(inplace=True),
        )

        # Nhánh skip_3: Đưa từ Transformer (8^3) -> (64^3) thông qua 3 lần upsample
        self.proj_3 = nn.Sequential(
            nn.ConvTranspose3d(embed_dim, 256, kernel_size=2, stride=2),
            nn.ConvTranspose3d(256, 128, kernel_size=2, stride=2),
            nn.ConvTranspose3d(128, 128, kernel_size=2, stride=2),
        )

        # Nhánh skip_6: Đưa từ Transformer (8^3) -> (32^3) thông qua 2 lần upsample
        self.proj_6 = nn.Sequential(
            nn.ConvTranspose3d(embed_dim, 256, kernel_size=2, stride=2),
            nn.ConvTranspose3d(256, 256, kernel_size=2, stride=2),
        )

        # Nhánh skip_9: Đưa từ Transformer (8^3) -> (16^3) thông qua 1 lần upsample
        self.proj_9 = nn.ConvTranspose3d(embed_dim, 512, kernel_size=2, stride=2)

        # Nhánh skip_12 (Bottleneck): Giữ nguyên (8^3)
        self.proj_12 = nn.Conv3d(embed_dim, 512, kernel_size=1)

        # 3. Decoder Path
        # decoder4: 8^3 -> 16^3 (cat với skip_9)
        self.decoder4 = DecoderBlock(in_channels=512, out_channels=256, skip_channels=512)
        # decoder3: 16^3 -> 32^3 (cat với skip_6)
        self.decoder3 = DecoderBlock(in_channels=256, out_channels=128, skip_channels=256)
        # decoder2: 32^3 -> 64^3 (cat với skip_3)
        self.decoder2 = DecoderBlock(in_channels=128, out_channels=64, skip_channels=128)
        # decoder1: 64^3 -> 128^3 (cat với skip_0 từ ảnh gốc)
        self.decoder1 = DecoderBlock(in_channels=64, out_channels=32, skip_channels=64)

        # 4. Final Output
        self.out_conv = nn.Conv3d(32, n_classes, kernel_size=1)

        # Init
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def reshape_to_3d(self, x):
        B = x.shape[0]
        D, H, W = self.grid_size
        x = x.transpose(1, 2).reshape(B, self.embed_dim, D, H, W)
        return x

    def forward(self, x):
        # Input: (B, 4, 128, 128, 128)
        
        # Nhánh encoder 0 (giữ resolution gốc)
        skip_0 = self.encoder0(x) # (B, 64, 128, 128, 128)

        # Transformer Encoder
        emb = self.patch_embed(x)
        emb = self.pos_drop(emb + self.pos_embed)
        
        hidden_states = []
        for i, block in enumerate(self.transformer):
            emb = block(emb)
            if (i + 1) in [3, 6, 9, 12]:
                hidden_states.append(self.transformer_norm(emb))

        # Trích xuất skip connections từ Transformer
        z3 = self.reshape_to_3d(hidden_states[0])  # (B, 768, 8, 8, 8)
        z6 = self.reshape_to_3d(hidden_states[1])  # (B, 768, 8, 8, 8)
        z9 = self.reshape_to_3d(hidden_states[2])  # (B, 768, 8, 8, 8)
        z12 = self.reshape_to_3d(hidden_states[3]) # (B, 768, 8, 8, 8)

        # Chiếu (Project) về các resolution khác nhau
        skip_12 = self.proj_12(z12) # (B, 512, 8, 8, 8)
        skip_9  = self.proj_9(z9)   # (B, 512, 16, 16, 16)
        skip_6  = self.proj_6(z6)   # (B, 256, 32, 32, 32)
        skip_3  = self.proj_3(z3)   # (B, 128, 64, 64, 64)

        # Decoder path
        d4 = self.decoder4(skip_12, skip_9) # 8->16, cat với 16 -> (B, 256, 16, 16, 16)
        d3 = self.decoder3(d4, skip_6)      # 16->32, cat với 32 -> (B, 128, 32, 32, 32)
        d2 = self.decoder2(d3, skip_3)      # 32->64, cat với 64 -> (B, 64, 64, 64, 64)
        d1 = self.decoder1(d2, skip_0)      # 64->128, cat với 128 -> (B, 32, 128, 128, 128)

        # Output
        logits = self.out_conv(d1) # (B, n_classes, 128, 128, 128)
        return logits

def build_unetr(n_channels=4, n_classes=4, img_size=(128, 128, 128)):
    return UNETR(n_channels=n_channels, n_classes=n_classes, img_size=img_size)

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = build_unetr().to(device)
    
    # Test với input BraTS chuẩn
    dummy_input = torch.randn(1, 4, 128, 128, 128).to(device)
    output = model(dummy_input)
    
    print(f"Input Shape: {dummy_input.shape}")
    print(f"Output Shape: {output.shape}")
    
    # Kiểm tra tính đúng đắn của kích thước
    assert output.shape == (1, 4, 128, 128, 128), "Lỗi kích thước đầu ra!"
    print("Mô hình hoạt động chính xác!")