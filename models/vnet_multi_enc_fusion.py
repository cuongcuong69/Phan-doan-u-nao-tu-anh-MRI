# -*- coding: utf-8 -*-
"""
VNet Multi-Encoder Fusion cho BraTS (4 modality riêng biệt).

Ý tưởng:
- Thay vì stack 4 modality vào 1 encoder chung,
  ta dùng 4 encoder con (cùng kiến trúc VNet-encoder, KHÔNG share weight),
  mỗi encoder xử lý 1 modality: FLAIR, T1, T1ce, T2 (1 kênh).
- Ở mỗi scale (x1..x5), concat feature của 4 encoder theo kênh,
  sau đó dùng 1x1 Conv3d (có thể kèm ReLU) để giảm kênh về đúng số channel như VNet gốc:
    x1: 4 * n_filters    -> n_filters
    x2: 4 * 2*n_filters  -> 2*n_filters
    x3: 4 * 4*n_filters  -> 4*n_filters
    x4: 4 * 8*n_filters  -> 8*n_filters
    x5: 4 * 16*n_filters -> 16*n_filters
- Decoder giữ nguyên cấu trúc VNet truyền thống, dùng các feature fused này để giải mã.

Ưu điểm:
- Mỗi modality có encoder riêng, học được đặc trưng chuyên biệt,
  sau đó mới học cách fusion ở mức sâu.

Chú ý:
- Nặng hơn ~4x so với VNet thường (vì 4 encoder), tốn VRAM hơn.
"""

import torch
from torch import nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Các block cơ bản (giữ nguyên như VNet)
# ---------------------------------------------------------------------------

class ConvBlock(nn.Module):
    def __init__(self, n_stages, n_filters_in, n_filters_out, normalization='none'):
        super(ConvBlock, self).__init__()

        ops = []
        for i in range(n_stages):
            if i == 0:
                input_channel = n_filters_in
            else:
                input_channel = n_filters_out

            ops.append(nn.Conv3d(input_channel, n_filters_out, 3, padding=1))
            if normalization == 'batchnorm':
                ops.append(nn.BatchNorm3d(n_filters_out))
            elif normalization == 'groupnorm':
                ops.append(nn.GroupNorm(num_groups=16, num_channels=n_filters_out))
            elif normalization == 'instancenorm':
                ops.append(nn.InstanceNorm3d(n_filters_out))
            elif normalization != 'none':
                raise ValueError(f"Unknown normalization: {normalization}")
            ops.append(nn.ReLU(inplace=True))

        self.conv = nn.Sequential(*ops)

    def forward(self, x):
        return self.conv(x)


class ResidualConvBlock(nn.Module):
    def __init__(self, n_stages, n_filters_in, n_filters_out, normalization='none'):
        super(ResidualConvBlock, self).__init__()

        ops = []
        for i in range(n_stages):
            if i == 0:
                input_channel = n_filters_in
            else:
                input_channel = n_filters_out

            ops.append(nn.Conv3d(input_channel, n_filters_out, 3, padding=1))
            if normalization == 'batchnorm':
                ops.append(nn.BatchNorm3d(n_filters_out))
            elif normalization == 'groupnorm':
                ops.append(nn.GroupNorm(num_groups=16, num_channels=n_filters_out))
            elif normalization == 'instancenorm':
                ops.append(nn.InstanceNorm3d(n_filters_out))
            elif normalization != 'none':
                raise ValueError(f"Unknown normalization: {normalization}")

            if i != n_stages - 1:
                ops.append(nn.ReLU(inplace=True))

        self.conv = nn.Sequential(*ops)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = (self.conv(x) + x)
        x = self.relu(x)
        return x


class DownsamplingConvBlock(nn.Module):
    def __init__(self, n_filters_in, n_filters_out, stride=2, normalization='none'):
        super(DownsamplingConvBlock, self).__init__()

        ops = []
        if normalization != 'none':
            ops.append(nn.Conv3d(n_filters_in, n_filters_out, stride, padding=0, stride=stride))
            if normalization == 'batchnorm':
                ops.append(nn.BatchNorm3d(n_filters_out))
            elif normalization == 'groupnorm':
                ops.append(nn.GroupNorm(num_groups=16, num_channels=n_filters_out))
            elif normalization == 'instancenorm':
                ops.append(nn.InstanceNorm3d(n_filters_out))
            else:
                raise ValueError(f"Unknown normalization: {normalization}")
        else:
            ops.append(nn.Conv3d(n_filters_in, n_filters_out, stride, padding=0, stride=stride))

        ops.append(nn.ReLU(inplace=True))

        self.conv = nn.Sequential(*ops)

    def forward(self, x):
        return self.conv(x)


class UpsamplingDeconvBlock(nn.Module):
    def __init__(self, n_filters_in, n_filters_out, stride=2, normalization='none'):
        super(UpsamplingDeconvBlock, self).__init__()

        ops = []
        if normalization != 'none':
            ops.append(nn.ConvTranspose3d(n_filters_in, n_filters_out, stride, padding=0, stride=stride))
            if normalization == 'batchnorm':
                ops.append(nn.BatchNorm3d(n_filters_out))
            elif normalization == 'groupnorm':
                ops.append(nn.GroupNorm(num_groups=16, num_channels=n_filters_out))
            elif normalization == 'instancenorm':
                ops.append(nn.InstanceNorm3d(n_filters_out))
            else:
                raise ValueError(f"Unknown normalization: {normalization}")
        else:
            ops.append(nn.ConvTranspose3d(n_filters_in, n_filters_out, stride, padding=0, stride=stride))

        ops.append(nn.ReLU(inplace=True))

        self.conv = nn.Sequential(*ops)

    def forward(self, x):
        return self.conv(x)


class Upsampling(nn.Module):
    def __init__(self, n_filters_in, n_filters_out, stride=2, normalization='none'):
        super(Upsampling, self).__init__()

        ops = []
        ops.append(nn.Upsample(scale_factor=stride, mode='trilinear', align_corners=False))
        ops.append(nn.Conv3d(n_filters_in, n_filters_out, kernel_size=3, padding=1))
        if normalization == 'batchnorm':
            ops.append(nn.BatchNorm3d(n_filters_out))
        elif normalization == 'groupnorm':
            ops.append(nn.GroupNorm(num_groups=16, num_channels=n_filters_out))
        elif normalization == 'instancenorm':
            ops.append(nn.InstanceNorm3d(n_filters_out))
        elif normalization != 'none':
            raise ValueError(f"Unknown normalization: {normalization}")
        ops.append(nn.ReLU(inplace=True))

        self.conv = nn.Sequential(*ops)

    def forward(self, x):
        return self.conv(x)


# ---------------------------------------------------------------------------
# Encoder cho 1 modality (n_channels=1)
# ---------------------------------------------------------------------------

class SingleModalityEncoder(nn.Module):
    """
    Encoder kiểu VNet chỉ cho 1 kênh (1 modality).
    Trả về list [x1, x2, x3, x4, x5] giống VNet.encoder.
    """

    def __init__(self, n_filters=16, normalization='none', has_dropout=False):
        super(SingleModalityEncoder, self).__init__()
        self.has_dropout = has_dropout

        # giống VNet nhưng n_channels cố định = 1
        self.block_one = ConvBlock(1, 1, n_filters, normalization=normalization)
        self.block_one_dw = DownsamplingConvBlock(n_filters, 2 * n_filters, normalization=normalization)

        self.block_two = ConvBlock(2, n_filters * 2, n_filters * 2, normalization=normalization)
        self.block_two_dw = DownsamplingConvBlock(n_filters * 2, n_filters * 4, normalization=normalization)

        self.block_three = ConvBlock(3, n_filters * 4, n_filters * 4, normalization=normalization)
        self.block_three_dw = DownsamplingConvBlock(n_filters * 4, n_filters * 8, normalization=normalization)

        self.block_four = ConvBlock(3, n_filters * 8, n_filters * 8, normalization=normalization)
        self.block_four_dw = DownsamplingConvBlock(n_filters * 8, n_filters * 16, normalization=normalization)

        self.block_five = ConvBlock(3, n_filters * 16, n_filters * 16, normalization=normalization)

        self.dropout = nn.Dropout3d(p=0.5, inplace=False)

    def forward(self, x):
        # x: (B,1,D,H,W)
        x1 = self.block_one(x)          # (B, f,   D,  H,  W)
        x1_dw = self.block_one_dw(x1)   # (B, 2f, D/2,H/2,W/2)

        x2 = self.block_two(x1_dw)      # (B, 2f, D/2,H/2,W/2)
        x2_dw = self.block_two_dw(x2)   # (B, 4f, D/4,H/4,W/4)

        x3 = self.block_three(x2_dw)    # (B, 4f, D/4,H/4,W/4)
        x3_dw = self.block_three_dw(x3) # (B, 8f, D/8,H/8,W/8)

        x4 = self.block_four(x3_dw)     # (B, 8f, D/8,H/8,W/8)
        x4_dw = self.block_four_dw(x4)  # (B,16f, D/16,H/16,W/16)

        x5 = self.block_five(x4_dw)     # (B,16f, D/16,H/16,W/16)

        if self.has_dropout:
            x5 = self.dropout(x5)

        return [x1, x2, x3, x4, x5]


# ---------------------------------------------------------------------------
# VNet Multi-Encoder Fusion
# ---------------------------------------------------------------------------

class VNetMultiEncFusion(nn.Module):
    """
    VNet với encoder đa nhánh (mỗi modality 1 encoder riêng),
    sau đó fusion feature bằng concat + 1x1 conv ở mỗi mức x1..x5.
    Decoder giữ nguyên kiểu VNet gốc.

    Tham số chính:
    - n_modalities: số modality đầu vào (mặc định 4 cho BraTS).
    - n_classes: số lớp output (4 cho BraTS multi-class).
    """

    def __init__(
        self,
        n_modalities: int = 4,
        n_classes: int = 4,
        n_filters: int = 16,
        normalization: str = 'none',
        has_dropout: bool = False,
    ):
        super(VNetMultiEncFusion, self).__init__()

        self.n_modalities = n_modalities
        self.n_classes = n_classes
        self.n_filters = n_filters
        self.normalization = normalization
        self.has_dropout = has_dropout

        # 4 encoder con (không share weight)
        self.encoders = nn.ModuleList([
            SingleModalityEncoder(
                n_filters=n_filters,
                normalization=normalization,
                has_dropout=has_dropout,
            )
            for _ in range(n_modalities)
        ])

        # FUSION LAYERS (1x1 conv) cho mỗi mức
        # x1: 4 * f   -> f
        # x2: 4 * 2f  -> 2f
        # x3: 4 * 4f  -> 4f
        # x4: 4 * 8f  -> 8f
        # x5: 4 * 16f -> 16f
        self.fuse_x1 = nn.Sequential(
            nn.Conv3d(n_modalities * n_filters, n_filters, kernel_size=1),
            nn.ReLU(inplace=True),
        )
        self.fuse_x2 = nn.Sequential(
            nn.Conv3d(n_modalities * 2 * n_filters, 2 * n_filters, kernel_size=1),
            nn.ReLU(inplace=True),
        )
        self.fuse_x3 = nn.Sequential(
            nn.Conv3d(n_modalities * 4 * n_filters, 4 * n_filters, kernel_size=1),
            nn.ReLU(inplace=True),
        )
        self.fuse_x4 = nn.Sequential(
            nn.Conv3d(n_modalities * 8 * n_filters, 8 * n_filters, kernel_size=1),
            nn.ReLU(inplace=True),
        )
        self.fuse_x5 = nn.Sequential(
            nn.Conv3d(n_modalities * 16 * n_filters, 16 * n_filters, kernel_size=1),
            nn.ReLU(inplace=True),
        )

        # Decoder VNet chuẩn (giống bản gốc) nhưng sử dụng feature đã fuse
        self.block_five_up = UpsamplingDeconvBlock(16 * n_filters, 8 * n_filters, normalization=normalization)

        self.block_six = ConvBlock(3, 8 * n_filters, 8 * n_filters, normalization=normalization)
        self.block_six_up = UpsamplingDeconvBlock(8 * n_filters, 4 * n_filters, normalization=normalization)

        self.block_seven = ConvBlock(3, 4 * n_filters, 4 * n_filters, normalization=normalization)
        self.block_seven_up = UpsamplingDeconvBlock(4 * n_filters, 2 * n_filters, normalization=normalization)

        self.block_eight = ConvBlock(2, 2 * n_filters, 2 * n_filters, normalization=normalization)
        self.block_eight_up = UpsamplingDeconvBlock(2 * n_filters, n_filters, normalization=normalization)

        self.block_nine = ConvBlock(1, n_filters, n_filters, normalization=normalization)
        self.out_conv = nn.Conv3d(n_filters, n_classes, 1, padding=0)

        self.dropout = nn.Dropout3d(p=0.5, inplace=False)

    # -------------------- Encoder multi-branch --------------------

    def encoder(self, x):
        """
        x: (B, n_modalities, D, H, W)
        Trả về list feature fused [x1_f, x2_f, x3_f, x4_f, x5_f]
        với số kênh tương ứng: f, 2f, 4f, 8f, 16f.
        """
        B, C, D, H, W = x.shape
        assert C == self.n_modalities, f"Expected {self.n_modalities} modalities, got {C}."

        # features_per_modality[m] là list [x1_m, x2_m, x3_m, x4_m, x5_m]
        features_per_modality = []

        for m in range(self.n_modalities):
            xm = x[:, m:m+1, ...]  # (B,1,D,H,W)
            feats_m = self.encoders[m](xm)  # list 5 level
            features_per_modality.append(feats_m)

        # Fuse từng level qua concat + 1x1 conv
        # Level 1
        x1_list = [features_per_modality[m][0] for m in range(self.n_modalities)]
        x1_cat = torch.cat(x1_list, dim=1)
        x1_f = self.fuse_x1(x1_cat)  # (B, f, D,H,W)

        # Level 2
        x2_list = [features_per_modality[m][1] for m in range(self.n_modalities)]
        x2_cat = torch.cat(x2_list, dim=1)
        x2_f = self.fuse_x2(x2_cat)  # (B, 2f, ...)

        # Level 3
        x3_list = [features_per_modality[m][2] for m in range(self.n_modalities)]
        x3_cat = torch.cat(x3_list, dim=1)
        x3_f = self.fuse_x3(x3_cat)  # (B, 4f, ...)

        # Level 4
        x4_list = [features_per_modality[m][3] for m in range(self.n_modalities)]
        x4_cat = torch.cat(x4_list, dim=1)
        x4_f = self.fuse_x4(x4_cat)  # (B, 8f, ...)

        # Level 5
        x5_list = [features_per_modality[m][4] for m in range(self.n_modalities)]
        x5_cat = torch.cat(x5_list, dim=1)
        x5_f = self.fuse_x5(x5_cat)  # (B,16f, ...)

        # có thể dropout thêm ở level 5
        if self.has_dropout:
            x5_f = self.dropout(x5_f)

        return [x1_f, x2_f, x3_f, x4_f, x5_f]

    # -------------------- Decoder --------------------

    def decoder(self, features):
        """
        features: list [x1, x2, x3, x4, x5] đã fuse.
        Giống VNet.decoder truyền thống.
        """
        x1, x2, x3, x4, x5 = features

        x5_up = self.block_five_up(x5)  # (B, 8f, ...)
        x5_up = x5_up + x4

        x6 = self.block_six(x5_up)      # (B, 8f, ...)
        x6_up = self.block_six_up(x6)   # (B, 4f, ...)
        x6_up = x6_up + x3

        x7 = self.block_seven(x6_up)    # (B, 4f, ...)
        x7_up = self.block_seven_up(x7) # (B, 2f, ...)
        x7_up = x7_up + x2

        x8 = self.block_eight(x7_up)    # (B, 2f, ...)
        x8_up = self.block_eight_up(x8) # (B, f, ...)
        x8_up = x8_up + x1

        x9 = self.block_nine(x8_up)     # (B, f, ...)

        if self.has_dropout:
            x9 = self.dropout(x9)

        out = self.out_conv(x9)         # (B, n_classes, D,H,W)
        return out

    # -------------------- Forward --------------------

    def forward(self, x, turnoff_drop: bool = False):
        """
        x: (B, n_modalities, D, H, W)
        """
        if turnoff_drop:
            # tạm tắt dropout trong forward (nếu dùng cho test TTA v.v.)
            has_dropout = self.has_dropout
            self.has_dropout = False

        features = self.encoder(x)
        out = self.decoder(features)

        if turnoff_drop:
            self.has_dropout = has_dropout

        return out


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from thop import profile, clever_format

    model = VNetMultiEncFusion(
        n_modalities=4,
        n_classes=4,
        n_filters=16,
        normalization='instancenorm',
        has_dropout=True,
    )

    inputs = torch.randn(2, 4, 96, 96, 96)  # (B, C=4, D,H,W)
    with torch.no_grad():
        seg = model(inputs)
    print("seg:", seg.shape)  # expected: (2, 4, 96, 96, 96)

    flops, params = profile(model, inputs=(inputs,))
    macs, params = clever_format([flops, params], "%.3f")
    print("FLOPs/Params:", macs, params)
