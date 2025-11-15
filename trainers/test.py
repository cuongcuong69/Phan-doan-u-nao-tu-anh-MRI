import torch
from models.vnet_multi_enc_fusion import VNetMultiEncFusion

model = VNetMultiEncFusion(
    n_modalities=4,
    n_classes=4,
    n_filters=32,
    normalization="instancenorm",
    has_dropout=True,
)

x = torch.randn(1, 4, 64, 64, 64)  # patch_size bạn đang dùng
with torch.no_grad():
    y = model(x)
print(y.shape)
