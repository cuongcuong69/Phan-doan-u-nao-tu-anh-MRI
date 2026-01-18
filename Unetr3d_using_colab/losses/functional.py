# -*- coding: utf-8 -*-
"""
Hàm tiện ích dùng chung (phiên bản torch nếu cần).
"""
import torch
from .util import compute_sdf_numpy


@torch.no_grad()
def compute_sdf_torch(mask: torch.Tensor, foreground_as_one: bool = True) -> torch.Tensor:
    """
    Tính SDF trên CPU bằng numpy rồi trả torch tensor (dùng cho inference/val).

    mask:
        - [N,1,D,H,W] hoặc [N,D,H,W]
        - Có thể là nhị phân 0/1 hoặc multi-class (0..C-1).

    Nếu foreground_as_one=True:
        - Mọi giá trị >0 được coi là foreground (WT) để tính SDF WT vs background.
    """
    if mask.dim() == 5:
        mask_np = mask[:, 0].detach().cpu().numpy()
    else:
        mask_np = mask.detach().cpu().numpy()

    if foreground_as_one:
        mask_np = (mask_np > 0).astype(mask_np.dtype)

    sdf_np = compute_sdf_numpy(mask_np)  # [N,D,H,W] hoặc [D,H,W]
    sdf = torch.from_numpy(sdf_np).to(mask.device, mask.dtype)
    if sdf.dim() == 4:
        return sdf.unsqueeze(1)  # [N,1,D,H,W]
    else:
        return sdf.unsqueeze(0).unsqueeze(0)
