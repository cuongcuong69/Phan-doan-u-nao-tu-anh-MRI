# -*- coding: utf-8 -*-
"""
Tiện ích huấn luyện & SDF chuẩn hoá cho segmentation.
- AverageMeter/Logger/UnifLabelSampler cho training loop.
- compute_sdf_numpy(mask) -> [-1,1] cho mask nhị phân (WT vs BG).
"""

import os
import pickle
import numpy as np
from scipy.ndimage import distance_transform_edt as distance
from skimage import segmentation as skimage_seg
from torch.utils.data.sampler import Sampler


class UnifLabelSampler(Sampler):
    """
    Lấy mẫu đều theo nhóm pseudo-labels.
    Args:
        N (int): số phần tử muốn rút trong 1 epoch
        images_lists (dict[int, list[int]]): mỗi key là id nhãn giả, value là list chỉ số
    """
    def __init__(self, N, images_lists):
        self.N = int(N)
        self.images_lists = images_lists
        self.indexes = self.generate_indexes_epoch()

    def generate_indexes_epoch(self):
        size_per = int(self.N / len(self.images_lists)) + 1
        res = np.zeros(size_per * len(self.images_lists), dtype=np.int64)
        for i in range(len(self.images_lists)):
            cand = np.asarray(self.images_lists[i], dtype=np.int64)
            choose = np.random.choice(
                cand, size=size_per, replace=(cand.size <= size_per)
            )
            res[i * size_per:(i + 1) * size_per] = choose
        np.random.shuffle(res)
        return res[:self.N]

    def __iter__(self):
        return iter(self.indexes.tolist())

    def __len__(self):
        return self.N


class AverageMeter:
    """Theo dõi giá trị trung bình/hiện tại của 1 thước đo."""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val, n=1):
        self.val = float(val)
        self.sum += float(val) * int(n)
        self.count += int(n)
        self.avg = self.sum / max(1, self.count)


class Logger:
    """
    Ghi log theo epoch/iter.
    Dùng: logger = Logger('experiments/train_log.pkl'); logger.log({...})
    """
    def __init__(self, path):
        self.path = path
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.data = []

    def log(self, train_point: dict):
        self.data.append(train_point)
        with open(self.path, 'wb') as fp:
            pickle.dump(self.data, fp, -1)


def compute_sdf_numpy(img_gt: np.ndarray) -> np.ndarray:
    """
    Tính Signed Distance Field chuẩn hoá [-1,1] cho mask nhị phân (numpy).

    Input:
        img_gt: [B, D, H, W] hoặc [D,H,W] bool/0-1
    Return:
        sdf: cùng shape, float32 trong ~[-1,1], đường biên = 0.
    """
    arr = img_gt.astype(np.uint8)
    out_shape = arr.shape
    if arr.ndim == 3:  # không có batch
        arr = arr[None, ...]

    B = arr.shape[0]
    normalized = np.zeros_like(arr, dtype=np.float32)

    for b in range(B):
        posmask = arr[b].astype(bool)
        if posmask.any():
            negmask = ~posmask
            posdis = distance(posmask)
            negdis = distance(negmask)
            boundary = skimage_seg.find_boundaries(posmask, mode='inner').astype(np.uint8)

            # normalize hai nhánh rồi lấy neg - pos (trong ~[-1,1])
            pn = (posdis - posdis.min()) / max(1e-6, (posdis.max() - posdis.min()))
            nn = (negdis - negdis.min()) / max(1e-6, (negdis.max() - negdis.min()))
            sdf = nn - pn
            sdf[boundary == 1] = 0.0
            normalized[b] = sdf
        else:
            normalized[b] = 1.0  # toàn background → khoảng cách dương

    return normalized.astype(np.float32).reshape(out_shape)
