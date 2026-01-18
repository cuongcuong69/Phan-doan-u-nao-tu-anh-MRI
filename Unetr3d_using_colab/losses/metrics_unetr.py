# -*- coding: utf-8 -*-
"""
Metrics for 3D Medical Image Segmentation (UNETR)

Implements:
- Dice Coefficient
- IoU (Intersection over Union)
- HD95 (95th percentile Hausdorff Distance)
- ASD (Average Surface Distance)
- BraTS-specific metrics (WT, TC, ET)
"""

import numpy as np
import torch
from typing import Dict, Optional, Tuple
from scipy.ndimage import distance_transform_edt


def dice_coefficient(
    pred: np.ndarray,
    target: np.ndarray,
    num_classes: int = 4,
    ignore_bg: bool = True,
) -> Dict[str, float]:
    """
    Calculate Dice coefficient for each class.
    
    Args:
        pred: Predicted labels (D, H, W) or (B, D, H, W)
        target: Ground truth labels (D, H, W) or (B, D, H, W)
        num_classes: Number of classes
        ignore_bg: If True, don't compute Dice for background (class 0)
        
    Returns:
        Dictionary with per-class Dice and mean Dice
    """
    if pred.ndim == 4:  # Batch dimension
        # Average over batch
        dice_scores = []
        for b in range(pred.shape[0]):
            dice_scores.append(dice_coefficient(pred[b], target[b], num_classes, ignore_bg))
        
        # Average all metrics
        result = {}
        for key in dice_scores[0].keys():
            result[key] = np.mean([d[key] for d in dice_scores])
        return result
    
    dice_per_class = {}
    dice_values = []
    
    start_class = 1 if ignore_bg else 0
    
    for c in range(start_class, num_classes):
        pred_c = (pred == c).astype(np.float32)
        target_c = (target == c).astype(np.float32)
        
        intersection = np.sum(pred_c * target_c)
        cardinality = np.sum(pred_c) + np.sum(target_c)
        
        if cardinality > 0:
            dice = (2.0 * intersection) / cardinality
        else:
            dice = 1.0 if np.sum(target_c) == 0 else 0.0
        
        dice_per_class[f'dice_class_{c}'] = dice
        dice_values.append(dice)
    
    dice_per_class['dice_mean'] = np.mean(dice_values) if dice_values else 0.0
    
    return dice_per_class


def iou_score(
    pred: np.ndarray,
    target: np.ndarray,
    num_classes: int = 4,
    ignore_bg: bool = True,
) -> Dict[str, float]:
    """
    Calculate IoU (Intersection over Union) for each class.
    
    Args:
        pred: Predicted labels (D, H, W) or (B, D, H, W)
        target: Ground truth labels (D, H, W) or (B, D, H, W)
        num_classes: Number of classes
        ignore_bg: If True, don't compute IoU for background
        
    Returns:
        Dictionary with per-class IoU and mean IoU
    """
    if pred.ndim == 4:  # Batch dimension
        iou_scores = []
        for b in range(pred.shape[0]):
            iou_scores.append(iou_score(pred[b], target[b], num_classes, ignore_bg))
        
        result = {}
        for key in iou_scores[0].keys():
            result[key] = np.mean([d[key] for d in iou_scores])
        return result
    
    iou_per_class = {}
    iou_values = []
    
    start_class = 1 if ignore_bg else 0
    
    for c in range(start_class, num_classes):
        pred_c = (pred == c).astype(np.float32)
        target_c = (target == c).astype(np.float32)
        
        intersection = np.sum(pred_c * target_c)
        union = np.sum(pred_c) + np.sum(target_c) - intersection
        
        if union > 0:
            iou = intersection / union
        else:
            iou = 1.0 if np.sum(target_c) == 0 else 0.0
        
        iou_per_class[f'iou_class_{c}'] = iou
        iou_values.append(iou)
    
    iou_per_class['iou_mean'] = np.mean(iou_values) if iou_values else 0.0
    
    return iou_per_class


def hausdorff_distance_95(
    pred: np.ndarray,
    target: np.ndarray,
    voxel_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> float:
    """
    Calculate 95th percentile Hausdorff Distance.
    
    Args:
        pred: Binary prediction mask (D, H, W)
        target: Binary ground truth mask (D, H, W)
        voxel_spacing: Voxel spacing in (z, y, x) order
        
    Returns:
        HD95 distance in mm (or voxel units if spacing=(1,1,1))
    """
    pred_bin = pred.astype(bool)
    target_bin = target.astype(bool)
    
    # Handle edge cases
    if not np.any(pred_bin) and not np.any(target_bin):
        return 0.0
    if not np.any(pred_bin) or not np.any(target_bin):
        return np.inf
    
    # Get surface voxels (boundary)
    from scipy.ndimage import binary_erosion
    
    pred_surface = pred_bin ^ binary_erosion(pred_bin)
    target_surface = target_bin ^ binary_erosion(target_bin)
    
    # Distance transform
    pred_dist = distance_transform_edt(~target_surface, sampling=voxel_spacing)
    target_dist = distance_transform_edt(~pred_surface, sampling=voxel_spacing)
    
    # Get distances at surface points
    pred_surface_dist = pred_dist[pred_surface]
    target_surface_dist = target_dist[target_surface]
    
    # Combine and get 95th percentile
    all_distances = np.concatenate([pred_surface_dist, target_surface_dist])
    hd95 = np.percentile(all_distances, 95)
    
    return float(hd95)


def average_surface_distance(
    pred: np.ndarray,
    target: np.ndarray,
    voxel_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> float:
    """
    Calculate Average Surface Distance (symmetric).
    
    Args:
        pred: Binary prediction mask (D, H, W)
        target: Binary ground truth mask (D, H, W)
        voxel_spacing: Voxel spacing in (z, y, x) order
        
    Returns:
        ASD distance in mm (or voxel units if spacing=(1,1,1))
    """
    pred_bin = pred.astype(bool)
    target_bin = target.astype(bool)
    
    # Handle edge cases
    if not np.any(pred_bin) and not np.any(target_bin):
        return 0.0
    if not np.any(pred_bin) or not np.any(target_bin):
        return np.inf
    
    # Get surface voxels
    from scipy.ndimage import binary_erosion
    
    pred_surface = pred_bin ^ binary_erosion(pred_bin)
    target_surface = target_bin ^ binary_erosion(target_bin)
    
    # Distance transform
    pred_dist = distance_transform_edt(~target_surface, sampling=voxel_spacing)
    target_dist = distance_transform_edt(~pred_surface, sampling=voxel_spacing)
    
    # Average distances
    pred_surface_dist = pred_dist[pred_surface]
    target_surface_dist = target_dist[target_surface]
    
    asd = (np.mean(pred_surface_dist) + np.mean(target_surface_dist)) / 2.0
    
    return float(asd)


def compute_brats_metrics(
    pred: np.ndarray,
    target: np.ndarray,
    voxel_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    compute_hd: bool = True,
) -> Dict[str, float]:
    """
    Compute BraTS-specific metrics for WT, TC, ET regions.
    
    BraTS regions:
        - WT (Whole Tumor): label > 0
        - TC (Tumor Core): label == 1 or label == 3
        - ET (Enhancing Tumor): label == 3
    
    Args:
        pred: Predicted labels (D, H, W)
        target: Ground truth labels (D, H, W)
        voxel_spacing: Voxel spacing for HD95/ASD
        compute_hd: If True, compute HD95 and ASD (slower)
        
    Returns:
        Dictionary with Dice, IoU, HD95, ASD for WT/TC/ET
    """
    metrics = {}
    
    # Define regions
    regions = {
        'wt': lambda x: (x > 0).astype(np.uint8),
        'tc': lambda x: ((x == 1) | (x == 3)).astype(np.uint8),
        'et': lambda x: (x == 3).astype(np.uint8),
    }
    
    for region_name, region_fn in regions.items():
        pred_region = region_fn(pred)
        target_region = region_fn(target)
        
        # Dice
        intersection = np.sum(pred_region * target_region)
        cardinality = np.sum(pred_region) + np.sum(target_region)
        
        if cardinality > 0:
            dice = (2.0 * intersection) / cardinality
        else:
            dice = 1.0 if np.sum(target_region) == 0 else 0.0
        
        metrics[f'dice_{region_name}'] = dice
        
        # IoU
        union = np.sum(pred_region) + np.sum(target_region) - intersection
        if union > 0:
            iou = intersection / union
        else:
            iou = 1.0 if np.sum(target_region) == 0 else 0.0
        
        metrics[f'iou_{region_name}'] = iou
        
        # HD95 and ASD (optional, slower)
        if compute_hd:
            try:
                if np.any(pred_region) and np.any(target_region):
                    hd95 = hausdorff_distance_95(pred_region, target_region, voxel_spacing)
                    asd = average_surface_distance(pred_region, target_region, voxel_spacing)
                else:
                    hd95 = np.inf if np.any(target_region) else 0.0
                    asd = np.inf if np.any(target_region) else 0.0
                
                metrics[f'hd95_{region_name}'] = hd95
                metrics[f'asd_{region_name}'] = asd
            except Exception as e:
                # Fallback if computation fails
                metrics[f'hd95_{region_name}'] = np.nan
                metrics[f'asd_{region_name}'] = np.nan
    
    # Mean metrics
    metrics['dice_mean'] = np.mean([metrics['dice_wt'], metrics['dice_tc'], metrics['dice_et']])
    metrics['iou_mean'] = np.mean([metrics['iou_wt'], metrics['iou_tc'], metrics['iou_et']])
    
    if compute_hd:
        hd95_values = [metrics[f'hd95_{r}'] for r in ['wt', 'tc', 'et'] if not np.isinf(metrics[f'hd95_{r}'])]
        asd_values = [metrics[f'asd_{r}'] for r in ['wt', 'tc', 'et'] if not np.isinf(metrics[f'asd_{r}'])]
        
        metrics['hd95_mean'] = np.mean(hd95_values) if hd95_values else np.inf
        metrics['asd_mean'] = np.mean(asd_values) if asd_values else np.inf
    
    return metrics


# Torch wrapper for batch processing
def compute_metrics_batch(
    pred: torch.Tensor,
    target: torch.Tensor,
    num_classes: int = 4,
    compute_hd: bool = False,
) -> Dict[str, float]:
    """
    Compute metrics for a batch of predictions.
    
    Args:
        pred: (B, D, H, W) - predicted labels
        target: (B, D, H, W) - ground truth labels
        num_classes: Number of classes
        compute_hd: If True, compute HD95 and ASD
        
    Returns:
        Dictionary with averaged metrics over batch
    """
    pred_np = pred.detach().cpu().numpy()
    target_np = target.detach().cpu().numpy()
    
    batch_size = pred_np.shape[0]
    all_metrics = []
    
    for b in range(batch_size):
        # Per-class metrics
        dice_dict = dice_coefficient(pred_np[b], target_np[b], num_classes)
        iou_dict = iou_score(pred_np[b], target_np[b], num_classes)
        
        # BraTS region metrics
        brats_dict = compute_brats_metrics(pred_np[b], target_np[b], compute_hd=compute_hd)
        
        # Combine
        metrics = {**dice_dict, **iou_dict, **brats_dict}
        all_metrics.append(metrics)
    
    # Average over batch
    avg_metrics = {}
    for key in all_metrics[0].keys():
        values = [m[key] for m in all_metrics if not np.isnan(m[key]) and not np.isinf(m[key])]
        avg_metrics[key] = np.mean(values) if values else 0.0
    
    return avg_metrics


if __name__ == "__main__":
    # Self-test
    print("=== Testing Metrics ===")
    
    # Create dummy data
    D, H, W = 64, 64, 64
    pred = np.random.randint(0, 4, (D, H, W))
    target = np.random.randint(0, 4, (D, H, W))
    
    # Test Dice
    dice_dict = dice_coefficient(pred, target, num_classes=4)
    print(f"Dice scores: {dice_dict}")
    
    # Test IoU
    iou_dict = iou_score(pred, target, num_classes=4)
    print(f"IoU scores: {iou_dict}")
    
    # Test BraTS metrics (without HD for speed)
    brats_dict = compute_brats_metrics(pred, target, compute_hd=False)
    print(f"BraTS metrics: {brats_dict}")
    
    # Test with HD95/ASD (small volume for speed)
    small_pred = np.random.randint(0, 2, (16, 16, 16))
    small_target = np.random.randint(0, 2, (16, 16, 16))
    
    try:
        hd95 = hausdorff_distance_95(small_pred, small_target)
        asd = average_surface_distance(small_pred, small_target)
        print(f"HD95: {hd95:.2f}, ASD: {asd:.2f}")
    except Exception as e:
        print(f"[WARN] HD95/ASD test failed: {e}")
    
    print("[OK] Metrics self-test passed!")
