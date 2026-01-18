# -*- coding: utf-8 -*-
"""
Combined Loss Functions for 3D Medical Image Segmentation (UNETR)

Implements:
- Soft Dice Loss (multi-class)
- Combined Loss (Soft Dice + Cross-Entropy)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class SoftDiceLoss(nn.Module):
    """
    Multi-class Soft Dice Loss for 3D segmentation.
    
    Args:
        num_classes: Number of classes (including background)
        smooth: Smoothing factor to avoid division by zero
        ignore_bg: If True, ignore background class (class 0) in loss calculation
        class_weights: Optional weights for each class
    """
    def __init__(
        self,
        num_classes: int = 4,
        smooth: float = 1e-5,
        ignore_bg: bool = True,
        class_weights: Optional[torch.Tensor] = None,
    ):
        super(SoftDiceLoss, self).__init__()
        self.num_classes = num_classes
        self.smooth = smooth
        self.ignore_bg = ignore_bg
        self.class_weights = class_weights
        
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: (B, C, D, H, W) - raw logits from model
            targets: (B, D, H, W) - ground truth labels [0, C-1]
            
        Returns:
            Scalar loss value
        """
        # Softmax to get probabilities
        probs = F.softmax(logits, dim=1)  # (B, C, D, H, W)
        
        # Convert targets to one-hot encoding
        targets_one_hot = F.one_hot(targets.long(), num_classes=self.num_classes)  # (B, D, H, W, C)
        targets_one_hot = targets_one_hot.permute(0, 4, 1, 2, 3).float()  # (B, C, D, H, W)
        
        # Calculate dice for each class
        dims = (0, 2, 3, 4)  # Reduce over batch and spatial dimensions
        intersection = torch.sum(probs * targets_one_hot, dim=dims)  # (C,)
        cardinality = torch.sum(probs + targets_one_hot, dim=dims)  # (C,)
        
        dice_per_class = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)  # (C,)
        
        # Apply class weights if provided
        if self.class_weights is not None:
            weights = self.class_weights.to(dice_per_class.device)
            if self.ignore_bg:
                dice_per_class = dice_per_class[1:] * weights[1:]
            else:
                dice_per_class = dice_per_class * weights
        else:
            if self.ignore_bg:
                dice_per_class = dice_per_class[1:]  # Ignore background
        
        # Return 1 - mean_dice as loss
        dice_loss = 1.0 - dice_per_class.mean()
        
        return dice_loss


class CombinedLoss(nn.Module):
    """
    Combined Loss = w_dice * Soft Dice Loss + w_ce * Cross-Entropy Loss
    
    Args:
        num_classes: Number of classes
        w_dice: Weight for Dice loss
        w_ce: Weight for Cross-Entropy loss
        smooth: Smoothing factor for Dice loss
        ignore_bg: If True, ignore background in Dice calculation
        class_weights: Optional weights for each class (applied to both losses)
    """
    def __init__(
        self,
        num_classes: int = 4,
        w_dice: float = 1.0,
        w_ce: float = 1.0,
        smooth: float = 1e-5,
        ignore_bg: bool = True,
        class_weights: Optional[torch.Tensor] = None,
    ):
        super(CombinedLoss, self).__init__()
        self.w_dice = w_dice
        self.w_ce = w_ce
        self.num_classes = num_classes
        
        # Dice loss
        self.dice_loss = SoftDiceLoss(
            num_classes=num_classes,
            smooth=smooth,
            ignore_bg=ignore_bg,
            class_weights=class_weights,
        )
        
        # Cross-Entropy loss
        if class_weights is not None:
            self.ce_loss = nn.CrossEntropyLoss(weight=class_weights)
        else:
            self.ce_loss = nn.CrossEntropyLoss()
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> dict:
        """
        Args:
            logits: (B, C, D, H, W)
            targets: (B, D, H, W)
            
        Returns:
            Dictionary with 'total', 'dice', 'ce' losses
        """
        dice = self.dice_loss(logits, targets)
        ce = self.ce_loss(logits, targets)
        
        total = self.w_dice * dice + self.w_ce * ce
        
        return {
            'total': total,
            'dice': dice,
            'ce': ce,
        }


# Convenience function
def get_combined_loss(
    num_classes: int = 4,
    w_dice: float = 1.0,
    w_ce: float = 1.0,
    smooth: float = 1e-5,
    ignore_bg: bool = True,
    class_weights: Optional[list] = None,
) -> CombinedLoss:
    """
    Factory function to create CombinedLoss.
    
    Example:
        >>> loss_fn = get_combined_loss(num_classes=4, w_dice=1.0, w_ce=1.0)
        >>> logits = torch.randn(2, 4, 64, 64, 64)
        >>> targets = torch.randint(0, 4, (2, 64, 64, 64))
        >>> loss_dict = loss_fn(logits, targets)
        >>> loss_dict['total'].backward()
    """
    if class_weights is not None:
        class_weights = torch.tensor(class_weights, dtype=torch.float32)
    
    return CombinedLoss(
        num_classes=num_classes,
        w_dice=w_dice,
        w_ce=w_ce,
        smooth=smooth,
        ignore_bg=ignore_bg,
        class_weights=class_weights,
    )


if __name__ == "__main__":
    # Self-test
    print("=== Testing Combined Loss ===")
    
    # Create dummy data
    batch_size = 2
    num_classes = 4
    D, H, W = 32, 32, 32
    
    logits = torch.randn(batch_size, num_classes, D, H, W)
    targets = torch.randint(0, num_classes, (batch_size, D, H, W))
    
    # Test SoftDiceLoss
    dice_loss_fn = SoftDiceLoss(num_classes=num_classes)
    dice_loss = dice_loss_fn(logits, targets)
    print(f"Soft Dice Loss: {dice_loss.item():.4f}")
    
    # Test CombinedLoss
    combined_loss_fn = CombinedLoss(num_classes=num_classes, w_dice=1.0, w_ce=1.0)
    loss_dict = combined_loss_fn(logits, targets)
    print(f"Combined Loss - Total: {loss_dict['total'].item():.4f}, "
          f"Dice: {loss_dict['dice'].item():.4f}, CE: {loss_dict['ce'].item():.4f}")
    
    # Test backward
    loss_dict['total'].backward()
    print("[OK] Backward pass successful")
    
    print("[OK] Combined Loss self-test passed!")
