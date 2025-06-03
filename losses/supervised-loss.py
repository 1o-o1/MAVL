import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class MultiLabelBCELoss(nn.Module):
    """
    Multi-label Binary Cross Entropy Loss for disease classification.
    Used as L_sup in the paper.
    """
    
    def __init__(
        self,
        reduction: str = 'mean',
        pos_weight: Optional[torch.Tensor] = None,
        label_smoothing: float = 0.0
    ):
        """
        Args:
            reduction: How to reduce the loss ('mean', 'sum', 'none')
            pos_weight: Weight for positive samples (for class imbalance)
            label_smoothing: Label smoothing factor
        """
        super().__init__()
        self.reduction = reduction
        self.pos_weight = pos_weight
        self.label_smoothing = label_smoothing
    
    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute multi-label BCE loss.
        
        Args:
            logits: Model predictions, shape (B, num_classes)
            targets: Ground truth labels, shape (B, num_classes)
            mask: Optional mask for valid labels, shape (B, num_classes)
            
        Returns:
            loss: Multi-label BCE loss
        """
        # Apply label smoothing if specified
        if self.label_smoothing > 0:
            targets = targets * (1 - self.label_smoothing) + 0.5 * self.label_smoothing
        
        # Compute BCE loss
        if self.pos_weight is not None:
            loss = F.binary_cross_entropy_with_logits(
                logits,
                targets,
                reduction='none',
                pos_weight=self.pos_weight
            )
        else:
            loss = F.binary_cross_entropy_with_logits(
                logits,
                targets,
                reduction='none'
            )
        
        # Apply mask if provided
        if mask is not None:
            loss = loss * mask
            # Adjust denominator for mean reduction
            if self.reduction == 'mean':
                return loss.sum() / mask.sum().clamp(min=1)
        
        # Apply reduction
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance in multi-label classification.
    """
    
    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
        reduction: str = 'mean',
        label_smoothing: float = 0.0
    ):
        """
        Args:
            alpha: Weighting factor for positive/negative samples
            gamma: Focusing parameter
            reduction: How to reduce the loss
            label_smoothing: Label smoothing factor
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.label_smoothing = label_smoothing
    
    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute focal loss.
        
        Args:
            logits: Model predictions, shape (B, num_classes)
            targets: Ground truth labels, shape (B, num_classes)
            mask: Optional mask for valid labels
            
        Returns:
            loss: Focal loss value
        """
        # Apply label smoothing
        if self.label_smoothing > 0:
            targets = targets * (1 - self.label_smoothing) + 0.5 * self.label_smoothing
        
        # Compute probabilities
        probs = torch.sigmoid(logits)
        
        # Compute focal loss components
        ce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        p_t = probs * targets + (1 - probs) * (1 - targets)
        focal_weight = (1 - p_t) ** self.gamma
        
        # Apply alpha weighting
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        
        # Compute focal loss
        loss = alpha_t * focal_weight * ce_loss
        
        # Apply mask if provided
        if mask is not None:
            loss = loss * mask
            if self.reduction == 'mean':
                return loss.sum() / mask.sum().clamp(min=1)
        
        # Apply reduction
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class AsymmetricLoss(nn.Module):
    """
    Asymmetric Loss for multi-label classification with different
    weights for positive and negative samples.
    """
    
    def __init__(
        self,
        gamma_neg: float = 4.0,
        gamma_pos: float = 1.0,
        clip: float = 0.05,
        eps: float = 1e-8,
        reduction: str = 'mean'
    ):
        """
        Args:
            gamma_neg: Focusing parameter for negative samples
            gamma_pos: Focusing parameter for positive samples
            clip: Clipping threshold
            eps: Small constant for numerical stability
            reduction: How to reduce the loss
        """
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.eps = eps
        self.reduction = reduction
    
    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute asymmetric loss.
        
        Args:
            logits: Model predictions, shape (B, num_classes)
            targets: Ground truth labels, shape (B, num_classes)
            
        Returns:
            loss: Asymmetric loss value
        """
        # Compute probabilities
        probs = torch.sigmoid(logits)
        
        # Positive and negative probabilities
        xs_pos = probs
        xs_neg = 1 - probs
        
        # Asymmetric clipping
        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1)
        
        # Compute positive and negative losses
        los_pos = targets * torch.log(xs_pos.clamp(min=self.eps))
        los_neg = (1 - targets) * torch.log(xs_neg.clamp(min=self.eps))
        
        # Apply asymmetric focusing
        los_pos = los_pos * (1 - xs_pos) ** self.gamma_pos
        los_neg = los_neg * xs_pos ** self.gamma_neg
        
        # Combine losses
        loss = -(los_pos + los_neg)
        
        # Apply reduction
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class WeightedBCELoss(nn.Module):
    """
    Weighted BCE loss with per-class weights based on class frequency.
    """
    
    def __init__(
        self,
        class_weights: Optional[torch.Tensor] = None,
        reduction: str = 'mean'
    ):
        """
        Args:
            class_weights: Per-class weights, shape (num_classes,)
            reduction: How to reduce the loss
        """
        super().__init__()
        self.class_weights = class_weights
        self.reduction = reduction
    
    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute weighted BCE loss.
        
        Args:
            logits: Model predictions, shape (B, num_classes)
            targets: Ground truth labels, shape (B, num_classes)
            
        Returns:
            loss: Weighted BCE loss
        """
        # Compute BCE loss
        loss = F.binary_cross_entropy_with_logits(
            logits,
            targets,
            reduction='none'
        )
        
        # Apply class weights if provided
        if self.class_weights is not None:
            # Ensure weights are on same device
            if self.class_weights.device != loss.device:
                self.class_weights = self.class_weights.to(loss.device)
            
            # Apply weights
            loss = loss * self.class_weights.unsqueeze(0)
        
        # Apply reduction
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss