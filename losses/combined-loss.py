import torch
import torch.nn as nn
from typing import Dict, Optional

from .contrastive_loss import ContrastiveLoss, InfoNCELoss
from .supervised_loss import MultiLabelBCELoss, FocalLoss


class CombinedLoss(nn.Module):
    """
    Combined loss function implementing Equation (10) from the paper:
    L_total = L_sup + λ * L_cont
    """
    
    def __init__(
        self,
        lambda_contrastive: float = 0.1,
        temperature: float = 0.1,
        supervised_loss_type: str = 'bce',
        contrastive_loss_type: str = 'infonce',
        use_focal_loss: bool = False,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
        label_smoothing: float = 0.0,
        pos_weight: Optional[torch.Tensor] = None
    ):
        """
        Args:
            lambda_contrastive: Weight for contrastive loss (λ in paper)
            temperature: Temperature for contrastive loss (τ in paper)
            supervised_loss_type: Type of supervised loss ('bce', 'focal', 'asymmetric')
            contrastive_loss_type: Type of contrastive loss ('infonce', 'ntxent')
            use_focal_loss: Whether to use focal loss for supervised branch
            focal_alpha: Alpha parameter for focal loss
            focal_gamma: Gamma parameter for focal loss
            label_smoothing: Label smoothing factor
            pos_weight: Positive class weights for handling imbalance
        """
        super().__init__()
        
        self.lambda_contrastive = lambda_contrastive
        
        # Initialize supervised loss
        if supervised_loss_type == 'bce':
            self.supervised_loss = MultiLabelBCELoss(
                reduction='mean',
                pos_weight=pos_weight,
                label_smoothing=label_smoothing
            )
        elif supervised_loss_type == 'focal':
            self.supervised_loss = FocalLoss(
                alpha=focal_alpha,
                gamma=focal_gamma,
                reduction='mean',
                label_smoothing=label_smoothing
            )
        else:
            raise ValueError(f"Unknown supervised loss type: {supervised_loss_type}")
        
        # Initialize contrastive loss
        if contrastive_loss_type == 'infonce':
            self.contrastive_loss = InfoNCELoss(
                temperature=temperature,
                reduction='mean'
            )
        else:
            self.contrastive_loss = ContrastiveLoss(
                temperature=temperature,
                loss_type=contrastive_loss_type,
                reduction='mean'
            )
    
    def forward(
        self,
        supervised_logits: torch.Tensor,
        contrastive_embeddings: Dict[str, torch.Tensor],
        labels: torch.Tensor,
        is_unseen: Optional[torch.Tensor] = None,
        return_components: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Compute combined loss.
        
        Args:
            supervised_logits: Predictions from supervised branch (B, num_classes)
            contrastive_embeddings: Dictionary containing:
                - 'vision': Vision embeddings (B, D)
                - 'text': Text embeddings (B, D)
                - 'negatives': Optional negative embeddings
            labels: Ground truth labels (B, num_classes)
            is_unseen: Binary mask for unseen diseases (B, num_classes)
            return_components: Whether to return individual loss components
            
        Returns:
            Dictionary containing:
                - 'loss': Total loss
                - 'supervised_loss': Supervised loss component
                - 'contrastive_loss': Contrastive loss component
        """
        # Compute supervised loss
        if is_unseen is not None:
            # Mask out unseen diseases
            mask = ~is_unseen
            supervised_loss = self.supervised_loss(
                supervised_logits,
                labels,
                mask=mask
            )
        else:
            supervised_loss = self.supervised_loss(
                supervised_logits,
                labels
            )
        
        # Compute contrastive loss
        vision_emb = contrastive_embeddings['vision']
        text_emb = contrastive_embeddings['text']
        negatives = contrastive_embeddings.get('negatives', None)
        
        if isinstance(self.contrastive_loss, InfoNCELoss):
            contrastive_loss = self.contrastive_loss(
                vision_emb,
                text_emb,
                negatives
            )
        else:
            contrastive_loss = self.contrastive_loss(
                vision_emb,
                text_emb
            )
        
        # Compute total loss (Equation 10)
        total_loss = supervised_loss + self.lambda_contrastive * contrastive_loss
        
        if return_components:
            return {
                'loss': total_loss,
                'supervised_loss': supervised_loss,
                'contrastive_loss': contrastive_loss
            }
        else:
            return total_loss


class AdaptiveCombinedLoss(CombinedLoss):
    """
    Combined loss with adaptive weighting that changes during training.
    """
    
    def __init__(
        self,
        initial_lambda: float = 0.1,
        final_lambda: float = 0.01,
        warmup_epochs: int = 5,
        total_epochs: int = 40,
        **kwargs
    ):
        """
        Args:
            initial_lambda: Initial weight for contrastive loss
            final_lambda: Final weight for contrastive loss
            warmup_epochs: Number of warmup epochs
            total_epochs: Total number of training epochs
            **kwargs: Additional arguments for parent class
        """
        super().__init__(lambda_contrastive=initial_lambda, **kwargs)
        
        self.initial_lambda = initial_lambda
        self.final_lambda = final_lambda
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.current_epoch = 0
    
    def update_epoch(self, epoch: int):
        """Update current epoch and adjust lambda accordingly."""
        self.current_epoch = epoch
        
        if epoch < self.warmup_epochs:
            # Linear warmup
            self.lambda_contrastive = self.initial_lambda * (epoch + 1) / self.warmup_epochs
        else:
            # Cosine annealing
            progress = (epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            self.lambda_contrastive = self.final_lambda + \
                (self.initial_lambda - self.final_lambda) * 0.5 * (1 + torch.cos(torch.tensor(progress * 3.14159)))
    
    def get_current_lambda(self) -> float:
        """Get current lambda value."""
        return self.lambda_contrastive