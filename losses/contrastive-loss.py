import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class InfoNCELoss(nn.Module):
    """
    InfoNCE loss for contrastive learning.
    Implements Equation (4) from the paper.
    """
    
    def __init__(
        self,
        temperature: float = 0.1,
        reduction: str = 'mean'
    ):
        """
        Args:
            temperature: Temperature parameter τ for scaling
            reduction: How to reduce the loss ('mean', 'sum', 'none')
        """
        super().__init__()
        self.temperature = temperature
        self.reduction = reduction
    
    def forward(
        self,
        anchor: torch.Tensor,
        positive: torch.Tensor,
        negatives: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute InfoNCE loss.
        
        Args:
            anchor: Anchor embeddings, shape (B, D)
            positive: Positive embeddings, shape (B, D)
            negatives: Negative embeddings, shape (B, N, D) or None
                      If None, other samples in batch are used as negatives
                      
        Returns:
            loss: InfoNCE loss value
        """
        B, D = anchor.shape
        
        # Normalize embeddings
        anchor = F.normalize(anchor, p=2, dim=1)
        positive = F.normalize(positive, p=2, dim=1)
        
        # Compute positive similarity
        positive_sim = torch.sum(anchor * positive, dim=1) / self.temperature  # (B,)
        
        if negatives is None:
            # Use other samples in batch as negatives
            # This is the standard NT-Xent loss formulation
            
            # Compute all pairwise similarities
            sim_matrix = torch.matmul(anchor, anchor.t()) / self.temperature  # (B, B)
            
            # Mask out diagonal (self-similarity)
            mask = torch.eye(B, dtype=torch.bool, device=anchor.device)
            sim_matrix = sim_matrix.masked_fill(mask, -float('inf'))
            
            # Add positive similarities
            sim_matrix[torch.arange(B), torch.arange(B)] = positive_sim
            
            # Compute loss
            log_prob = F.log_softmax(sim_matrix, dim=1)
            loss = -log_prob[torch.arange(B), torch.arange(B)]
        else:
            # Use provided negatives
            negatives = F.normalize(negatives, p=2, dim=-1)
            
            # Compute negative similarities
            negative_sim = torch.matmul(
                anchor.unsqueeze(1),  # (B, 1, D)
                negatives.transpose(1, 2)  # (B, D, N)
            ).squeeze(1) / self.temperature  # (B, N)
            
            # Concatenate positive and negative similarities
            logits = torch.cat([positive_sim.unsqueeze(1), negative_sim], dim=1)  # (B, 1+N)
            
            # Labels: positive is at index 0
            labels = torch.zeros(B, dtype=torch.long, device=anchor.device)
            
            # Compute cross entropy loss
            loss = F.cross_entropy(logits, labels, reduction='none')
        
        # Apply reduction
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class ContrastiveLoss(nn.Module):
    """
    General contrastive loss supporting multiple formulations.
    """
    
    def __init__(
        self,
        temperature: float = 0.1,
        loss_type: str = 'infonce',
        margin: float = 1.0,
        reduction: str = 'mean'
    ):
        """
        Args:
            temperature: Temperature for scaling similarities
            loss_type: Type of contrastive loss ('infonce', 'triplet', 'ntxent')
            margin: Margin for triplet loss
            reduction: How to reduce the loss
        """
        super().__init__()
        self.temperature = temperature
        self.loss_type = loss_type
        self.margin = margin
        self.reduction = reduction
        
        if loss_type == 'infonce':
            self.loss_fn = InfoNCELoss(temperature, reduction)
        elif loss_type == 'triplet':
            self.loss_fn = nn.TripletMarginLoss(margin=margin, reduction=reduction)
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")
    
    def forward(
        self,
        embeddings1: torch.Tensor,
        embeddings2: torch.Tensor,
        labels: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute contrastive loss between two sets of embeddings.
        
        Args:
            embeddings1: First set of embeddings (e.g., vision)
            embeddings2: Second set of embeddings (e.g., text)
            labels: Optional labels for supervised contrastive learning
            
        Returns:
            loss: Contrastive loss value
        """
        if self.loss_type == 'infonce':
            return self.loss_fn(embeddings1, embeddings2)
        elif self.loss_type == 'triplet':
            # For triplet loss, we need to create triplets
            # This is a simplified version
            B = embeddings1.shape[0]
            
            # Use cyclic shifts to create negatives
            anchor = embeddings1
            positive = embeddings2
            negative = torch.roll(embeddings2, shifts=1, dims=0)
            
            return self.loss_fn(anchor, positive, negative)
        else:
            raise ValueError(f"Unsupported loss type: {self.loss_type}")


class NTXentLoss(nn.Module):
    """
    Normalized Temperature-scaled Cross Entropy Loss (NT-Xent).
    Used in SimCLR and similar frameworks.
    """
    
    def __init__(
        self,
        temperature: float = 0.1,
        reduction: str = 'mean'
    ):
        super().__init__()
        self.temperature = temperature
        self.reduction = reduction
        self.similarity = nn.CosineSimilarity(dim=2)
    
    def forward(
        self,
        z_i: torch.Tensor,
        z_j: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            z_i: First view embeddings, shape (B, D)
            z_j: Second view embeddings, shape (B, D)
            
        Returns:
            loss: NT-Xent loss
        """
        B = z_i.shape[0]
        
        # Concatenate representations
        representations = torch.cat([z_i, z_j], dim=0)  # (2B, D)
        
        # Compute similarity matrix
        similarity_matrix = self.similarity(
            representations.unsqueeze(1),
            representations.unsqueeze(0)
        ) / self.temperature  # (2B, 2B)
        
        # Create mask for positive pairs
        mask = torch.eye(2 * B, dtype=torch.bool, device=z_i.device)
        mask = mask ^ torch.roll(mask, B, dims=0)  # XOR to get positive pairs
        
        # Extract positive and negative similarities
        positive_samples = similarity_matrix[mask].view(2 * B, 1)
        negative_samples = similarity_matrix[~mask].view(2 * B, -1)
        
        # Compute loss
        logits = torch.cat([positive_samples, negative_samples], dim=1)
        labels = torch.zeros(2 * B, dtype=torch.long, device=z_i.device)
        
        loss = F.cross_entropy(logits, labels, reduction=self.reduction)
        
        return loss