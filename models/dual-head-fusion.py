import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import math


class DualHeadFusion(nn.Module):
    """
    Dual-head architecture with contrastive and supervised branches.
    Implements the fusion mechanism described in Section 3.4 and 3.5 of the paper.
    """
    
    def __init__(
        self,
        vision_dim: int = 1024,
        text_dim: int = 1024,
        num_heads: int = 8,
        num_classes: int = 14,
        num_aspect_queries: int = 10,
        hidden_dim: Optional[int] = None,
        dropout: float = 0.5,
        temperature: float = 0.1
    ):
        """
        Args:
            vision_dim: Dimension of vision features
            text_dim: Dimension of text features
            num_heads: Number of attention heads
            num_classes: Number of disease classes
            num_aspect_queries: Number of aspect queries per class (K)
            hidden_dim: Hidden dimension for MLPs
            dropout: Dropout rate
            temperature: Temperature for contrastive loss (τ)
        """
        super().__init__()
        
        if hidden_dim is None:
            hidden_dim = vision_dim // 2
        
        self.vision_dim = vision_dim
        self.text_dim = text_dim
        self.num_heads = num_heads
        self.num_classes = num_classes
        self.num_aspect_queries = num_aspect_queries
        self.temperature = temperature
        
        # Initialize aspect queries for each disease (Section 3.3)
        # Q_j ∈ R^{K×D} for each disease category j
        self.aspect_queries = nn.Parameter(
            torch.randn(num_classes, num_aspect_queries, vision_dim) / math.sqrt(vision_dim)
        )
        
        # Contrastive Head (Section 3.4)
        self.contrastive_head = nn.MultiheadAttention(
            embed_dim=vision_dim,
            num_heads=num_heads,
            batch_first=True,
            dropout=dropout
        )
        
        # Vision and text projectors for alignment
        self.vision_projector = nn.Linear(vision_dim, text_dim)
        self.text_projector = nn.Linear(text_dim, vision_dim)
        
        # Supervised Head - Cross Attention (Section 3.5)
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=vision_dim,
            num_heads=num_heads,
            batch_first=True,
            dropout=dropout
        )
        
        # Classification heads
        # Cross-attention based classifier (Eq. 5)
        self.fc_cross = nn.Sequential(
            nn.LayerNorm(vision_dim),
            nn.Dropout(dropout),
            nn.Linear(vision_dim, num_classes)
        )
        
        # Direct classification branch (Eq. 6)
        self.fc_direct = nn.Sequential(
            nn.LayerNorm(vision_dim),
            nn.Dropout(dropout),
            nn.Linear(vision_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )
        
        # Optional: Gating mechanism to combine branches
        self.use_gating = False
        if self.use_gating:
            self.gate = nn.Sequential(
                nn.Linear(vision_dim * 2, vision_dim),
                nn.ReLU(),
                nn.Linear(vision_dim, 2),
                nn.Softmax(dim=-1)
            )
    
    def forward(
        self,
        vision_features: torch.Tensor,
        text_embeddings: torch.Tensor,
        vision_global: torch.Tensor,
        return_attention: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass through dual-head architecture.
        
        Args:
            vision_features: Patch features from vision encoder (B, N, vision_dim)
            text_embeddings: Text embeddings (B, text_dim)
            vision_global: Global vision features (B, vision_dim)
            return_attention: Whether to return attention weights
            
        Returns:
            contrastive_scores: Similarity scores for contrastive branch (B, num_classes)
            supervised_logits: Logits from supervised branch (B, num_classes)
            final_logits: Combined final predictions (B, num_classes)
            attention_weights: Cross-attention weights if requested
        """
        B = vision_features.shape[0]
        
        # === Contrastive Branch (Section 3.4) ===
        # Align global vision and text embeddings
        vision_global_seq = vision_global.unsqueeze(1)  # (B, 1, vision_dim)
        text_embeddings_seq = text_embeddings.unsqueeze(1)  # (B, 1, text_dim)
        
        # Project to common space
        text_proj = self.text_projector(text_embeddings_seq)  # (B, 1, vision_dim)
        
        # Multi-head attention for alignment
        contrastive_out, _ = self.contrastive_head(
            vision_global_seq,
            text_proj,
            text_proj
        )
        
        # Normalize for contrastive loss (Eq. 4)
        contrastive_out_norm = F.normalize(contrastive_out, p=2, dim=-1)
        text_proj_norm = F.normalize(text_proj, p=2, dim=-1)
        
        # Compute similarity scores
        contrastive_scores = torch.bmm(
            contrastive_out_norm,
            text_proj_norm.transpose(1, 2)
        ).squeeze(1).squeeze(-1) / self.temperature  # (B,)
        
        # For multi-class, we need to compute scores for each disease
        # This is a simplified version - in practice, you'd have embeddings for each disease
        contrastive_logits = contrastive_scores.unsqueeze(-1).expand(-1, self.num_classes)
        
        # === Supervised Branch (Section 3.5) ===
        # Cross-attention with aspect queries
        all_attended_features = []
        all_attention_weights = []
        
        for j in range(self.num_classes):
            # Get aspect queries for disease j
            queries = self.aspect_queries[j].unsqueeze(0).expand(B, -1, -1)  # (B, K, vision_dim)
            
            # Cross-attention: queries attend to vision features
            attended_features, attn_weights = self.cross_attention(
                queries,
                vision_features,
                vision_features,
                need_weights=return_attention
            )  # attended_features: (B, K, vision_dim)
            
            # Aggregate attended features (mean pooling)
            aggregated = attended_features.mean(dim=1)  # (B, vision_dim)
            
            all_attended_features.append(aggregated)
            if return_attention:
                all_attention_weights.append(attn_weights)
        
        # Stack features for all classes
        attended_features = torch.stack(all_attended_features, dim=1)  # (B, num_classes, vision_dim)
        
        # Classification from attended features (Eq. 5)
        logits_cross = self.fc_cross(attended_features).squeeze(-1)  # (B, num_classes)
        
        # Direct classification from global features (Eq. 6)
        logits_direct = self.fc_direct(vision_global)  # (B, num_classes)
        
        # Combine supervised predictions (Eq. 7)
        supervised_logits = (logits_cross + logits_direct) / 2.0
        
        # === Final Combination ===
        # Combine contrastive and supervised branches
        if self.use_gating:
            # Learnable gating mechanism
            combined_features = torch.cat([vision_global, text_embeddings_seq.squeeze(1)], dim=-1)
            gates = self.gate(combined_features)  # (B, 2)
            final_logits = gates[:, 0:1] * supervised_logits + gates[:, 1:2] * contrastive_logits
        else:
            # Simple average (paper uses 1/2 weighting)
            final_logits = (supervised_logits + contrastive_logits) / 2.0
        
        # Prepare attention weights
        attention_weights = None
        if return_attention and all_attention_weights:
            attention_weights = torch.stack(all_attention_weights, dim=1)  # (B, num_classes, K, N)
        
        return contrastive_logits, supervised_logits, final_logits, attention_weights
    
    def get_disease_attention_map(
        self,
        vision_features: torch.Tensor,
        disease_idx: int
    ) -> torch.Tensor:
        """
        Get attention map for a specific disease.
        
        Args:
            vision_features: Patch features (B, N, vision_dim)
            disease_idx: Index of the disease class
            
        Returns:
            attention_map: Attention weights (B, K, N)
        """
        B = vision_features.shape[0]
        
        # Get aspect queries for the specified disease
        queries = self.aspect_queries[disease_idx].unsqueeze(0).expand(B, -1, -1)
        
        # Compute cross-attention
        _, attention_weights = self.cross_attention(
            queries,
            vision_features,
            vision_features,
            need_weights=True
        )
        
        return attention_weights