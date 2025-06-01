import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional

from .vision_encoder import VisionEncoder
from .text_encoder import TextEncoder
from .dual_head_fusion import DualHeadFusion
from .neural_memory import NeuralMemory


class MAVLModel(nn.Module):
    """
    Multi-Aspect Vision-Language (MAVL) model for zero-shot lung disease detection.
    Integrates all components described in the paper.
    """
    
    def __init__(self, config: Dict):
        """
        Args:
            config: Configuration dictionary containing model parameters
        """
        super().__init__()
        
        self.config = config
        model_config = config['model']
        
        # Initialize components
        # Vision Encoder (Section 3.3)
        self.vision_encoder = VisionEncoder(
            model_name=model_config['vision_encoder_name'],
            pretrained=True,
            hidden_dim=model_config['vision_hidden_dim'],
            output_dim=model_config['vision_output_dim'],
            dropout=model_config['vision_dropout']
        )
        
        # Text Encoder (Section 3.3)
        self.text_encoder = TextEncoder(
            model_name=model_config['text_encoder_name'],
            hidden_dim=model_config['text_hidden_dim'],
            output_dim=model_config['text_output_dim'],
            max_length=model_config['text_max_length'],
            dropout=model_config['text_dropout']
        )
        
        # Dual-Head Fusion (Section 3.4 & 3.5)
        self.dual_head = DualHeadFusion(
            vision_dim=model_config['vision_output_dim'],
            text_dim=model_config['text_output_dim'],
            num_heads=model_config['num_heads'],
            num_classes=len(config['data']['disease_labels']),
            num_aspect_queries=model_config['num_aspect_queries'],
            hidden_dim=model_config['fusion_hidden_dim'],
            dropout=model_config['fusion_dropout'],
            temperature=model_config['temperature']
        )
        
        # Neural Memory (Section 3.6)
        self.neural_memory = NeuralMemory(
            memory_size=model_config['memory_size'],
            memory_dim=model_config['memory_dim'],
            learning_rate=model_config['memory_lr'],
            momentum_beta=model_config['memory_beta'],
            update_alpha=model_config['memory_alpha']
        )
        
        # Disease labels for creating prompts
        self.disease_labels = config['data']['disease_labels']
        
        # Temperature for contrastive loss
        self.temperature = model_config['temperature']
        
        # Lambda for loss weighting
        self.lambda_contrastive = model_config['lambda_contrastive']
    
    def forward(
        self,
        images: torch.Tensor,
        reports: List[str],
        use_memory: bool = True,
        return_features: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the MAVL model.
        
        Args:
            images: Batch of images, shape (B, 3, H, W)
            reports: List of radiology reports
            use_memory: Whether to use neural memory adaptation
            return_features: Whether to return intermediate features
            
        Returns:
            Dictionary containing:
                - logits: Final disease predictions (B, num_classes)
                - contrastive_logits: Contrastive branch outputs
                - supervised_logits: Supervised branch outputs
                - attention_weights: Cross-attention weights
                - features: Intermediate features if requested
        """
        # Extract visual features
        patch_features, global_features = self.vision_encoder(images)
        # patch_features: (B, N, vision_dim)
        # global_features: (B, vision_dim)
        
        # Adapt global features using neural memory if enabled
        if use_memory and self.neural_memory is not None:
            adapted_global = self.neural_memory.adapt(global_features)
        else:
            adapted_global = global_features
        
        # Encode text reports
        text_embeddings = self.text_encoder(reports)
        # text_embeddings: (B, text_dim)
        
        # Dual-head fusion
        contrastive_logits, supervised_logits, final_logits, attention_weights = self.dual_head(
            vision_features=patch_features,
            text_embeddings=text_embeddings,
            vision_global=adapted_global,
            return_attention=True
        )
        
        # Prepare output dictionary
        outputs = {
            'logits': final_logits,
            'contrastive_logits': contrastive_logits,
            'supervised_logits': supervised_logits,
            'attention_weights': attention_weights
        }
        
        # Add features if requested
        if return_features:
            outputs['features'] = {
                'patch_features': patch_features,
                'global_features': global_features,
                'adapted_global': adapted_global,
                'text_embeddings': text_embeddings
            }
        
        return outputs
    
    def compute_loss(
        self,
        outputs: Dict[str, torch.Tensor],
        labels: torch.Tensor,
        is_unseen: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute combined loss (Equation 10).
        
        Args:
            outputs: Model outputs from forward pass
            labels: Ground truth labels (B, num_classes)
            is_unseen: Binary mask indicating unseen diseases
            
        Returns:
            Dictionary containing individual and total losses
        """
        # Supervised loss (multi-label BCE)
        supervised_loss = F.binary_cross_entropy_with_logits(
            outputs['supervised_logits'],
            labels,
            reduction='none'
        )
        
        # Mask out unseen diseases if provided
        if is_unseen is not None:
            supervised_loss = supervised_loss * (~is_unseen).float()
        
        supervised_loss = supervised_loss.mean()
        
        # Contrastive loss
        # For simplicity, we use a proxy contrastive loss
        # In practice, this would compare against all disease embeddings
        contrastive_scores = outputs['contrastive_logits']
        
        # Create positive/negative masks from labels
        positive_mask = labels > 0.5
        negative_mask = ~positive_mask
        
        # Compute contrastive loss (simplified version of Eq. 4)
        positive_scores = contrastive_scores[positive_mask]
        negative_scores = contrastive_scores[negative_mask]
        
        if positive_scores.numel() > 0 and negative_scores.numel() > 0:
            # InfoNCE-style loss
            positive_exp = torch.exp(positive_scores)
            negative_exp = torch.exp(negative_scores)
            
            contrastive_loss = -torch.log(
                positive_exp.sum() / (positive_exp.sum() + negative_exp.sum() + 1e-8)
            )
        else:
            contrastive_loss = torch.tensor(0.0, device=contrastive_scores.device)
        
        # Total loss (Eq. 10)
        total_loss = supervised_loss + self.lambda_contrastive * contrastive_loss
        
        return {
            'loss': total_loss,
            'supervised_loss': supervised_loss,
            'contrastive_loss': contrastive_loss
        }
    
    def update_memory(self, loss: torch.Tensor):
        """Update neural memory with current loss."""
        if self.neural_memory is not None and self.training:
            self.neural_memory.update_memory(loss, retain_graph=True)
    
    @torch.no_grad()
    def predict(
        self,
        images: torch.Tensor,
        reports: Optional[List[str]] = None,
        threshold: float = 0.5
    ) -> Dict[str, torch.Tensor]:
        """
        Make predictions on a batch of images.
        
        Args:
            images: Batch of images
            reports: Optional radiology reports
            threshold: Threshold for binary predictions
            
        Returns:
            Dictionary with predictions and probabilities
        """
        # Use default prompts if no reports provided
        if reports is None:
            reports = ["Chest X-ray image."] * images.shape[0]
        
        # Forward pass
        outputs = self.forward(images, reports, use_memory=True)
        
        # Convert logits to probabilities
        probabilities = torch.sigmoid(outputs['logits'])
        
        # Binary predictions
        predictions = (probabilities > threshold).float()
        
        return {
            'predictions': predictions,
            'probabilities': probabilities,
            'attention_weights': outputs['attention_weights']
        }
    
    def generate_disease_heatmaps(
        self,
        images: torch.Tensor,
        disease_indices: List[int]
    ) -> torch.Tensor:
        """
        Generate attention heatmaps for specific diseases.
        
        Args:
            images: Batch of images
            disease_indices: Indices of diseases to visualize
            
        Returns:
            Heatmaps of shape (B, len(disease_indices), H, W)
        """
        B = images.shape[0]
        
        # Get patch features
        patch_features, _ = self.vision_encoder(images)
        
        # Get attention maps for each disease
        heatmaps = []
        
        for disease_idx in disease_indices:
            attention_map = self.dual_head.get_disease_attention_map(
                patch_features,
                disease_idx
            )  # (B, K, N)
            
            # Average over aspect queries
            attention_map = attention_map.mean(dim=1)  # (B, N)
            
            # Reshape to 2D (assuming square patches)
            num_patches = attention_map.shape[1]
            h = w = int(num_patches ** 0.5)
            attention_map = attention_map.reshape(B, h, w)
            
            # Upsample to image size
            attention_map = F.interpolate(
                attention_map.unsqueeze(1),
                size=(224, 224),
                mode='bilinear',
                align_corners=False
            ).squeeze(1)
            
            heatmaps.append(attention_map)
        
        return torch.stack(heatmaps, dim=1)
    
    def freeze_encoders(self):
        """Freeze vision and text encoder parameters."""
        self.vision_encoder.freeze_backbone()
        self.text_encoder.freeze_backbone()
    
    def unfreeze_encoders(self):
        """Unfreeze vision and text encoder parameters."""
        self.vision_encoder.unfreeze_backbone()
        self.text_encoder.unfreeze_backbone()
    
    def get_num_parameters(self) -> Dict[str, int]:
        """Get number of parameters in each component."""
        def count_params(module):
            return sum(p.numel() for p in module.parameters() if p.requires_grad)
        
        return {
            'vision_encoder': count_params(self.vision_encoder),
            'text_encoder': count_params(self.text_encoder),
            'dual_head': count_params(self.dual_head),
            'neural_memory': count_params(self.neural_memory),
            'total': count_params(self)
        }