import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import timm
from einops import rearrange


class VisionEncoder(nn.Module):
    """
    Vision Transformer (ViT) based encoder for extracting visual features.
    Based on Equation (1) and (2) in the paper.
    """
    
    def __init__(
        self,
        model_name: str = "vit_base_patch16_224",
        pretrained: bool = True,
        hidden_dim: int = 768,
        output_dim: int = 1024,
        dropout: float = 0.1,
        freeze_backbone: bool = False
    ):
        """
        Args:
            model_name: Name of the ViT model from timm
            pretrained: Whether to use pretrained weights
            hidden_dim: Hidden dimension of ViT (768 for base)
            output_dim: Output dimension for projections
            dropout: Dropout rate
            freeze_backbone: Whether to freeze the ViT backbone
        """
        super().__init__()
        
        # Load pretrained ViT
        self.vit = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,  # Remove classification head
            global_pool=''  # Keep all tokens
        )
        
        # Get the actual hidden dimension from the model
        self.hidden_dim = self.vit.embed_dim
        self.output_dim = output_dim
        
        # Freeze backbone if specified
        if freeze_backbone:
            for param in self.vit.parameters():
                param.requires_grad = False
        
        # Projection layers
        # Project patch tokens to output_dim (Eq. 1)
        self.patch_proj = nn.Sequential(
            nn.Linear(self.hidden_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.Dropout(dropout)
        )
        
        # Project global (CLS) token to output_dim (Eq. 2)
        self.global_proj = nn.Sequential(
            nn.Linear(self.hidden_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.Dropout(dropout)
        )
        
        # Optional: Add adapter layers for fine-tuning
        self.use_adapter = False
        if self.use_adapter:
            self.adapter = nn.Sequential(
                nn.Linear(output_dim, output_dim // 4),
                nn.ReLU(),
                nn.Linear(output_dim // 4, output_dim)
            )
    
    def forward(self, images: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract patch-level and global features from images.
        
        Args:
            images: Batch of images, shape (B, 3, H, W)
            
        Returns:
            patch_features: Patch-level features, shape (B, N, output_dim)
            global_features: Global image features, shape (B, output_dim)
        """
        # Extract features from ViT
        # Output shape: (B, num_patches + 1, hidden_dim)
        features = self.vit.forward_features(images)
        
        # Separate CLS token and patch tokens
        cls_token = features[:, 0]  # (B, hidden_dim)
        patch_tokens = features[:, 1:]  # (B, num_patches, hidden_dim)
        
        # Project to output dimension
        # Equation (1): F_vis = {f_1, f_2, ..., f_N} ∈ R^{N×D}
        patch_features = self.patch_proj(patch_tokens)  # (B, N, output_dim)
        
        # Equation (2): g_vis = CLS(F_vis) ∈ R^D
        global_features = self.global_proj(cls_token)  # (B, output_dim)
        
        # Apply adapter if enabled
        if self.use_adapter:
            patch_features = patch_features + self.adapter(patch_features)
            global_features = global_features + self.adapter(global_features)
        
        return patch_features, global_features
    
    def get_attention_maps(self, images: torch.Tensor) -> torch.Tensor:
        """
        Extract attention maps from the last layer of ViT.
        
        Args:
            images: Batch of images, shape (B, 3, H, W)
            
        Returns:
            attention_maps: Attention maps, shape (B, num_heads, num_patches+1, num_patches+1)
        """
        # This requires accessing internal ViT layers
        # Implementation depends on specific ViT architecture
        with torch.no_grad():
            # Forward pass through ViT blocks
            x = self.vit.patch_embed(images)
            cls_token = self.vit.cls_token.expand(x.shape[0], -1, -1)
            x = torch.cat((cls_token, x), dim=1)
            x = self.vit.pos_drop(x + self.vit.pos_embed)
            
            # Get attention from last block
            for i, blk in enumerate(self.vit.blocks):
                if i == len(self.vit.blocks) - 1:
                    # Hook to get attention weights
                    B, N, C = x.shape
                    qkv = blk.attn.qkv(blk.norm1(x)).reshape(B, N, 3, blk.attn.num_heads, C // blk.attn.num_heads).permute(2, 0, 3, 1, 4)
                    q, k, v = qkv[0], qkv[1], qkv[2]
                    
                    attn = (q @ k.transpose(-2, -1)) * blk.attn.scale
                    attn = attn.softmax(dim=-1)
                    return attn
                else:
                    x = blk(x)
        
        return None
    
    def freeze_backbone(self):
        """Freeze the ViT backbone parameters."""
        for param in self.vit.parameters():
            param.requires_grad = False
    
    def unfreeze_backbone(self):
        """Unfreeze the ViT backbone parameters."""
        for param in self.vit.parameters():
            param.requires_grad = True
    
    def get_num_patches(self) -> int:
        """Get the number of patches for the current image size."""
        # For ViT-B/16 with 224x224 images: (224/16)^2 = 196 patches
        return self.vit.patch_embed.num_patches