import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Dict
from transformers import AutoTokenizer, AutoModel


class TextEncoder(nn.Module):
    """
    ClinicalBERT-based text encoder for encoding radiology reports.
    Based on Equation (3) in the paper.
    """
    
    def __init__(
        self,
        model_name: str = "emilyalsentzer/Bio_ClinicalBERT",
        hidden_dim: int = 768,
        output_dim: int = 1024,
        max_length: int = 512,
        dropout: float = 0.1,
        freeze_backbone: bool = False,
        use_pooler: bool = True
    ):
        """
        Args:
            model_name: Name of the pretrained BERT model
            hidden_dim: Hidden dimension of BERT (768 for base)
            output_dim: Output dimension for projection
            max_length: Maximum sequence length
            dropout: Dropout rate
            freeze_backbone: Whether to freeze BERT parameters
            use_pooler: Whether to use pooler output or CLS token
        """
        super().__init__()
        
        # Load pretrained ClinicalBERT
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.bert = AutoModel.from_pretrained(model_name)
        
        self.hidden_dim = self.bert.config.hidden_size
        self.output_dim = output_dim
        self.max_length = max_length
        self.use_pooler = use_pooler
        
        # Freeze backbone if specified
        if freeze_backbone:
            for param in self.bert.parameters():
                param.requires_grad = False
        
        # Projection layer for aspect embeddings
        # Maps BERT output to desired dimension (Eq. 3)
        self.aspect_projection = nn.Sequential(
            nn.Linear(self.hidden_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.Dropout(dropout)
        )
        
        # Optional: Additional projection for better alignment
        self.use_mlp_projection = True
        if self.use_mlp_projection:
            self.mlp_projection = nn.Sequential(
                nn.Linear(output_dim, output_dim * 2),
                nn.ReLU(),
                nn.Linear(output_dim * 2, output_dim),
                nn.LayerNorm(output_dim),
                nn.Dropout(dropout)
            )
    
    def forward(self, texts: List[str]) -> torch.Tensor:
        """
        Encode text descriptions into embeddings.
        
        Args:
            texts: List of text descriptions (radiology reports)
            
        Returns:
            text_embeddings: Text embeddings, shape (B, output_dim)
        """
        # Tokenize texts
        inputs = self.tokenizer(
            texts,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=self.max_length
        )
        
        # Move to same device as model
        device = next(self.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Forward through BERT
        outputs = self.bert(**inputs)
        
        # Get text representation
        if self.use_pooler and hasattr(outputs, 'pooler_output'):
            # Use pooler output (recommended for sentence-level tasks)
            text_features = outputs.pooler_output  # (B, hidden_dim)
        else:
            # Use CLS token
            text_features = outputs.last_hidden_state[:, 0]  # (B, hidden_dim)
        
        # Project to output dimension
        # Equation (3): E_text = f_text(T) ∈ R^D
        text_embeddings = self.aspect_projection(text_features)  # (B, output_dim)
        
        # Apply additional MLP projection if enabled
        if self.use_mlp_projection:
            text_embeddings = self.mlp_projection(text_embeddings)
        
        return text_embeddings
    
    def encode_batch(
        self,
        texts: List[str],
        batch_size: int = 32
    ) -> torch.Tensor:
        """
        Encode a large batch of texts in smaller chunks to avoid OOM.
        
        Args:
            texts: List of text descriptions
            batch_size: Size of each processing batch
            
        Returns:
            text_embeddings: Combined text embeddings
        """
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            with torch.no_grad():
                embeddings = self.forward(batch_texts)
            all_embeddings.append(embeddings)
        
        return torch.cat(all_embeddings, dim=0)
    
    def get_token_embeddings(self, texts: List[str]) -> torch.Tensor:
        """
        Get token-level embeddings for fine-grained analysis.
        
        Args:
            texts: List of text descriptions
            
        Returns:
            token_embeddings: Token embeddings, shape (B, seq_len, hidden_dim)
        """
        # Tokenize texts
        inputs = self.tokenizer(
            texts,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=self.max_length
        )
        
        # Move to device
        device = next(self.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Forward through BERT
        outputs = self.bert(**inputs)
        
        # Return all token embeddings
        return outputs.last_hidden_state
    
    def create_disease_prompts(self, disease_names: List[str]) -> List[str]:
        """
        Create text prompts for disease categories.
        
        Args:
            disease_names: List of disease names
            
        Returns:
            prompts: List of formatted prompts
        """
        prompt_templates = [
            "Chest X-ray showing evidence of {}",
            "Radiographic findings consistent with {}",
            "Imaging demonstrates {}",
            "X-ray reveals {}",
            "Findings suggestive of {}"
        ]
        
        prompts = []
        for disease in disease_names:
            # Use different templates for variety
            template = prompt_templates[len(prompts) % len(prompt_templates)]
            prompt = template.format(disease.lower())
            prompts.append(prompt)
        
        return prompts
    
    def freeze_backbone(self):
        """Freeze BERT backbone parameters."""
        for param in self.bert.parameters():
            param.requires_grad = False
    
    def unfreeze_backbone(self):
        """Unfreeze BERT backbone parameters."""
        for param in self.bert.parameters():
            param.requires_grad = True