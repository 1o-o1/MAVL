import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertTokenizer, BertModel
from torchvision import models
import random
##############################################
# Utility Functions
###############################################

def _get_activation_fn(activation):
    """Return an activation function given a string."""
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu
    elif activation == "glu":
        return F.glu
    else:
        raise RuntimeError(f"activation should be relu/gelu, not {activation}")

def _get_clones(module, N):
    """Produce N identical layers."""
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

###############################################
# Neural Memory Module (for test-time adaptation)
###############################################

class NeuralMemory(nn.Module):
    def __init__(self, memory_dim=1024, num_entries=10):
        """
        memory_dim: Dimension of each memory entry (must match image embedding dim).
        num_entries: Number of memory entries.
        """
        super(NeuralMemory, self).__init__()
        self.memory = nn.Parameter(torch.randn(num_entries, memory_dim))
        self.persistent_memory = nn.Parameter(torch.randn(num_entries, memory_dim))
        self.register_buffer('delta', torch.zeros(num_entries, memory_dim))
    
    def retrieve(self, query):
        """
        Retrieve memory content given a query vector.
        Args:
            query (Tensor): (B, memory_dim)
        Returns:
            retrieved (Tensor): (B, memory_dim)
        """
        scores = torch.matmul(query, self.memory.t())  # (B, num_entries)
        attn = F.softmax(scores, dim=-1)
        retrieved = torch.matmul(attn, self.memory)
        return retrieved

    def update_memory(self, loss, lr, beta, alpha):
        """
        Update memory using the rule:
            Δ_t = lr * beta * Δ_(t-1) - lr * ∇ℓ(M; x)
            M_t = (1 - alpha) * M_(t-1) + Δ_t
        This update is done manually before the main backward pass.
        """
        grad_memory = torch.autograd.grad(loss, self.memory, retain_graph=True)[0]
        self.delta = lr * beta * self.delta - lr * grad_memory
        new_memory = (1 - alpha) * self.memory + self.delta
        self.memory.data.copy_(new_memory.data)

############################################################
# 2. Vision-Language Encoder with Multi-Aspect, Dual-Head Training
############################################################

# 2.2 Visual Encoder
# Backbone Architecture: Use ResNet-50 to extract a spatial feature map.
# Attention Pooling (Optional): Here, we compute a global embedding by averaging spatial features.
class VisualEncoder(nn.Module):
    def __init__(self, output_dim=1024):
        super(VisualEncoder, self).__init__()
        # Load a pretrained ResNet-50 and remove the final two layers (avgpool & fc)
        resnet = models.resnet50(pretrained=True)
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-2])  # Output: (B, 2048, H/32, W/32)
        # Downsample spatial dimensions to a fixed size (e.g., 16x16)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((16, 16))
        # Projection layer to convert 2048-dim features to 1024-dim features for alignment
        self.projection = nn.Linear(2048, output_dim)

    def forward(self, images):
        # Extract a spatial feature map V ∈ ℝ^(B×2048×H×W)
        x = self.feature_extractor(images)  # (B, 2048, H/32, W/32)
        x = self.adaptive_pool(x)             # (B, 2048, 16, 16)
        x = x.flatten(2)                      # Flatten spatial dims: (B, 2048, 256)
        x = x.permute(0, 2, 1)                # Rearrange to (B, 256, 2048)
        # Project each spatial vector to the target dimension (1024)
        x_proj = self.projection(x)           # (B, 256, 1024)
        spatial_features = x_proj             # Multi-aspect visual features
        # Global image embedding f ∈ ℝ^(B×1024) by averaging across spatial positions
        global_embedding = x_proj.mean(dim=1)
        return spatial_features, global_embedding

# 2.3 Text Encoder
# Domain-Specific Model: Use ClinicalBERT to encode textual aspect descriptions.
# Aspect Clustering: In practice, aspect embeddings would be grouped per disease (handled downstream).
class TextEncoder(nn.Module):
    def __init__(self, aspect_dim=1024):
        super(TextEncoder, self).__init__()
        self.tokenizer = BertTokenizer.from_pretrained('emilyalsentzer/Bio_ClinicalBERT')
        self.bert = BertModel.from_pretrained('emilyalsentzer/Bio_ClinicalBERT')
        # Project BERT's pooled output (768-dim) to the desired aspect embedding dimension (1024)
        self.aspect_projection = nn.Linear(self.bert.config.hidden_size, aspect_dim)

    def forward(self, texts):
        inputs = self.tokenizer(texts, return_tensors='pt', padding=True, truncation=True, max_length=512)
        outputs = self.bert(**inputs)
        pooled_output = outputs.pooler_output  # (B, 768)
        aspect_embeddings = self.aspect_projection(pooled_output)  # (B, 1024)
        return aspect_embeddings

# 2.4 Dual-Head Transformer Architecture
# (A) Contrastive Head for Zero-Shot Detection and
# (B) Supervised Head for Seen Diseases using cross-attention grounding.
class DualHeadModel(nn.Module):
    def __init__(self, vision_dim=1024, text_dim=1024, num_heads=8, num_classes=10):
        super(DualHeadModel, self).__init__()
        # Contrastive Head: Align the global image embedding with text aspect embeddings.
        self.contrastive_head = nn.MultiheadAttention(embed_dim=vision_dim, num_heads=num_heads, batch_first=True)
        # Supervised Head: Fuse spatial features with learned aspect queries via cross-attention.
        self.cross_attention = nn.MultiheadAttention(embed_dim=vision_dim, num_heads=num_heads, batch_first=True)
        self.classifier = nn.Linear(vision_dim, num_classes)
        # Optional feature projectors (for advanced alignment, not used explicitly here)
        self.vision_projector = nn.Linear(vision_dim, text_dim)
        self.text_projector = nn.Linear(text_dim, vision_dim)

    def forward(self, vision_features, text_embeddings, aspect_queries, neural_memory=None):
        """
        vision_features: (B, N, 1024) spatial features from visual encoder.
        text_embeddings: (B, 1024) text aspect embedding.
        aspect_queries: (num_aspects, B, 1024) learned queries for multi-aspect grounding.
        neural_memory: (optional) NeuralMemory module for test-time adaptation.
        """
        B, N, C = vision_features.shape
        # Compute a global image embedding by averaging spatial features.
        vision_global = vision_features.mean(dim=1)  # (B, 1024)
        
        # If test-time neural memory is provided, retrieve and integrate its output.
        if neural_memory is not None:
            retrieved = neural_memory.retrieve(vision_global)  # (B, 1024)
            # Memory as Context: Integrate memory output with the global embedding.
            vision_global = vision_global + retrieved

        # Contrastive Head: Align global image embedding with textual aspect embedding.
        # Add a sequence dimension to each embedding.
        vision_global_seq = vision_global.unsqueeze(1)  # (B, 1, 1024)
        text_embeddings_seq = text_embeddings.unsqueeze(1)  # (B, 1, 1024)
        contrastive_output, _ = self.contrastive_head(vision_global_seq, text_embeddings_seq, text_embeddings_seq)
        # Compute similarity (e.g., dot product) as logits.
        contrastive_logits = torch.bmm(contrastive_output, text_embeddings_seq.transpose(1, 2)).squeeze()  # (B,)

        # Supervised Head: Ground the spatial features using cross-attention with aspect queries.
        # Permute aspect queries from (num_aspects, B, 1024) to (B, num_aspects, 1024).
        aspect_queries = aspect_queries.permute(1, 0, 2)
        attended_features, attn_weights = self.cross_attention(aspect_queries, vision_features, vision_features)
        # Aggregate attended features to form a final representation.
        supervised_representation = attended_features.mean(dim=1)  # (B, 1024)
        supervised_logits = self.classifier(supervised_representation)  # (B, num_classes)

        # Return contrastive logits, supervised logits, and attention weights (e.g., for interpretability)
        return contrastive_logits, supervised_logits, attn_weights

############################################################
# 3. Integrating Test-Time Neural Memory Inspired by Titans
############################################################

class NeuralMemory(nn.Module):
    def __init__(self, memory_dim=1024, num_entries=10):
        """
        memory_dim: Dimension of each memory entry (should match the image embedding dimension).
        num_entries: Number of memory entries.
        """
        super(NeuralMemory, self).__init__()
        # Initialize memory parameters M ∈ ℝ^(num_entries x memory_dim) as learnable parameters.
        self.memory = nn.Parameter(torch.randn(num_entries, memory_dim))
        # Persistent memory vectors P serve as a global knowledge base.
        self.persistent_memory = nn.Parameter(torch.randn(num_entries, memory_dim))
        # Momentum term for updates, stored as a buffer (non-learnable)
        self.register_buffer('delta', torch.zeros(num_entries, memory_dim))
    
    def retrieve(self, query):
        """
        Retrieve memory content given a query vector.
        query: (B, memory_dim)
        Returns: retrieved memory content of shape (B, memory_dim)
        """
        # Compute similarity scores between query and each memory entry.
        scores = torch.matmul(query, self.memory.t())  # (B, num_entries)
        attn = F.softmax(scores, dim=-1)                # (B, num_entries)
        # Compute a weighted sum of memory entries.
        retrieved = torch.matmul(attn, self.memory)       # (B, memory_dim)
        return retrieved
    
    def update_memory(self, loss, lr, beta, alpha):
        """
        Update the memory using the surprise metric and momentum update.
        loss: Scalar loss computed from the current input.
        lr: Learning rate (ε_t).
        beta: Momentum coefficient (β_t).
        alpha: Forgetting rate (α_t).
        """
        # Compute gradient of the loss with respect to the memory parameters.
        grad_memory = torch.autograd.grad(loss, self.memory, retain_graph=True)[0]
        # Memory Update with Momentum: Δ_t = ε_t * β_t * Δ_(t-1) - ε_t * ∇ℓ(M_(t-1); x_t)
        self.delta = lr * beta * self.delta - lr * grad_memory
        # Forgetting Mechanism: M_t = (1 - α_t) * M_(t-1) + Δ_t
        new_memory = (1 - alpha) * self.memory + self.delta
        self.memory.data.copy_(new_memory.data)



