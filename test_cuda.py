import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import random
import numpy as np
from transformers import BertTokenizer, BertModel
from torchvision import models
from torch.optim import Adam

###############################################
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
        """
        grad_memory = torch.autograd.grad(loss, self.memory, retain_graph=True)[0]
        self.delta = lr * beta * self.delta - lr * grad_memory
        new_memory = (1 - alpha) * self.memory + self.delta
        self.memory.data.copy_(new_memory.data)

###############################################
# Visual Encoder using ViT Backbone
###############################################

class VisualEncoderViT(nn.Module):
    def __init__(self, output_dim=1024):
        """
        Uses a Vision Transformer (ViT) to extract patch embeddings.
        Loads a pretrained vit_b_16 from torchvision and projects both patch tokens
        and the class token to output_dim.
        """
        super(VisualEncoderViT, self).__init__()
        # Load pretrained ViT (vit_b_16) from torchvision.
        self.vit = models.vit_b_16(pretrained=True)
        # Instead of forward_features (deprecated), we use the internal methods.
        # Project patch tokens (original hidden dim is 768) to output_dim.
        self.patch_proj = nn.Linear(self.vit.hidden_dim, output_dim)
        # Also project the class token (global representation).
        self.global_proj = nn.Linear(self.vit.hidden_dim, output_dim)
        self.layernorm = nn.LayerNorm(output_dim)
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, images):
        """
        Args:
            images: (B, 3, 224, 224)
        Returns:
            patch_tokens_proj: (B, num_patches, output_dim)
            global_embedding: (B, output_dim)
        """
        # Manually process images through the ViT encoder:
        x = self.vit._process_input(images)  # Preprocess input (B, 3, 224, 224) -> (B, num_patches+1, hidden_dim)
        x = self.vit.encoder(x)              # (B, 1+num_patches, hidden_dim)
        class_token = x[:, 0]                # (B, hidden_dim)
        patch_tokens = x[:, 1:]              # (B, num_patches, hidden_dim)
        # Project patch tokens.
        patch_tokens_proj = self.patch_proj(patch_tokens)  # (B, num_patches, output_dim)
        patch_tokens_proj = self.layernorm(patch_tokens_proj)
        patch_tokens_proj = self.dropout(patch_tokens_proj)
        # Global image embedding from the class token.
        global_embedding = self.global_proj(class_token)   # (B, output_dim)
        global_embedding = self.layernorm(global_embedding)
        global_embedding = self.dropout(global_embedding)
        return patch_tokens_proj, global_embedding

###############################################
# Text Encoder using ClinicalBERT
###############################################

class TextEncoder(nn.Module):
    def __init__(self, aspect_dim=1024):
        """
        Encodes text descriptions (e.g., radiology reports) into embeddings.
        Projects ClinicalBERT's pooled output (768-dim) to aspect_dim.
        """
        super(TextEncoder, self).__init__()
        self.tokenizer = BertTokenizer.from_pretrained('emilyalsentzer/Bio_ClinicalBERT')
        self.bert = BertModel.from_pretrained('emilyalsentzer/Bio_ClinicalBERT')
        self.aspect_projection = nn.Linear(self.bert.config.hidden_size, aspect_dim)
        self.layernorm = nn.LayerNorm(aspect_dim)
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, texts):
        """
        Args:
            texts (list of str): Text descriptions.
        Returns:
            aspect_embeddings (Tensor): (B, aspect_dim)
        """
        inputs = self.tokenizer(texts, return_tensors='pt', padding=True, truncation=True, max_length=512)
        for key in inputs:
            inputs[key] = inputs[key].to(next(self.parameters()).device)
        outputs = self.bert(**inputs)
        pooled_output = outputs.pooler_output
        aspect_embeddings = self.aspect_projection(pooled_output)
        aspect_embeddings = self.layernorm(aspect_embeddings)
        aspect_embeddings = self.dropout(aspect_embeddings)
        return aspect_embeddings

###############################################
# Dual-Head Vision-Language Model
###############################################

class DualHeadModel(nn.Module):
    def __init__(self, vision_dim=1024, text_dim=1024, num_heads=8, num_classes=10):
        """
        Two branches:
          (A) Contrastive Head: Aligns the global image embedding with the text embedding.
          (B) Supervised Head: Uses cross-attention between learned aspect queries and spatial features,
              then classifies using a fully connected network (with LayerNorm and Dropout).
          Optionally integrates a NeuralMemory module.
        """
        super(DualHeadModel, self).__init__()
        self.contrastive_head = nn.MultiheadAttention(embed_dim=vision_dim, num_heads=num_heads, batch_first=True)
        self.cross_attention = nn.MultiheadAttention(embed_dim=vision_dim, num_heads=num_heads, batch_first=True)
        self.fc = nn.Sequential(
            nn.LayerNorm(vision_dim),
            nn.Dropout(0.5),
            nn.Linear(vision_dim, num_classes)
        )
        self.vision_projector = nn.Linear(vision_dim, text_dim)
        self.text_projector = nn.Linear(text_dim, vision_dim)

    def forward(self, vision_features, text_embeddings, aspect_queries, neural_memory=None):
        """
        Args:
            vision_features (Tensor): (B, N, 1024) patch features.
            text_embeddings (Tensor): (B, 1024) text embeddings.
            aspect_queries (Tensor): (num_aspects, B, 1024) learned queries.
            neural_memory (NeuralMemory, optional): For test-time adaptation.
        Returns:
            contrastive_logits (Tensor): (B,) similarity scores.
            supervised_logits (Tensor): (B, num_classes) classification scores.
            attn_weights (Tensor): Attention weights from cross-attention.
        """
        B, N, C = vision_features.shape
        vision_global = vision_features.mean(dim=1)
        if neural_memory is not None:
            retrieved = neural_memory.retrieve(vision_global)
            vision_global = vision_global + retrieved
        
        # Contrastive Head.
        vision_global_seq = vision_global.unsqueeze(1)      # (B, 1, 1024)
        text_embeddings_seq = text_embeddings.unsqueeze(1)  # (B, 1, 1024)
        contrastive_out, _ = self.contrastive_head(vision_global_seq, text_embeddings_seq, text_embeddings_seq)
        contrastive_out_norm = F.normalize(contrastive_out, p=2, dim=-1)
        text_embeddings_seq_norm = F.normalize(text_embeddings_seq, p=2, dim=-1)
        contrastive_logits = torch.bmm(contrastive_out_norm, text_embeddings_seq_norm.transpose(1,2))
        contrastive_logits = contrastive_logits.squeeze(1).squeeze(-1)
        
        # Supervised Head.
        aspect_queries = aspect_queries.permute(1, 0, 2)  # (B, num_aspects, 1024)
        attended_features, attn_weights = self.cross_attention(aspect_queries, vision_features, vision_features)
        supervised_representation = attended_features.mean(dim=1)
        supervised_logits = self.fc(supervised_representation)
        
        return contrastive_logits, supervised_logits, attn_weights

###############################################
# Dummy Data Generation
###############################################

def get_dummy_data(batch_size, num_classes, device, train=True):
    """
    Generates dummy data:
      - Images: constant pattern (scaled by label) + Gaussian noise.
      - Texts: descriptive string with disease label and aspects.
      - Labels: integers in [0, num_classes-1].
      - Aspect queries: random tensor simulating learned queries.
    Returns:
        images: (B, 3, 224, 224)
        texts: list of strings (length B)
        labels: (B,)
        aspect_queries: (num_aspects, B, 1024)
    """
    images = []
    texts = []
    labels = torch.randint(0, num_classes, (batch_size,), device=device)
    for label in labels:
        base = torch.ones(3, 224, 224, device=device) * (label.item() / num_classes)
        noise = torch.randn(3, 224, 224, device=device) * 0.1
        img = base + noise
        images.append(img.unsqueeze(0))
        text = (f"Disease label {label.item()}: Aspect1-high; Aspect2-low; "
                f"Aspect3-moderate; Aspect4-severe; Extra {random.random():.2f}")
        texts.append(text)
    images = torch.cat(images, dim=0)
    num_aspects = 7
    aspect_queries = torch.randn(num_aspects, batch_size, 1024, device=device)
    return images, texts, labels, aspect_queries

###############################################
# Training and Evaluation Functions (Optimized)
###############################################

def train_model(num_epochs=2, num_iterations=20):
    """
    Trains the vision-language model on dummy data.
    Uses AMP (if available) for faster training.
    Returns trained model components.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 8
    num_classes = 10

    # Initialize modules (using ViT-based visual encoder).
    visual_encoder = VisualEncoderViT(output_dim=1024).to(device)
    text_encoder = TextEncoder(aspect_dim=1024).to(device)
    dual_head = DualHeadModel(num_classes=num_classes).to(device)
    neural_memory = NeuralMemory(memory_dim=1024, num_entries=10).to(device)
    
    visual_encoder.train()
    text_encoder.train()
    dual_head.train()
    neural_memory.train()
    
    optimizer = Adam(list(visual_encoder.parameters()) + 
                     list(text_encoder.parameters()) + 
                     list(dual_head.parameters()) +
                     list(neural_memory.parameters()), lr=1e-4)
    
    criterion_supervised = nn.CrossEntropyLoss()
    criterion_contrastive = nn.MSELoss()
    
    scaler = torch.amp.GradScaler(device_type='cuda') if device.type == 'cuda' else None

    for epoch in range(num_epochs):
        for iteration in range(num_iterations):
            optimizer.zero_grad()
            images, texts, labels, aspect_queries = get_dummy_data(batch_size, num_classes, device, train=True)
            if scaler is not None:
                with torch.amp.autocast(device_type='cuda'):
                    patch_tokens, global_embedding = visual_encoder(images)
                    text_embeddings = text_encoder(texts)
                    contrastive_logits, supervised_logits, attn_weights = dual_head(
                        patch_tokens, text_embeddings, aspect_queries, neural_memory=neural_memory)
                    loss_sup = criterion_supervised(supervised_logits, labels)
                    target_contr = torch.ones_like(contrastive_logits, device=device)
                    loss_contr = criterion_contrastive(contrastive_logits, target_contr)
                    loss_total = 0.1 * loss_contr + 1.0 * loss_sup
                scaler.scale(loss_total).backward()
                torch.nn.utils.clip_grad_norm_(list(visual_encoder.parameters()) +
                                               list(text_encoder.parameters()) +
                                               list(dual_head.parameters()) +
                                               list(neural_memory.parameters()), max_norm=5.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                patch_tokens, global_embedding = visual_encoder(images)
                text_embeddings = text_encoder(texts)
                contrastive_logits, supervised_logits, attn_weights = dual_head(
                    patch_tokens, text_embeddings, aspect_queries, neural_memory=neural_memory)
                loss_sup = criterion_supervised(supervised_logits, labels)
                target_contr = torch.ones_like(contrastive_logits, device=device)
                loss_contr = criterion_contrastive(contrastive_logits, target_contr)
                loss_total = 0.1 * loss_contr + 1.0 * loss_sup
                loss_total.backward()
                torch.nn.utils.clip_grad_norm_(list(visual_encoder.parameters()) +
                                               list(text_encoder.parameters()) +
                                               list(dual_head.parameters()) +
                                               list(neural_memory.parameters()), max_norm=5.0)
                optimizer.step()
            # Update neural memory after the backward pass.
            neural_memory.update_memory(loss_total, lr=0.01, beta=0.9, alpha=0.1)
            print(f"Epoch {epoch+1}, Iter {iteration+1}, Total Loss: {loss_total.item():.4f}, "
                  f"Sup Loss: {loss_sup.item():.4f}, Contr Loss: {loss_contr.item():.4f}")
    print("Training completed.\n")
    return visual_encoder, text_encoder, dual_head, neural_memory

def test_model(visual_encoder, text_encoder, dual_head, neural_memory):
    """
    Evaluates the trained model on dummy test data.
    Uses torch.no_grad() for faster evaluation.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 4
    num_classes = 10
    visual_encoder.eval()
    text_encoder.eval()
    dual_head.eval()
    neural_memory.eval()
    with torch.no_grad():
        images, texts, labels, aspect_queries = get_dummy_data(batch_size, num_classes, device, train=False)
        patch_tokens, global_embedding = visual_encoder(images)
        text_embeddings = text_encoder(texts)
        contrastive_logits, supervised_logits, attn_weights = dual_head(
            patch_tokens, text_embeddings, aspect_queries, neural_memory=neural_memory)
        _, preds = torch.max(supervised_logits, dim=1)
        print("Test Results:")
        print("True labels:      ", labels.cpu().numpy())
        print("Predicted labels: ", preds.cpu().numpy())
        print("Contrastive logits:\n", contrastive_logits.cpu().numpy())
        print("Supervised logits shape:", supervised_logits.shape)

###############################################
# Main Execution
###############################################

def main():
    visual_encoder, text_encoder, dual_head, neural_memory = train_model(num_epochs=2, num_iterations=20)
    test_model(visual_encoder, text_encoder, dual_head, neural_memory)

if __name__ == "__main__":
    main()
