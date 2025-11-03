import torch
import torch.nn.functional as F
from typing import Dict, Tuple, Optional
import numpy as np


def memory_retrieval(
    query: torch.Tensor,
    memory_bank: torch.Tensor,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    similarity_metric: str = 'cosine'
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Retrieve relevant memory entries based on query.
    
    Args:
        query: Query vectors, shape (B, D)
        memory_bank: Memory bank, shape (K, D)
        temperature: Temperature for softmax
        top_k: Return only top-k entries
        similarity_metric: Similarity metric ('cosine', 'euclidean', 'dot')
        
    Returns:
        retrieved: Retrieved memory vectors, shape (B, D) or (B, k, D)
        weights: Attention weights, shape (B, K) or (B, k)
    """
    B, D = query.shape
    K, _ = memory_bank.shape
    
    # Compute similarity scores
    if similarity_metric == 'cosine':
        # Normalize vectors
        query_norm = F.normalize(query, p=2, dim=1)
        memory_norm = F.normalize(memory_bank, p=2, dim=1)
        scores = torch.matmul(query_norm, memory_norm.t())  # (B, K)
    elif similarity_metric == 'euclidean':
        # Negative euclidean distance
        scores = -torch.cdist(query, memory_bank, p=2)  # (B, K)
    elif similarity_metric == 'dot':
        scores = torch.matmul(query, memory_bank.t())  # (B, K)
    else:
        raise ValueError(f"Unknown similarity metric: {similarity_metric}")
    
    # Apply temperature scaling
    scores = scores / temperature
    
    # Get top-k if specified
    if top_k is not None and top_k < K:
        top_scores, top_indices = torch.topk(scores, k=top_k, dim=1)
        # Compute attention weights only for top-k
        weights = F.softmax(top_scores, dim=1)  # (B, k)
        
        # Gather top-k memory entries
        top_memory = memory_bank[top_indices]  # (B, k, D)
        
        # Weighted sum
        retrieved = torch.bmm(weights.unsqueeze(1), top_memory).squeeze(1)  # (B, D)
        
        return retrieved, weights
    else:
        # Use all memory entries
        weights = F.softmax(scores, dim=1)  # (B, K)
        retrieved = torch.matmul(weights, memory_bank)  # (B, D)
        
        return retrieved, weights


def memory_update(
    memory_bank: torch.Tensor,
    updates: torch.Tensor,
    indices: Optional[torch.Tensor] = None,
    momentum: float = 0.9,
    update_strategy: str = 'momentum'
) -> torch.Tensor:
    """
    Update memory bank with new information.
    
    Args:
        memory_bank: Current memory bank, shape (K, D)
        updates: Update vectors, shape (B, D)
        indices: Indices to update, shape (B,)
        momentum: Momentum factor for updates
        update_strategy: Update strategy ('momentum', 'replace', 'add')
        
    Returns:
        updated_memory: Updated memory bank
    """
    K, D = memory_bank.shape
    B = updates.shape[0]
    
    # Clone memory to avoid in-place operations
    updated_memory = memory_bank.clone()
    
    if indices is None:
        # Update the first B entries
        indices = torch.arange(B, device=memory_bank.device)
    
    # Ensure indices are valid
    indices = indices % K
    
    if update_strategy == 'momentum':
        # Momentum-based update: m_new = momentum * m_old + (1 - momentum) * update
        updated_memory[indices] = momentum * memory_bank[indices] + (1 - momentum) * updates
    elif update_strategy == 'replace':
        # Direct replacement
        updated_memory[indices] = updates
    elif update_strategy == 'add':
        # Additive update
        updated_memory[indices] = memory_bank[indices] + updates
    else:
        raise ValueError(f"Unknown update strategy: {update_strategy}")
    
    return updated_memory


def get_memory_statistics(
    memory_bank: torch.Tensor,
    return_similarity_matrix: bool = False
) -> Dict[str, float]:
    """
    Compute statistics about the memory bank.
    
    Args:
        memory_bank: Memory bank, shape (K, D)
        return_similarity_matrix: Whether to return full similarity matrix
        
    Returns:
        Dictionary with memory statistics
    """
    K, D = memory_bank.shape
    
    # Compute norms
    memory_norms = torch.norm(memory_bank, dim=1)
    
    # Compute pairwise similarities
    memory_normalized = F.normalize(memory_bank, p=2, dim=1)
    similarity_matrix = torch.matmul(memory_normalized, memory_normalized.t())
    
    # Remove diagonal for statistics
    mask = ~torch.eye(K, dtype=torch.bool, device=memory_bank.device)
    off_diagonal = similarity_matrix[mask]
    
    stats = {
        'num_entries': K,
        'dimension': D,
        'mean_norm': memory_norms.mean().item(),
        'std_norm': memory_norms.std().item(),
        'min_norm': memory_norms.min().item(),
        'max_norm': memory_norms.max().item(),
        'mean_similarity': off_diagonal.mean().item(),
        'std_similarity': off_diagonal.std().item(),
        'max_similarity': off_diagonal.max().item(),
        'min_similarity': off_diagonal.min().item(),
    }
    
    # Compute diversity score (1 - mean similarity)
    stats['diversity_score'] = 1.0 - stats['mean_similarity']
    
    # Find most similar pairs
    if K > 1:
        similarity_matrix_masked = similarity_matrix.clone()
        similarity_matrix_masked.fill_diagonal_(-float('inf'))
        max_sim_val, max_sim_idx = similarity_matrix_masked.max(dim=1)
        
        stats['most_similar_pairs'] = [
            (i, max_sim_idx[i].item(), max_sim_val[i].item())
            for i in range(min(5, K))  # Top 5 most similar pairs
        ]
    
    if return_similarity_matrix:
        stats['similarity_matrix'] = similarity_matrix
    
    return stats


def initialize_memory_bank(
    num_entries: int,
    dimension: int,
    initialization: str = 'gaussian',
    scale: float = 1.0
) -> torch.Tensor:
    """
    Initialize a memory bank with different strategies.
    
    Args:
        num_entries: Number of memory entries (K)
        dimension: Dimension of each entry (D)
        initialization: Initialization strategy
        scale: Scaling factor
        
    Returns:
        memory_bank: Initialized memory bank, shape (K, D)
    """
    if initialization == 'gaussian':
        # Standard Gaussian initialization
        memory_bank = torch.randn(num_entries, dimension) * scale / np.sqrt(dimension)
    elif initialization == 'uniform':
        # Uniform initialization
        memory_bank = torch.rand(num_entries, dimension) * 2 - 1  # [-1, 1]
        memory_bank = memory_bank * scale / np.sqrt(dimension)
    elif initialization == 'orthogonal':
        # Orthogonal initialization (if K <= D)
        if num_entries <= dimension:
            # Use QR decomposition
            random_matrix = torch.randn(dimension, num_entries)
            q, _ = torch.qr(random_matrix)
            memory_bank = q.t()[:num_entries] * scale
        else:
            # Fall back to Gaussian
            memory_bank = torch.randn(num_entries, dimension) * scale / np.sqrt(dimension)
    elif initialization == 'xavier':
        # Xavier/Glorot initialization
        limit = np.sqrt(6.0 / (num_entries + dimension))
        memory_bank = torch.rand(num_entries, dimension) * 2 * limit - limit
        memory_bank = memory_bank * scale
    else:
        raise ValueError(f"Unknown initialization: {initialization}")
    
    return memory_bank


def memory_pruning(
    memory_bank: torch.Tensor,
    usage_counts: torch.Tensor,
    prune_ratio: float = 0.1,
    min_usage: int = 5
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Prune least used memory entries.
    
    Args:
        memory_bank: Current memory bank, shape (K, D)
        usage_counts: Usage count for each entry, shape (K,)
        prune_ratio: Ratio of entries to prune
        min_usage: Minimum usage count to keep
        
    Returns:
        pruned_memory: Pruned memory bank
        kept_indices: Indices of kept entries
    """
    K = memory_bank.shape[0]
    
    # Find entries to keep
    keep_mask = usage_counts >= min_usage
    
    # If too many entries would be pruned, keep top entries by usage
    if keep_mask.sum() < K * (1 - prune_ratio):
        num_keep = int(K * (1 - prune_ratio))
        _, top_indices = torch.topk(usage_counts, k=num_keep)
        keep_mask = torch.zeros_like(usage_counts, dtype=torch.bool)
        keep_mask[top_indices] = True
    
    # Extract kept entries
    kept_indices = torch.where(keep_mask)[0]
    pruned_memory = memory_bank[kept_indices]
    
    return pruned_memory, kept_indices