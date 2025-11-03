import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class NeuralMemory(nn.Module):
    """
    Neural Memory Module for test-time adaptation.
    Implements the memory mechanism described in Section 3.6 of the paper.
    """
    
    def __init__(
        self,
        memory_size: int = 100,
        memory_dim: int = 1024,
        learning_rate: float = 0.01,
        momentum_beta: float = 0.9,
        update_alpha: float = 0.1,
        use_persistent_memory: bool = True
    ):
        """
        Args:
            memory_size: Number of memory slots (K)
            memory_dim: Dimension of each memory slot (D)
            learning_rate: Learning rate for memory updates (η)
            momentum_beta: Momentum parameter (β)
            update_alpha: Update strength parameter (α)
            use_persistent_memory: Whether to use persistent memory
        """
        super().__init__()
        
        self.memory_size = memory_size
        self.memory_dim = memory_dim
        self.learning_rate = learning_rate
        self.momentum_beta = momentum_beta
        self.update_alpha = update_alpha
        
        # Initialize memory matrix M ∈ R^{K×D}
        self.memory = nn.Parameter(
            torch.randn(memory_size, memory_dim) / torch.sqrt(torch.tensor(memory_dim, dtype=torch.float))
        )
        
        # Persistent memory for long-term storage (optional)
        if use_persistent_memory:
            self.persistent_memory = nn.Parameter(
                torch.randn(memory_size, memory_dim) / torch.sqrt(torch.tensor(memory_dim, dtype=torch.float))
            )
        else:
            self.register_buffer('persistent_memory', None)
        
        # Momentum buffer for updates (Δ_t in Eq. 9)
        self.register_buffer('delta', torch.zeros(memory_size, memory_dim))
        
        # Optional: Learnable temperature for attention
        self.temperature = nn.Parameter(torch.tensor(1.0))
        
        # Optional: Memory value projection
        self.value_projection = nn.Linear(memory_dim, memory_dim)
        
        # Track if we're in training or evaluation mode for different update strategies
        self.update_in_eval = False
    
    def retrieve(
        self,
        query: torch.Tensor,
        return_weights: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Retrieve information from memory using attention mechanism.
        Implements Equation (8) from the paper.
        
        Args:
            query: Query vector (typically global image embedding), shape (B, memory_dim)
            return_weights: Whether to return attention weights
            
        Returns:
            retrieved: Retrieved memory vector, shape (B, memory_dim)
            attention_weights: Attention weights if requested, shape (B, memory_size)
        """
        B = query.shape[0]
        
        # Compute attention scores
        # α = softmax(qM^T) from Eq. 8
        scores = torch.matmul(query, self.memory.t()) / self.temperature  # (B, K)
        attention_weights = F.softmax(scores, dim=-1)  # (B, K)
        
        # Retrieve weighted memory
        # r = αM from Eq. 8
        retrieved = torch.matmul(attention_weights, self.memory)  # (B, D)
        
        # Optional: Apply value projection for better adaptation
        retrieved = self.value_projection(retrieved)
        
        # Add persistent memory if available
        if self.persistent_memory is not None:
            persistent_scores = torch.matmul(query, self.persistent_memory.t()) / self.temperature
            persistent_weights = F.softmax(persistent_scores, dim=-1)
            persistent_retrieved = torch.matmul(persistent_weights, self.persistent_memory)
            
            # Combine adaptive and persistent memory
            retrieved = (retrieved + persistent_retrieved) / 2.0
        
        if return_weights:
            return retrieved, attention_weights
        else:
            return retrieved, None
    
    def update_memory(
        self,
        loss: torch.Tensor,
        retain_graph: bool = True
    ):
        """
        Update memory using gradient-based adaptation.
        Implements Equation (9) from the paper.
        
        This should be called manually before the main backward pass during training.
        
        Args:
            loss: Loss value to compute gradients
            retain_graph: Whether to retain computation graph
        """
        if not self.training and not self.update_in_eval:
            return
        
        # Compute gradient with respect to memory
        # ∇_M L from Eq. 9
        grad_memory = torch.autograd.grad(
            loss,
            self.memory,
            retain_graph=retain_graph,
            create_graph=False
        )[0]
        
        # Update momentum buffer
        # Δ_t = η β Δ_{t-1} - η ∇_M L from Eq. 9
        self.delta = self.learning_rate * self.momentum_beta * self.delta - self.learning_rate * grad_memory
        
        # Update memory
        # M_t = (1 - α) M_{t-1} + Δ_t from Eq. 9
        new_memory = (1 - self.update_alpha) * self.memory + self.delta
        
        # Update memory in-place
        self.memory.data.copy_(new_memory.data)
    
    def adapt(
        self,
        query: torch.Tensor,
        neural_memory: Optional['NeuralMemory'] = None
    ) -> torch.Tensor:
        """
        Adapt query vector using retrieved memory.
        Implements q' = q + r from Eq. 8.
        
        Args:
            query: Query vector, shape (B, memory_dim)
            neural_memory: Optional external memory module
            
        Returns:
            adapted_query: Adapted query vector, shape (B, memory_dim)
        """
        # Use provided memory or self
        memory_module = neural_memory if neural_memory is not None else self
        
        # Retrieve from memory
        retrieved, _ = memory_module.retrieve(query)
        
        # Adapt query
        # q' = q + r from Eq. 8
        adapted_query = query + retrieved
        
        return adapted_query
    
    def reset_momentum(self):
        """Reset momentum buffer to zeros."""
        self.delta.zero_()
    
    def enable_eval_updates(self):
        """Enable memory updates during evaluation (test-time adaptation)."""
        self.update_in_eval = True
    
    def disable_eval_updates(self):
        """Disable memory updates during evaluation."""
        self.update_in_eval = False
    
    def get_memory_statistics(self) -> dict:
        """Get statistics about memory usage."""
        with torch.no_grad():
            # Memory norms
            memory_norms = torch.norm(self.memory, dim=1)
            
            # Memory similarity matrix
            memory_similarity = torch.matmul(
                F.normalize(self.memory, dim=1),
                F.normalize(self.memory, dim=1).t()
            )
            
            # Remove diagonal for similarity stats
            mask = ~torch.eye(self.memory_size, dtype=torch.bool, device=self.memory.device)
            similarity_values = memory_similarity[mask]
            
            stats = {
                'memory_norm_mean': memory_norms.mean().item(),
                'memory_norm_std': memory_norms.std().item(),
                'memory_similarity_mean': similarity_values.mean().item(),
                'memory_similarity_max': similarity_values.max().item(),
                'delta_norm': torch.norm(self.delta).item(),
                'temperature': self.temperature.item()
            }
            
            return stats