import torch
import numpy as np
import random
import os
from typing import Optional


def set_seed(seed: int = 42):
    """
    Set random seed for reproducibility across all libraries.
    
    Args:
        seed: Random seed value
    """
    # Python random
    random.seed(seed)
    
    # Numpy
    np.random.seed(seed)
    
    # PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU
    
    # Environment variables
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    print(f"Random seed set to {seed}")


def set_deterministic(enable: bool = True):
    """
    Enable/disable deterministic mode in PyTorch.
    
    Note: Enabling deterministic mode may impact performance.
    
    Args:
        enable: Whether to enable deterministic mode
    """
    if enable:
        # PyTorch deterministic operations
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
        # For PyTorch >= 1.8
        if hasattr(torch, 'use_deterministic_algorithms'):
            torch.use_deterministic_algorithms(True)
        
        # Set environment variable for CUBLAS
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
        
        print("Deterministic mode enabled (may impact performance)")
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
        
        if hasattr(torch, 'use_deterministic_algorithms'):
            torch.use_deterministic_algorithms(False)
        
        print("Deterministic mode disabled")


def worker_init_fn(worker_id: int, base_seed: Optional[int] = None):
    """
    Worker initialization function for DataLoader to ensure reproducibility.
    
    Args:
        worker_id: Worker ID from DataLoader
        base_seed: Base seed (if None, uses current random state)
    """
    if base_seed is None:
        base_seed = torch.initial_seed() % 2**32
    
    # Set seeds for each worker
    worker_seed = base_seed + worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_random_state() -> dict:
    """
    Get current random state from all libraries.
    
    Returns:
        Dictionary containing random states
    """
    state = {
        'python_random': random.getstate(),
        'numpy_random': np.random.get_state(),
        'torch_random': torch.get_rng_state(),
    }
    
    if torch.cuda.is_available():
        state['torch_cuda_random'] = torch.cuda.get_rng_state_all()
    
    return state


def set_random_state(state: dict):
    """
    Restore random state for all libraries.
    
    Args:
        state: Dictionary containing random states
    """
    random.setstate(state['python_random'])
    np.random.set_state(state['numpy_random'])
    torch.set_rng_state(state['torch_random'])
    
    if 'torch_cuda_random' in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state['torch_cuda_random'])


class RandomContext:
    """
    Context manager for temporary random seed setting.
    
    Usage:
        with RandomContext(seed=123):
            # Code with fixed random seed
            pass
        # Original random state restored
    """
    
    def __init__(self, seed: int):
        self.seed = seed
        self.state = None
    
    def __enter__(self):
        # Save current state
        self.state = get_random_state()
        # Set new seed
        set_seed(self.seed)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        # Restore original state
        if self.state is not None:
            set_random_state(self.state)


def ensure_reproducibility(
    seed: int = 42,
    deterministic: bool = True,
    warn_performance: bool = True
):
    """
    Ensure full reproducibility with a single function call.
    
    Args:
        seed: Random seed
        deterministic: Whether to enable deterministic mode
        warn_performance: Whether to warn about performance impact
    """
    # Set seed
    set_seed(seed)
    
    # Enable deterministic mode
    if deterministic:
        set_deterministic(True)
        
        if warn_performance:
            print("\nWARNING: Deterministic mode is enabled.")
            print("This ensures reproducibility but may significantly impact performance.")
            print("Consider disabling for final training runs.\n")
    
    # Print system info for debugging
    print("System configuration for reproducibility:")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"cuDNN version: {torch.backends.cudnn.version()}")
    print(f"Number of threads: {torch.get_num_threads()}")
    print()


def check_reproducibility(
    func,
    args=(),
    kwargs=None,
    seed: int = 42,
    num_runs: int = 3
) -> bool:
    """
    Check if a function produces reproducible results.
    
    Args:
        func: Function to test
        args: Positional arguments for function
        kwargs: Keyword arguments for function
        seed: Random seed to use
        num_runs: Number of runs to test
        
    Returns:
        True if results are reproducible
    """
    if kwargs is None:
        kwargs = {}
    
    results = []
    
    for _ in range(num_runs):
        # Set seed before each run
        set_seed(seed)
        
        # Run function
        result = func(*args, **kwargs)
        
        # Convert to numpy for comparison
        if torch.is_tensor(result):
            result = result.detach().cpu().numpy()
        
        results.append(result)
    
    # Check if all results are identical
    reproducible = True
    for i in range(1, num_runs):
        if not np.allclose(results[0], results[i], rtol=1e-5, atol=1e-8):
            reproducible = False
            break
    
    return reproducible