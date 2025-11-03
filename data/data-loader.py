import numpy as np
from typing import Dict, List, Tuple, Optional
import torch
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
import random

from .dataset import ChestXrayDataset, MultiDatasetWrapper
from .preprocessing import get_transforms


def split_datasets_with_unseen(
    dataset: ChestXrayDataset,
    unseen_percentage: float,
    seed: int = 42
) -> Tuple[List[str], List[str]]:
    """
    Split disease labels into seen and unseen categories.
    
    Args:
        dataset: ChestXrayDataset instance
        unseen_percentage: Percentage of diseases to hold out as unseen
        seed: Random seed for reproducibility
        
    Returns:
        seen_diseases: List of disease names for training
        unseen_diseases: List of disease names for zero-shot evaluation
    """
    np.random.seed(seed)
    random.seed(seed)
    
    # Get all disease labels except "No Finding"
    all_diseases = [d for d in dataset.disease_labels if d != "No Finding"]
    
    # Calculate number of unseen diseases
    num_unseen = max(1, int(len(all_diseases) * unseen_percentage))
    
    # Randomly select unseen diseases
    unseen_diseases = np.random.choice(all_diseases, size=num_unseen, replace=False).tolist()
    seen_diseases = [d for d in all_diseases if d not in unseen_diseases]
    
    # Always include "No Finding" in seen diseases
    seen_diseases = ["No Finding"] + seen_diseases
    
    return seen_diseases, unseen_diseases


def create_train_val_test_splits(
    dataset: ChestXrayDataset,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42
) -> Tuple[Subset, Subset, Subset]:
    """
    Split dataset into train, validation, and test sets.
    
    Args:
        dataset: ChestXrayDataset instance
        train_ratio: Proportion for training
        val_ratio: Proportion for validation
        test_ratio: Proportion for testing
        seed: Random seed
        
    Returns:
        train_dataset, val_dataset, test_dataset
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-5
    
    # Get indices
    indices = list(range(len(dataset)))
    
    # First split: train+val vs test
    train_val_indices, test_indices = train_test_split(
        indices,
        test_size=test_ratio,
        random_state=seed
    )
    
    # Second split: train vs val
    val_size = val_ratio / (train_ratio + val_ratio)
    train_indices, val_indices = train_test_split(
        train_val_indices,
        test_size=val_size,
        random_state=seed
    )
    
    # Create subsets
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    test_dataset = Subset(dataset, test_indices)
    
    return train_dataset, val_dataset, test_dataset


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """Custom collate function for batching."""
    # Stack images and labels
    images = torch.stack([item['image'] for item in batch])
    labels = torch.stack([item['labels'] for item in batch])
    is_unseen = torch.stack([item['is_unseen'] for item in batch])
    
    # Collect reports and metadata
    reports = [item['report'] for item in batch]
    metadata = [item['metadata'] for item in batch]
    indices = torch.tensor([item['idx'] for item in batch])
    
    # Dataset info if using MultiDatasetWrapper
    dataset_indices = None
    dataset_names = None
    if 'dataset_idx' in batch[0]:
        dataset_indices = torch.tensor([item['dataset_idx'] for item in batch])
        dataset_names = [item['dataset_name'] for item in batch]
    
    output = {
        'images': images,
        'labels': labels,
        'reports': reports,
        'is_unseen': is_unseen,
        'metadata': metadata,
        'indices': indices
    }
    
    if dataset_indices is not None:
        output['dataset_indices'] = dataset_indices
        output['dataset_names'] = dataset_names
    
    return output


def create_data_loaders(
    config: Dict,
    unseen_percentage: float = 0.05,
    dataset_names: List[str] = ['chexpert']
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create data loaders for training, validation, and testing.
    
    Args:
        config: Configuration dictionary
        unseen_percentage: Percentage of diseases to treat as unseen
        dataset_names: List of datasets to use ('chexpert', 'mimic', 'padchest')
        
    Returns:
        train_loader, val_loader, test_loader
    """
    # Get transforms
    train_transform = get_transforms(
        image_size=config['data']['image_size'],
        is_training=True,
        normalize_mean=config['data']['normalize_mean'],
        normalize_std=config['data']['normalize_std']
    )
    
    val_transform = get_transforms(
        image_size=config['data']['image_size'],
        is_training=False,
        normalize_mean=config['data']['normalize_mean'],
        normalize_std=config['data']['normalize_std']
    )
    
    all_train_datasets = []
    all_val_datasets = []
    all_test_datasets = []
    
    for dataset_name in dataset_names:
        # Get dataset root
        if dataset_name == 'chexpert':
            data_root = config['data']['chexpert_root']
        elif dataset_name == 'mimic':
            data_root = config['data']['mimic_root']
        elif dataset_name == 'padchest':
            data_root = config['data']['padchest_root']
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")
        
        # Create full dataset
        full_dataset = ChestXrayDataset(
            data_root=data_root,
            csv_file=config['data']['train_csv'],
            disease_labels=config['data']['disease_labels'],
            transform=None,  # Will apply later
            dataset_name=dataset_name
        )
        
        # Split diseases into seen and unseen
        seen_diseases, unseen_diseases = split_datasets_with_unseen(
            full_dataset, 
            unseen_percentage,
            seed=config['experiment']['seed']
        )
        
        print(f"\n{dataset_name.upper()} Dataset:")
        print(f"Seen diseases ({len(seen_diseases)}): {seen_diseases}")
        print(f"Unseen diseases ({len(unseen_diseases)}): {unseen_diseases}")
        
        # Create train dataset (no unseen diseases)
        train_dataset = ChestXrayDataset(
            data_root=data_root,
            csv_file=config['data']['train_csv'],
            disease_labels=config['data']['disease_labels'],
            transform=train_transform,
            unseen_diseases=[],  # No unseen diseases in training
            dataset_name=dataset_name
        )
        
        # Create val/test datasets (with unseen diseases marked)
        val_test_dataset = ChestXrayDataset(
            data_root=data_root,
            csv_file=config['data']['train_csv'],
            disease_labels=config['data']['disease_labels'],
            transform=val_transform,
            unseen_diseases=unseen_diseases,
            dataset_name=dataset_name
        )
        
        # Split into train/val/test
        train_subset, val_subset, test_subset = create_train_val_test_splits(
            train_dataset,
            train_ratio=config['data']['train_ratio'],
            val_ratio=config['data']['val_ratio'],
            test_ratio=config['data']['test_ratio'],
            seed=config['experiment']['seed']
        )
        
        # For val/test, we need to use the dataset with unseen diseases marked
        val_indices = val_subset.indices
        test_indices = test_subset.indices
        
        val_subset = Subset(val_test_dataset, val_indices)
        test_subset = Subset(val_test_dataset, test_indices)
        
        all_train_datasets.append(train_subset)
        all_val_datasets.append(val_subset)
        all_test_datasets.append(test_subset)
    
    # Create multi-dataset wrappers if using multiple datasets
    if len(dataset_names) > 1:
        train_dataset = MultiDatasetWrapper([d.dataset for d in all_train_datasets])
        val_dataset = MultiDatasetWrapper([d.dataset for d in all_val_datasets])
        test_dataset = MultiDatasetWrapper([d.dataset for d in all_test_datasets])
    else:
        train_dataset = all_train_datasets[0]
        val_dataset = all_val_datasets[0]
        test_dataset = all_test_datasets[0]
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['training']['num_workers'],
        pin_memory=config['training']['pin_memory'],
        collate_fn=collate_fn,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['training']['num_workers'],
        pin_memory=config['training']['pin_memory'],
        collate_fn=collate_fn
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['training']['num_workers'],
        pin_memory=config['training']['pin_memory'],
        collate_fn=collate_fn
    )
    
    print(f"\nDataset sizes:")
    print(f"Train: {len(train_dataset)} samples")
    print(f"Val: {len(val_dataset)} samples")
    print(f"Test: {len(test_dataset)} samples")
    
    return train_loader, val_loader, test_loader