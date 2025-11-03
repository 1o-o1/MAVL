import os
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import torch
from torch.utils.data import Dataset
from PIL import Image
import json

from .preprocessing import preprocess_image


class ChestXrayDataset(Dataset):
    """Dataset for chest X-ray images with reports and multi-label disease annotations."""
    
    def __init__(
        self,
        data_root: str,
        csv_file: str,
        disease_labels: List[str],
        transform=None,
        report_column: str = 'report',
        unseen_diseases: Optional[List[str]] = None,
        dataset_name: str = 'chexpert'
    ):
        """
        Args:
            data_root: Root directory of the dataset
            csv_file: CSV file containing image paths and labels
            disease_labels: List of disease names
            transform: Image transformations
            report_column: Column name containing reports
            unseen_diseases: List of diseases to treat as unseen (zero-shot)
            dataset_name: Name of dataset (chexpert, mimic, padchest)
        """
        self.data_root = Path(data_root)
        self.transform = transform
        self.disease_labels = disease_labels
        self.report_column = report_column
        self.unseen_diseases = unseen_diseases or []
        self.dataset_name = dataset_name
        
        # Load dataset
        self.df = pd.read_csv(csv_file)
        
        # Preprocess labels based on dataset
        if dataset_name == 'chexpert':
            self._preprocess_chexpert_labels()
        elif dataset_name == 'mimic':
            self._preprocess_mimic_labels()
        elif dataset_name == 'padchest':
            self._preprocess_padchest_labels()
            
        # Filter valid samples
        self.df = self.df[self.df['Path'].notna()]
        
        # Create label indices
        self.label_to_idx = {label: idx for idx, label in enumerate(disease_labels)}
        self.idx_to_label = {idx: label for label, idx in self.label_to_idx.items()}
        
        # Mark unseen diseases
        self.unseen_indices = [self.label_to_idx[d] for d in unseen_diseases if d in self.label_to_idx]
        
    def _preprocess_chexpert_labels(self):
        """Preprocess CheXpert labels following paper's approach."""
        # Uncertain labels mapping (from paper Eq. after line 179)
        u_one_features = ['Atelectasis', 'Edema']
        u_zero_features = ['Cardiomegaly', 'Consolidation', 'Pleural Effusion']
        
        for col in u_one_features:
            if col in self.df.columns:
                self.df[col] = self.df[col].replace(-1, 1)
        
        for col in u_zero_features:
            if col in self.df.columns:
                self.df[col] = self.df[col].replace(-1, 0)
        
        # Replace remaining -1 with NaN
        for col in self.disease_labels:
            if col in self.df.columns:
                self.df[col] = self.df[col].replace(-1, np.nan)
    
    def _preprocess_mimic_labels(self):
        """Preprocess MIMIC-CXR labels."""
        # Similar preprocessing as CheXpert
        self._preprocess_chexpert_labels()
    
    def _preprocess_padchest_labels(self):
        """Preprocess PadChest labels."""
        # Map Spanish labels to English if needed
        # This is a placeholder - actual mapping would depend on PadChest format
        pass
    
    def __len__(self) -> int:
        return len(self.df)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Returns:
            Dictionary containing:
                - image: Tensor of shape (3, H, W)
                - labels: Binary tensor of shape (num_diseases,)
                - report: String containing the radiology report
                - is_unseen: Binary tensor marking unseen diseases
                - metadata: Dictionary with patient info
        """
        row = self.df.iloc[idx]
        
        # Load image
        img_path = self.data_root / row['Path']
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        else:
            image = preprocess_image(image)
        
        # Get labels
        labels = []
        for disease in self.disease_labels:
            if disease in row and pd.notna(row[disease]):
                labels.append(float(row[disease]))
            else:
                labels.append(0.0)  # Default to negative
        
        labels = torch.tensor(labels, dtype=torch.float32)
        
        # Mask unseen diseases
        is_unseen = torch.zeros_like(labels, dtype=torch.bool)
        for idx in self.unseen_indices:
            is_unseen[idx] = True
            labels[idx] = 0  # Zero out unseen labels during training
        
        # Get report
        report = row.get(self.report_column, "No report available.")
        if pd.isna(report):
            report = "No report available."
        
        # Get metadata
        metadata = {
            'patient_id': row.get('patient', 'unknown'),
            'study_id': row.get('study', 'unknown'),
            'sex': row.get('Sex', 'unknown'),
            'age': row.get('Age', -1),
            'view': row.get('Frontal/Lateral', 'unknown'),
            'projection': row.get('AP/PA', 'unknown')
        }
        
        return {
            'image': image,
            'labels': labels,
            'report': report,
            'is_unseen': is_unseen,
            'metadata': metadata,
            'idx': idx
        }


class MultiDatasetWrapper(Dataset):
    """Wrapper to combine multiple chest X-ray datasets."""
    
    def __init__(self, datasets: List[ChestXrayDataset]):
        """
        Args:
            datasets: List of ChestXrayDataset instances
        """
        self.datasets = datasets
        self.cumulative_sizes = []
        cumsum = 0
        
        for dataset in datasets:
            cumsum += len(dataset)
            self.cumulative_sizes.append(cumsum)
    
    def __len__(self) -> int:
        return self.cumulative_sizes[-1]
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get item from the appropriate dataset."""
        dataset_idx = 0
        sample_idx = idx
        
        for i, cumsum in enumerate(self.cumulative_sizes):
            if idx < cumsum:
                dataset_idx = i
                if i > 0:
                    sample_idx = idx - self.cumulative_sizes[i-1]
                break
        
        sample = self.datasets[dataset_idx][sample_idx]
        sample['dataset_idx'] = dataset_idx
        sample['dataset_name'] = self.datasets[dataset_idx].dataset_name
        
        return sample