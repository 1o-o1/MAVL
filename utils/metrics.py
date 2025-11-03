import numpy as np
import torch
from sklearn.metrics import roc_auc_score, f1_score, precision_recall_fscore_support, accuracy_score
from sklearn.metrics import average_precision_score, roc_curve, precision_recall_curve
from typing import Dict, Tuple, Optional, List
import warnings


def compute_metrics(
    predictions: torch.Tensor,
    labels: torch.Tensor,
    threshold: float = 0.5,
    is_unseen: Optional[torch.Tensor] = None,
    compute_per_class: bool = False,
    disease_names: Optional[List[str]] = None
) -> Dict[str, float]:
    """
    Compute comprehensive metrics for multi-label classification.
    
    Args:
        predictions: Model predictions (probabilities), shape (B, num_classes)
        labels: Ground truth labels, shape (B, num_classes)
        threshold: Threshold for binary predictions
        is_unseen: Binary mask for unseen diseases
        compute_per_class: Whether to compute per-class metrics
        disease_names: Names of diseases for per-class reporting
        
    Returns:
        Dictionary containing computed metrics
    """
    # Convert to numpy
    if torch.is_tensor(predictions):
        predictions = predictions.detach().cpu().numpy()
    if torch.is_tensor(labels):
        labels = labels.detach().cpu().numpy()
    if torch.is_tensor(is_unseen) and is_unseen is not None:
        is_unseen = is_unseen.detach().cpu().numpy()
    
    # Binary predictions
    binary_preds = (predictions > threshold).astype(int)
    
    # Overall metrics
    metrics = {}
    
    # Filter warnings for UndefinedMetricWarning
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=UserWarning)
        
        # AUC (Area Under ROC Curve)
        try:
            if is_unseen is None:
                auc = roc_auc_score(labels, predictions, average='macro')
                metrics['auc'] = auc
            else:
                # Separate AUC for seen and unseen
                seen_mask = ~is_unseen
                if seen_mask.any():
                    seen_auc = roc_auc_score(
                        labels[:, seen_mask[0]], 
                        predictions[:, seen_mask[0]], 
                        average='macro'
                    )
                    metrics['auc_seen'] = seen_auc
                
                if is_unseen.any():
                    unseen_auc = roc_auc_score(
                        labels[:, is_unseen[0]], 
                        predictions[:, is_unseen[0]], 
                        average='macro'
                    )
                    metrics['auc_unseen'] = unseen_auc
                
                # Overall AUC
                auc = roc_auc_score(labels, predictions, average='macro')
                metrics['auc'] = auc
        except ValueError:
            # Handle case where some classes have only one label
            metrics['auc'] = 0.0
        
        # F1 Score
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, binary_preds, average='macro', zero_division=0
        )
        metrics['f1'] = f1
        metrics['precision'] = precision
        metrics['recall'] = recall
        
        # Accuracy (exact match ratio for multi-label)
        accuracy = accuracy_score(labels, binary_preds)
        metrics['accuracy'] = accuracy
        
        # Average Precision
        try:
            avg_precision = average_precision_score(labels, predictions, average='macro')
            metrics['avg_precision'] = avg_precision
        except ValueError:
            metrics['avg_precision'] = 0.0
    
    # Per-class metrics if requested
    if compute_per_class:
        per_class_metrics = {}
        num_classes = labels.shape[1]
        
        for i in range(num_classes):
            class_name = disease_names[i] if disease_names else f"class_{i}"
            
            try:
                # AUC for this class
                class_auc = roc_auc_score(labels[:, i], predictions[:, i])
            except ValueError:
                class_auc = 0.0
            
            # F1 for this class
            class_precision, class_recall, class_f1, _ = precision_recall_fscore_support(
                labels[:, i], binary_preds[:, i], average='binary', zero_division=0
            )
            
            per_class_metrics[class_name] = {
                'auc': class_auc,
                'f1': class_f1,
                'precision': class_precision,
                'recall': class_recall
            }
        
        metrics['per_class'] = per_class_metrics
    
    return metrics


def compute_auc(
    predictions: torch.Tensor,
    labels: torch.Tensor,
    return_curves: bool = False
) -> Dict[str, np.ndarray]:
    """
    Compute AUC and optionally return ROC curves.
    
    Args:
        predictions: Model predictions (probabilities)
        labels: Ground truth labels
        return_curves: Whether to return ROC curve points
        
    Returns:
        Dictionary with AUC scores and optionally ROC curves
    """
    if torch.is_tensor(predictions):
        predictions = predictions.detach().cpu().numpy()
    if torch.is_tensor(labels):
        labels = labels.detach().cpu().numpy()
    
    results = {}
    num_classes = labels.shape[1]
    
    # Per-class AUC
    auc_scores = []
    roc_curves = []
    
    for i in range(num_classes):
        try:
            if return_curves:
                fpr, tpr, thresholds = roc_curve(labels[:, i], predictions[:, i])
                auc = roc_auc_score(labels[:, i], predictions[:, i])
                roc_curves.append({
                    'fpr': fpr,
                    'tpr': tpr,
                    'thresholds': thresholds,
                    'auc': auc
                })
            else:
                auc = roc_auc_score(labels[:, i], predictions[:, i])
            auc_scores.append(auc)
        except ValueError:
            auc_scores.append(0.0)
            if return_curves:
                roc_curves.append(None)
    
    results['per_class_auc'] = np.array(auc_scores)
    results['macro_auc'] = np.mean(auc_scores)
    
    if return_curves:
        results['roc_curves'] = roc_curves
    
    return results


def compute_f1_score(
    predictions: torch.Tensor,
    labels: torch.Tensor,
    threshold: float = 0.5,
    average: str = 'macro'
) -> Dict[str, float]:
    """
    Compute F1 score with different averaging strategies.
    
    Args:
        predictions: Model predictions (probabilities)
        labels: Ground truth labels
        threshold: Threshold for binary predictions
        average: Averaging strategy ('macro', 'micro', 'weighted')
        
    Returns:
        Dictionary with F1 scores
    """
    if torch.is_tensor(predictions):
        predictions = predictions.detach().cpu().numpy()
    if torch.is_tensor(labels):
        labels = labels.detach().cpu().numpy()
    
    binary_preds = (predictions > threshold).astype(int)
    
    results = {}
    
    # Different averaging strategies
    for avg in ['macro', 'micro', 'weighted']:
        f1 = f1_score(labels, binary_preds, average=avg, zero_division=0)
        results[f'f1_{avg}'] = f1
    
    # Per-class F1
    f1_per_class = f1_score(labels, binary_preds, average=None, zero_division=0)
    results['f1_per_class'] = f1_per_class
    
    return results


def compute_optimal_threshold(
    predictions: torch.Tensor,
    labels: torch.Tensor,
    metric: str = 'f1'
) -> Tuple[float, float]:
    """
    Find optimal threshold for binary classification.
    
    Args:
        predictions: Model predictions (probabilities)
        labels: Ground truth labels
        metric: Metric to optimize ('f1', 'accuracy', 'balanced')
        
    Returns:
        optimal_threshold: Best threshold value
        best_score: Best metric score
    """
    if torch.is_tensor(predictions):
        predictions = predictions.detach().cpu().numpy()
    if torch.is_tensor(labels):
        labels = labels.detach().cpu().numpy()
    
    thresholds = np.linspace(0.1, 0.9, 17)
    best_score = 0
    optimal_threshold = 0.5
    
    for thresh in thresholds:
        binary_preds = (predictions > thresh).astype(int)
        
        if metric == 'f1':
            score = f1_score(labels, binary_preds, average='macro', zero_division=0)
        elif metric == 'accuracy':
            score = accuracy_score(labels, binary_preds)
        elif metric == 'balanced':
            # Balanced accuracy
            score = balanced_accuracy_score(labels, binary_preds)
        
        if score > best_score:
            best_score = score
            optimal_threshold = thresh
    
    return optimal_threshold, best_score


def compute_confusion_matrix_multilabel(
    predictions: torch.Tensor,
    labels: torch.Tensor,
    threshold: float = 0.5
) -> Dict[str, np.ndarray]:
    """
    Compute confusion matrix statistics for multi-label classification.
    
    Args:
        predictions: Model predictions (probabilities)
        labels: Ground truth labels
        threshold: Threshold for binary predictions
        
    Returns:
        Dictionary with TP, FP, TN, FN counts per class
    """
    if torch.is_tensor(predictions):
        predictions = predictions.detach().cpu().numpy()
    if torch.is_tensor(labels):
        labels = labels.detach().cpu().numpy()
    
    binary_preds = (predictions > threshold).astype(int)
    
    # Compute confusion matrix elements
    TP = ((binary_preds == 1) & (labels == 1)).sum(axis=0)
    FP = ((binary_preds == 1) & (labels == 0)).sum(axis=0)
    TN = ((binary_preds == 0) & (labels == 0)).sum(axis=0)
    FN = ((binary_preds == 0) & (labels == 1)).sum(axis=0)
    
    return {
        'true_positives': TP,
        'false_positives': FP,
        'true_negatives': TN,
        'false_negatives': FN
    }