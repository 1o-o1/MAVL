#!/usr/bin/env python3
"""
ABLATION STUDY: MAVL WITHOUT NEURAL MEMORY

This script is IDENTICAL to full_scale_training_corrected.py EXCEPT:
- Neural memory module is DISABLED (pass-through)
- All other hyperparameters remain the same
- Results saved to results_full_scale_ablation/

CHANGES FROM BASELINE (full_scale_training_corrected.py):
1. Line 257: Memory forward pass replaced with identity (no memory retrieval)
2. Line 568: Memory update disabled (commented out)
3. Line 264: Memory features replaced with zeros
4. Line 42: RESULTS_DIR changed to "results_full_scale_ablation"

This allows direct paired comparison to measure memory contribution.
"""

import os
import sys
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import logging
from tqdm import tqdm
import warnings
import time
from collections import defaultdict

warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

ROOT_DIR = Path("E:\\Research\\zero shot lung")
DATA_ROOT = ROOT_DIR / "CheXpert-v1.0-small"
RESULTS_DIR = ROOT_DIR / "results_full_scale_ablation"  # CHANGED
RESULTS_DIR.mkdir(exist_ok=True)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(RESULTS_DIR / 'full_scale_training_ablation.log'),  # CHANGED
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

CONFIG = {
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'batch_size': 8,
    'num_epochs': 40,
    'num_seeds': 3,
    'learning_rate': 1e-4,
    'weight_decay': 1e-5,
    'grad_clip': 5.0,
    'warmup_epochs': 2,
    'lambda_contrastive': 0.1,
    'memory_size': 256,
    'memory_dim': 768,
    'memory_update_rate': 0.1,
    'num_pathologies': 14,
    'num_aspects': 5,
    'vision_dim': 768,
    'max_samples': 50000,
    'aspect_temp_init': 2.0,
}

# ============================================================================
# NEURAL MEMORY MODULE - DISABLED FOR ABLATION
# ============================================================================

class NeuralMemoryBankDisabled(nn.Module):
    """
    ABLATION: Memory module disabled (pass-through identity).

    This returns zero features and does not update memory.
    """

    def __init__(self, num_slots=256, dim=768, update_rate=0.1, temperature=1.0):
        super().__init__()
        self.num_slots = num_slots
        self.dim = dim
        # No actual memory storage needed
        logger.info(f"[ABLATION] Neural Memory Bank DISABLED")

    def forward(self, features):
        """
        Return zero memory features (identity/no-op).
        """
        B, D = features.shape
        # Return zeros as memory features
        memory_features = torch.zeros_like(features)
        attention = torch.zeros(B, self.num_slots, device=features.device)
        return memory_features, attention

    @torch.no_grad()
    def update_memory_image_wise(self, features):
        """
        ABLATION: No memory update (no-op).
        """
        pass  # Do nothing


# ============================================================================
# VISION-LANGUAGE MODEL (IDENTICAL TO BASELINE EXCEPT MEMORY DISABLED)
# ============================================================================

class MAVLWithMemoryAndAspects(nn.Module):
    """
    ABLATION VERSION: Memory module replaced with disabled version.
    Everything else identical to baseline.
    """

    def __init__(self, num_pathologies=14, num_aspects=5, memory_size=256,
                 memory_dim=768, vision_dim=768):
        super().__init__()
        self.num_pathologies = num_pathologies
        self.num_aspects = num_aspects
        self.memory_size = memory_size
        self.vision_dim = vision_dim

        # 1. Vision encoder (pretrained ViT-B, frozen)
        from torchvision.models import vit_b_16
        self.vision_encoder = vit_b_16(pretrained=True)
        self.vision_encoder.heads = nn.Identity()

        for param in self.vision_encoder.parameters():
            param.requires_grad = False

        logger.info(f"Vision encoder (ViT-B) loaded")

        # 2. ABLATION: Use disabled memory module
        self.memory = NeuralMemoryBankDisabled(
            num_slots=memory_size,
            dim=memory_dim,
            update_rate=0.1,
            temperature=1.0
        )

        # 3. Aspect embeddings (same as baseline)
        self.aspect_embeddings = nn.Embedding(
            num_pathologies * num_aspects,
            vision_dim
        )
        nn.init.xavier_uniform_(self.aspect_embeddings.weight)

        # 4. Learned aspect temperature
        self.aspect_temp = nn.Parameter(torch.tensor(CONFIG['aspect_temp_init']))

        # 5. Classification head (SAME architecture as baseline)
        classifier_input_dim = vision_dim * 2 + num_pathologies

        self.classifier = nn.Sequential(
            nn.Linear(classifier_input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_pathologies)
        )

        # 6. Contrastive projection head
        self.contrastive_head = nn.Sequential(
            nn.Linear(vision_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256)
        )

        logger.info(f"[ABLATION] MAVL model created WITHOUT memory")

    def forward(self, images, return_memory=False, return_aspects=False):
        """Forward pass (memory features will be zeros)."""
        # 1. Extract visual features
        with torch.no_grad():
            visual_features = self.vision_encoder(images)

        # 2. ABLATION: Memory returns zeros
        memory_features, memory_attention = self.memory(visual_features)

        # 3. Aspect-based scores (unchanged)
        aspect_scores = self._compute_aspect_scores(visual_features)

        # 4. Combine all features (memory_features are zeros)
        combined_features = torch.cat(
            [visual_features, memory_features, aspect_scores],
            dim=1
        )

        # 5. Classification
        logits = self.classifier(combined_features)

        if return_memory or return_aspects:
            return_dict = {'logits': logits}
            if return_memory:
                return_dict['memory_features'] = memory_features
                return_dict['memory_attention'] = memory_attention
            if return_aspects:
                return_dict['aspect_scores'] = aspect_scores
            return return_dict

        return logits

    def _compute_aspect_scores(self, visual_features):
        """Aspect matching (unchanged from baseline)."""
        B = visual_features.shape[0]
        device = visual_features.device
        visual_norm = F.normalize(visual_features, p=2, dim=1)
        aspect_scores_list = []

        for c in range(self.num_pathologies):
            aspect_sims = []
            for j in range(self.num_aspects):
                aspect_idx = c * self.num_aspects + j
                aspect_emb = self.aspect_embeddings(
                    torch.tensor(aspect_idx, device=device)
                )
                aspect_emb = F.normalize(aspect_emb.unsqueeze(0), p=2, dim=1).squeeze(0)
                similarity = torch.matmul(visual_norm, aspect_emb)
                aspect_sims.append(similarity)

            aspect_sims = torch.stack(aspect_sims, dim=1)
            aspect_weights = F.softmax(self.aspect_temp * aspect_sims, dim=1)
            pathology_score = torch.sum(aspect_weights * aspect_sims, dim=1)
            aspect_scores_list.append(pathology_score)

        aspect_scores = torch.stack(aspect_scores_list, dim=1)
        return aspect_scores

    def get_contrastive_features(self, images):
        """Get features for contrastive learning."""
        with torch.no_grad():
            visual_features = self.vision_encoder(images)
        return self.contrastive_head(visual_features)

    def update_memory_with_batch(self, visual_features):
        """ABLATION: No memory update."""
        pass  # No-op


# ============================================================================
# LOSS FUNCTIONS (IDENTICAL TO BASELINE)
# ============================================================================

class CombinedLoss(nn.Module):
    """Combined supervised + contrastive loss (unchanged)."""

    def __init__(self, lambda_contrastive=0.1, temperature=0.07):
        super().__init__()
        self.lambda_contrastive = lambda_contrastive
        self.temperature = temperature
        self.ce_loss = nn.BCEWithLogitsLoss()

    def forward(self, logits, labels, contrastive_features=None, text_features=None):
        supervised_loss = self.ce_loss(logits, labels.float())
        if contrastive_features is not None and text_features is not None:
            contrastive_loss = self._compute_contrastive_loss(
                contrastive_features, labels
            )
            total_loss = supervised_loss + self.lambda_contrastive * contrastive_loss
        else:
            total_loss = supervised_loss
        return total_loss

    def _compute_contrastive_loss(self, image_features, labels):
        """Simple contrastive loss."""
        image_features = F.normalize(image_features, p=2, dim=1)
        loss = 0
        count = 0
        for c in range(labels.shape[1]):
            pos_mask = labels[:, c] == 1
            neg_mask = labels[:, c] == 0
            if pos_mask.sum() > 0 and neg_mask.sum() > 0:
                pos_features = image_features[pos_mask]
                if pos_features.shape[0] > 1:
                    pos_sim = torch.matmul(pos_features, pos_features.T) / self.temperature
                    pos_loss = -torch.log_softmax(pos_sim, dim=1).diag().mean()
                    loss += pos_loss
                    count += 1
        return loss / max(count, 1)


# ============================================================================
# DATASET (IDENTICAL TO BASELINE)
# ============================================================================

class CheXpertDataset(Dataset):
    """CheXpert dataset (unchanged)."""

    def __init__(self, csv_path, image_dir, pathology_cols=None, transform=None, max_samples=None):
        self.image_dir = Path(image_dir)
        self.transform = transform

        self.pathology_cols = [
            'No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly',
            'Lung Opacity', 'Lung Lesion', 'Edema', 'Consolidation',
            'Pneumonia', 'Atelectasis', 'Pneumothorax', 'Pleural Effusion',
            'Pleural Other', 'Fracture', 'Support Devices'
        ]

        df = pd.read_csv(csv_path)
        df = df[['Path'] + self.pathology_cols].dropna(subset=['Path'])

        for col in self.pathology_cols:
            df[col] = df[col].fillna(0).clip(lower=0).astype(int)

        if max_samples:
            df = df.sample(min(max_samples, len(df)), random_state=42)

        self.data = df
        logger.info(f"CheXpertDataset loaded: {len(df)} samples")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        csv_path = row['Path']
        if 'CheXpert-v1.0-small/' in csv_path:
            relative_path = csv_path.split('CheXpert-v1.0-small/')[-1]
        else:
            relative_path = csv_path

        img_path = self.image_dir.parent / relative_path

        from PIL import Image
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        labels = torch.tensor([row[col] for col in self.pathology_cols], dtype=torch.float32)
        return image, labels


# ============================================================================
# TRAINING (IDENTICAL EXCEPT NO MEMORY UPDATE)
# ============================================================================

def validate(model, val_loader, device):
    """Validate model and compute metrics."""
    from sklearn.metrics import roc_auc_score, f1_score, average_precision_score

    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc="Validating", leave=False):
            images = images.to(device)
            logits = model(images)
            probs = torch.sigmoid(logits)
            all_preds.append(probs.cpu().numpy())
            all_labels.append(labels.numpy())

    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    aurocs = []
    f1s = []
    auprcs = []

    for i in range(all_labels.shape[1]):
        try:
            unique_labels = np.unique(all_labels[:, i])
            if len(unique_labels) < 2:
                aurocs.append(np.nan)
            else:
                auroc = roc_auc_score(all_labels[:, i], all_preds[:, i])
                aurocs.append(auroc)

            auprc = average_precision_score(all_labels[:, i], all_preds[:, i])
            auprcs.append(auprc)

            pred_binary = (all_preds[:, i] > 0.5).astype(int)
            f1 = f1_score(all_labels[:, i], pred_binary, zero_division=0)
            f1s.append(f1)
        except Exception as e:
            aurocs.append(np.nan)
            f1s.append(0.0)
            auprcs.append(0.5)

    aurocs_valid = [a for a in aurocs if not np.isnan(a)]

    return {
        'auroc': np.mean(aurocs_valid) if aurocs_valid else np.nan,
        'auroc_std': np.std(aurocs_valid) if aurocs_valid else np.nan,
        'f1': np.mean(f1s),
        'f1_std': np.std(f1s),
        'auprc': np.mean(auprcs),
        'auprc_std': np.std(auprcs),
    }


def train_epoch(model, train_loader, optimizer, criterion, device):
    """Train for one epoch WITHOUT memory updates."""
    model.train()
    total_loss = 0
    num_batches = 0

    pbar = tqdm(train_loader, desc="Training", leave=False)

    for images, labels in pbar:
        images = images.to(device)
        labels = labels.to(device)

        with autocast():
            logits = model(images)
            loss = criterion(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG['grad_clip'])
        optimizer.step()

        # ABLATION: NO MEMORY UPDATE (commented out)
        # with torch.no_grad():
        #     visual_features = model.vision_encoder(images)
        #     model.update_memory_with_batch(visual_features.detach())

        total_loss += loss.item()
        num_batches += 1
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    return total_loss / num_batches


def train_full_experiment(seed=42):
    """Run full training experiment (ablation version)."""
    logger.info(f"\n{'='*80}")
    logger.info(f"ABLATION TRAINING (MEMORY OFF) - SEED {seed}")
    logger.info(f"{'='*80}\n")

    start_time = time.time()

    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

    device = torch.device(CONFIG['device'])

    logger.info("Loading CheXpert dataset...")
    from torchvision import transforms

    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_csv = DATA_ROOT / "train.csv"
    valid_csv = DATA_ROOT / "valid.csv"

    train_dataset = CheXpertDataset(
        train_csv, DATA_ROOT / "train",
        transform=train_transform,
        max_samples=CONFIG['max_samples']
    )

    val_dataset = CheXpertDataset(
        valid_csv, DATA_ROOT / "valid",
        transform=val_transform,
        max_samples=CONFIG['max_samples'] // 10
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=CONFIG['batch_size'],
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=CONFIG['batch_size'],
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )

    logger.info(f"Train loader: {len(train_loader)} batches")
    logger.info(f"Val loader: {len(val_loader)} batches")

    logger.info("Creating MAVL model (ABLATION: memory disabled)...")
    model = MAVLWithMemoryAndAspects(
        num_pathologies=CONFIG['num_pathologies'],
        num_aspects=CONFIG['num_aspects'],
        memory_size=CONFIG['memory_size'],
        memory_dim=CONFIG['memory_dim'],
        vision_dim=CONFIG['vision_dim']
    )
    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")

    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=CONFIG['learning_rate'],
        weight_decay=CONFIG['weight_decay']
    )

    scheduler = CosineAnnealingLR(optimizer, T_max=CONFIG['num_epochs'])
    criterion = CombinedLoss(lambda_contrastive=CONFIG['lambda_contrastive'])

    best_auroc = 0
    results_log = []

    for epoch in range(1, CONFIG['num_epochs'] + 1):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_metrics = validate(model, val_loader, device)
        scheduler.step()

        logger.info(f"Epoch {epoch}/{CONFIG['num_epochs']}")
        logger.info(f"  Train Loss: {train_loss:.4f}")
        logger.info(f"  Val AUROC: {val_metrics['auroc']:.4f} ± {val_metrics['auroc_std']:.4f}")
        logger.info(f"  Val F1: {val_metrics['f1']:.4f} ± {val_metrics['f1_std']:.4f}")
        logger.info(f"  Val AUPRC: {val_metrics['auprc']:.4f} ± {val_metrics['auprc_std']:.4f}")

        if not np.isnan(val_metrics['auroc']) and val_metrics['auroc'] > best_auroc:
            best_auroc = val_metrics['auroc']
            logger.info(f"*** NEW BEST AUROC: {best_auroc:.4f} ***")

        results_log.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'val_auroc': val_metrics['auroc'],
            'val_auroc_std': val_metrics['auroc_std'],
            'val_f1': val_metrics['f1'],
            'val_f1_std': val_metrics['f1_std'],
            'val_auprc': val_metrics['auprc'],
            'val_auprc_std': val_metrics['auprc_std'],
        })

    elapsed = time.time() - start_time

    result = {
        'seed': seed,
        'best_auroc': best_auroc,
        'final_auroc': val_metrics['auroc'],
        'final_f1': val_metrics['f1'],
        'final_auprc': val_metrics['auprc'],
        'elapsed_time': elapsed,
        'results_log': results_log
    }

    return result


def main():
    """Run ablation training with multiple seeds."""
    logger.info("\n" + "="*80)
    logger.info("ABLATION STUDY - MEMORY DISABLED")
    logger.info("="*80)
    logger.info(f"Configuration: {json.dumps(CONFIG, indent=2)}")
    logger.info("="*80 + "\n")

    all_results = []

    for seed in [42, 123, 456][:CONFIG['num_seeds']]:
        result = train_full_experiment(seed=seed)
        all_results.append(result)

        with open(RESULTS_DIR / f'seed_{seed}_results.json', 'w') as f:
            json.dump(result, f, indent=2, default=str)

    aurocs = [r['best_auroc'] for r in all_results]
    summary = {
        'num_seeds': len(all_results),
        'auroc_mean': np.mean(aurocs),
        'auroc_std': np.std(aurocs),
        'all_results': all_results,
        'timestamp': datetime.now().isoformat()
    }

    with open(RESULTS_DIR / 'summary_results.json', 'w') as f:
        json.dump(summary, f, indent=2, default=str)

    logger.info("\n" + "="*80)
    logger.info("ABLATION TRAINING COMPLETE")
    logger.info("="*80)
    logger.info(f"Results saved to: {RESULTS_DIR}")
    logger.info(f"Mean AUROC (best): {summary['auroc_mean']:.4f} ± {summary['auroc_std']:.4f}")


if __name__ == '__main__':
    main()
