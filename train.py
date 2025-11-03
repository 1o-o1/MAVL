import argparse
import os
import time
from pathlib import Path
import yaml
import torch
import torch.nn as nn
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
import numpy as np

from models import MAVLModel
from data import create_data_loaders
from losses import CombinedLoss
from utils import (
    set_seed, set_deterministic, compute_metrics,
    get_logger, Logger
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train MAVL model for zero-shot lung disease detection')
    
    # Config file
    parser.add_argument('--config', type=str, default='config.yaml',
                        help='Path to configuration file')
    
    # Data arguments
    parser.add_argument('--data-root', type=str, default=None,
                        help='Root directory for datasets (overrides config)')
    parser.add_argument('--dataset', type=str, nargs='+', default=['chexpert'],
                        choices=['chexpert', 'mimic', 'padchest'],
                        help='Datasets to use for training')
    parser.add_argument('--unseen-percentage', type=float, default=0.05,
                        help='Percentage of diseases to treat as unseen')
    
    # Model arguments
    parser.add_argument('--model-size', type=str, default='14b',
                        choices=['7b', '14b', '32b'],
                        help='DeepSeek model size for report generation')
    parser.add_argument('--vision-encoder', type=str, default='vit_base_patch16_224',
                        help='Vision encoder model name')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=40,
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=16,
                        help='Batch size for training')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-5,
                        help='Weight decay')
    parser.add_argument('--grad-clip', type=float, default=5.0,
                        help='Gradient clipping norm')
    
    # Optimization arguments
    parser.add_argument('--optimizer', type=str, default='adamw',
                        choices=['adam', 'adamw'],
                        help='Optimizer to use')
    parser.add_argument('--scheduler', type=str, default='cosine',
                        choices=['cosine', 'plateau', 'none'],
                        help='Learning rate scheduler')
    parser.add_argument('--warmup-epochs', type=int, default=2,
                        help='Number of warmup epochs')
    
    # Loss arguments
    parser.add_argument('--lambda-contrastive', type=float, default=0.1,
                        help='Weight for contrastive loss')
    parser.add_argument('--use-focal-loss', action='store_true',
                        help='Use focal loss instead of BCE')
    
    # Memory arguments
    parser.add_argument('--memory-size', type=int, default=100,
                        help='Size of neural memory bank')
    parser.add_argument('--memory-update-freq', type=int, default=1,
                        help='Frequency of memory updates (steps)')
    
    # Experiment arguments
    parser.add_argument('--experiment-name', type=str, default=None,
                        help='Name for experiment (auto-generated if not provided)')
    parser.add_argument('--output-dir', type=str, default='./results',
                        help='Output directory for logs and checkpoints')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--deterministic', action='store_true',
                        help='Enable deterministic mode (slower)')
    
    # Logging arguments
    parser.add_argument('--log-interval', type=int, default=100,
                        help='Log metrics every N steps')
    parser.add_argument('--val-interval', type=int, default=1,
                        help='Validate every N epochs')
    parser.add_argument('--save-interval', type=int, default=5,
                        help='Save checkpoint every N epochs')
    parser.add_argument('--use-wandb', action='store_true',
                        help='Use Weights & Biases for logging')
    
    # Performance arguments
    parser.add_argument('--num-workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--mixed-precision', action='store_true',
                        help='Use mixed precision training')
    parser.add_argument('--gradient-accumulation', type=int, default=1,
                        help='Gradient accumulation steps')
    
    # Resume training
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    
    return parser.parse_args()


def load_config(config_path: str, args: argparse.Namespace) -> dict:
    """Load configuration and override with command line arguments."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Override config with command line arguments
    if args.data_root:
        for dataset in ['chexpert', 'mimic', 'padchest']:
            config['data'][f'{dataset}_root'] = args.data_root
    
    config['data']['unseen_percentages'] = [args.unseen_percentage]
    config['model']['vision_encoder_name'] = args.vision_encoder
    config['model']['lambda_contrastive'] = args.lambda_contrastive
    config['model']['memory_size'] = args.memory_size
    
    config['training']['num_epochs'] = args.epochs
    config['training']['batch_size'] = args.batch_size
    config['training']['learning_rate'] = args.lr
    config['training']['weight_decay'] = args.weight_decay
    config['training']['grad_clip_norm'] = args.grad_clip
    config['training']['optimizer'] = args.optimizer
    config['training']['scheduler'] = args.scheduler
    config['training']['warmup_epochs'] = args.warmup_epochs
    config['training']['num_workers'] = args.num_workers
    config['training']['mixed_precision'] = args.mixed_precision
    config['training']['gradient_accumulation_steps'] = args.gradient_accumulation
    
    config['experiment']['seed'] = args.seed
    config['experiment']['deterministic'] = args.deterministic
    config['experiment']['output_dir'] = args.output_dir
    
    config['logging']['use_wandb'] = args.use_wandb
    config['logging']['log_frequency'] = args.log_interval
    
    # Generate experiment name if not provided
    if args.experiment_name:
        config['experiment']['experiment_name'] = args.experiment_name
    else:
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        config['experiment']['experiment_name'] = f"mavl_{args.model_size}_{timestamp}"
    
    return config


def create_optimizer(model: nn.Module, config: dict) -> torch.optim.Optimizer:
    """Create optimizer based on configuration."""
    opt_config = config['training']
    
    # Get parameters with different learning rates
    params = [
        {'params': model.vision_encoder.parameters(), 'lr': opt_config['learning_rate'] * 0.1},
        {'params': model.text_encoder.parameters(), 'lr': opt_config['learning_rate'] * 0.1},
        {'params': model.dual_head.parameters(), 'lr': opt_config['learning_rate']},
        {'params': model.neural_memory.parameters(), 'lr': opt_config['learning_rate']}
    ]
    
    if opt_config['optimizer'] == 'adam':
        optimizer = Adam(
            params,
            lr=opt_config['learning_rate'],
            weight_decay=opt_config['weight_decay'],
            betas=(opt_config.get('adam_beta1', 0.9), opt_config.get('adam_beta2', 0.999)),
            eps=opt_config.get('adam_eps', 1e-8)
        )
    elif opt_config['optimizer'] == 'adamw':
        optimizer = AdamW(
            params,
            lr=opt_config['learning_rate'],
            weight_decay=opt_config['weight_decay'],
            betas=(opt_config.get('adam_beta1', 0.9), opt_config.get('adam_beta2', 0.999)),
            eps=opt_config.get('adam_eps', 1e-8)
        )
    else:
        raise ValueError(f"Unknown optimizer: {opt_config['optimizer']}")
    
    return optimizer


def create_scheduler(optimizer: torch.optim.Optimizer, config: dict):
    """Create learning rate scheduler."""
    opt_config = config['training']
    
    if opt_config['scheduler'] == 'cosine':
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=opt_config['num_epochs'] - opt_config['warmup_epochs'],
            eta_min=opt_config.get('min_lr', 1e-6)
        )
    elif opt_config['scheduler'] == 'plateau':
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode='max',
            factor=0.5,
            patience=5,
            min_lr=opt_config.get('min_lr', 1e-6)
        )
    elif opt_config['scheduler'] == 'none':
        scheduler = None
    else:
        raise ValueError(f"Unknown scheduler: {opt_config['scheduler']}")
    
    return scheduler


def train_epoch(
    model: nn.Module,
    train_loader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    config: dict,
    logger: Logger,
    scaler: Optional[GradScaler] = None
) -> dict:
    """Train for one epoch."""
    model.train()
    
    total_loss = 0
    total_supervised_loss = 0
    total_contrastive_loss = 0
    num_batches = len(train_loader)
    
    # Progress bar
    pbar = tqdm(train_loader, desc=f'Epoch {epoch} [Train]')
    
    for batch_idx, batch in enumerate(pbar):
        # Move to device
        device = next(model.parameters()).device
        images = batch['images'].to(device)
        labels = batch['labels'].to(device)
        reports = batch['reports']
        is_unseen = batch['is_unseen'].to(device)
        
        # Mixed precision training
        if scaler is not None:
            with autocast():
                # Forward pass
                outputs = model(images, reports)
                
                # Compute loss
                loss_dict = model.compute_loss(outputs, labels, is_unseen)
                loss = loss_dict['loss'] / config['training']['gradient_accumulation_steps']
        else:
            # Forward pass
            outputs = model(images, reports)
            
            # Compute loss
            loss_dict = model.compute_loss(outputs, labels, is_unseen)
            loss = loss_dict['loss'] / config['training']['gradient_accumulation_steps']
        
        # Backward pass
        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        
        # Update memory before optimizer step (manual update)
        if (batch_idx + 1) % config['training'].get('memory_update_freq', 1) == 0:
            model.update_memory(loss_dict['loss'])
        
        # Gradient accumulation
        if (batch_idx + 1) % config['training']['gradient_accumulation_steps'] == 0:
            # Gradient clipping
            if scaler is not None:
                scaler.unscale_(optimizer)
            
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                config['training']['grad_clip_norm']
            )
            
            # Optimizer step
            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            
            optimizer.zero_grad()
        
        # Update metrics
        total_loss += loss_dict['loss'].item()
        total_supervised_loss += loss_dict['supervised_loss'].item()
        total_contrastive_loss += loss_dict['contrastive_loss'].item()
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f"{loss_dict['loss'].item():.4f}",
            'sup_loss': f"{loss_dict['supervised_loss'].item():.4f}",
            'cont_loss': f"{loss_dict['contrastive_loss'].item():.4f}"
        })
        
        # Log metrics
        if (batch_idx + 1) % config['logging']['log_frequency'] == 0:
            step = epoch * num_batches + batch_idx
            logger.log_metrics({
                'loss': loss_dict['loss'].item(),
                'supervised_loss': loss_dict['supervised_loss'].item(),
                'contrastive_loss': loss_dict['contrastive_loss'].item(),
                'learning_rate': optimizer.param_groups[0]['lr']
            }, step=step, phase='train')
    
    # Average metrics
    avg_metrics = {
        'loss': total_loss / num_batches,
        'supervised_loss': total_supervised_loss / num_batches,
        'contrastive_loss': total_contrastive_loss / num_batches
    }
    
    return avg_metrics


@torch.no_grad()
def validate(
    model: nn.Module,
    val_loader,
    epoch: int,
    config: dict,
    logger: Logger
) -> dict:
    """Validate the model."""
    model.eval()
    
    all_predictions = []
    all_labels = []
    all_is_unseen = []
    total_loss = 0
    num_batches = len(val_loader)
    
    # Progress bar
    pbar = tqdm(val_loader, desc=f'Epoch {epoch} [Val]')
    
    for batch in pbar:
        # Move to device
        device = next(model.parameters()).device
        images = batch['images'].to(device)
        labels = batch['labels'].to(device)
        reports = batch['reports']
        is_unseen = batch['is_unseen'].to(device)
        
        # Forward pass
        outputs = model(images, reports)
        
        # Compute loss
        loss_dict = model.compute_loss(outputs, labels, is_unseen)
        total_loss += loss_dict['loss'].item()
        
        # Get predictions
        predictions = torch.sigmoid(outputs['logits'])
        
        # Collect for metrics
        all_predictions.append(predictions.cpu())
        all_labels.append(labels.cpu())
        all_is_unseen.append(is_unseen.cpu())
    
    # Concatenate all batches
    all_predictions = torch.cat(all_predictions, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    all_is_unseen = torch.cat(all_is_unseen, dim=0)
    
    # Compute metrics
    metrics = compute_metrics(
        all_predictions,
        all_labels,
        is_unseen=all_is_unseen,
        disease_names=config['data']['disease_labels']
    )
    
    # Add loss to metrics
    metrics['loss'] = total_loss / num_batches
    
    # Log metrics
    logger.log_metrics(metrics, step=epoch, phase='val')
    
    return metrics


def main():
    """Main training function."""
    # Parse arguments
    args = parse_args()
    
    # Load configuration
    config = load_config(args.config, args)
    
    # Set random seed
    set_seed(config['experiment']['seed'])
    if config['experiment']['deterministic']:
        set_deterministic(True)
    
    # Create output directory
    output_dir = Path(config['experiment']['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize logger
    logger = get_logger(
        log_dir=str(output_dir),
        experiment_name=config['experiment']['experiment_name'],
        config=config,
        use_tensorboard=config['logging'].get('use_tensorboard', True),
        use_wandb=config['logging'].get('use_wandb', False),
        project=config['logging'].get('wandb_project', 'mavl-lung-disease'),
        entity=config['logging'].get('wandb_entity', None)
    )
    
    # Log initial info
    logger.log_text(f"Starting experiment: {config['experiment']['experiment_name']}")
    logger.log_text(f"Configuration:\n{yaml.dump(config, indent=2)}")
    
    # Create data loaders
    logger.log_text("Creating data loaders...")
    train_loader, val_loader, test_loader = create_data_loaders(
        config,
        unseen_percentage=args.unseen_percentage,
        dataset_names=args.dataset
    )
    
    # Create model
    logger.log_text("Creating model...")
    model = MAVLModel(config)
    
    # Move to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    logger.log_text(f"Model moved to {device}")
    
    # Log model parameters
    param_counts = model.get_num_parameters()
    logger.log_text(f"Model parameters: {param_counts}")
    
    # Create loss function
    criterion = CombinedLoss(
        lambda_contrastive=config['model']['lambda_contrastive'],
        temperature=config['model']['temperature'],
        supervised_loss_type='focal' if args.use_focal_loss else 'bce'
    )
    
    # Create optimizer
    optimizer = create_optimizer(model, config)
    
    # Create scheduler
    scheduler = create_scheduler(optimizer, config)
    
    # Mixed precision scaler
    scaler = GradScaler() if config['training']['mixed_precision'] else None
    
    # Resume from checkpoint if specified
    start_epoch = 0
    best_metric = 0
    
    if args.resume:
        logger.log_text(f"Resuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_metric = checkpoint.get('best_metric', 0)
        logger.log_text(f"Resumed from epoch {start_epoch}")
    
    # Training loop
    logger.log_text("Starting training...")
    
    for epoch in range(start_epoch, config['training']['num_epochs']):
        # Learning rate warmup
        if epoch < config['training']['warmup_epochs']:
            warmup_lr = config['training']['learning_rate'] * (epoch + 1) / config['training']['warmup_epochs']
            for param_group in optimizer.param_groups:
                param_group['lr'] = warmup_lr * (param_group['lr'] / config['training']['learning_rate'])
        
        # Train
        train_metrics = train_epoch(
            model, train_loader, criterion, optimizer,
            epoch, config, logger, scaler
        )
        
        # Validate
        if epoch % args.val_interval == 0:
            val_metrics = validate(model, val_loader, epoch, config, logger)
            
            # Check if best model
            if val_metrics['auc'] > best_metric:
                best_metric = val_metrics['auc']
                is_best = True
                logger.log_text(f"New best model! AUC: {best_metric:.4f}")
            else:
                is_best = False
            
            # Save checkpoint
            if epoch % args.save_interval == 0 or is_best:
                logger.save_checkpoint(
                    model_state=model.state_dict(),
                    optimizer_state=optimizer.state_dict(),
                    epoch=epoch,
                    metrics=val_metrics,
                    is_best=is_best
                )
        
        # Update scheduler
        if scheduler is not None:
            if isinstance(scheduler, ReduceLROnPlateau):
                scheduler.step(val_metrics['auc'])
            else:
                scheduler.step()
        
        # Log epoch summary
        logger.log_text(
            f"Epoch {epoch}: "
            f"Train Loss={train_metrics['loss']:.4f}, "
            f"Val Loss={val_metrics['loss']:.4f}, "
            f"Val AUC={val_metrics['auc']:.4f}"
        )
    
    # Final evaluation on test set
    logger.log_text("Running final evaluation on test set...")
    test_metrics = validate(model, test_loader, config['training']['num_epochs'], config, logger)
    logger.log_text(f"Test metrics: {test_metrics}")
    
    # Save final model
    final_path = output_dir / config['experiment']['experiment_name'] / 'final_model.pth'
    torch.save(model.state_dict(), final_path)
    logger.log_text(f"Saved final model to {final_path}")
    
    # Close logger
    logger.close()


if __name__ == '__main__':
    main()