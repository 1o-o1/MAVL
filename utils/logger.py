import os
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List
import torch
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# Note: wandb is optional
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


class Logger:
    """Base logger class for experiment tracking."""
    
    def __init__(
        self,
        log_dir: str,
        experiment_name: str,
        config: Optional[Dict] = None
    ):
        """
        Args:
            log_dir: Directory for saving logs
            experiment_name: Name of the experiment
            config: Configuration dictionary
        """
        self.log_dir = Path(log_dir)
        self.experiment_name = experiment_name
        self.config = config or {}
        
        # Create log directory
        self.log_path = self.log_dir / experiment_name
        self.log_path.mkdir(parents=True, exist_ok=True)
        
        # Save configuration
        self._save_config()
        
        # Initialize metrics storage
        self.metrics = {}
        self.start_time = time.time()
    
    def _save_config(self):
        """Save configuration to file."""
        config_path = self.log_path / 'config.json'
        with open(config_path, 'w') as f:
            json.dump(self.config, f, indent=2)
    
    def log_metrics(
        self,
        metrics: Dict[str, float],
        step: int,
        phase: str = 'train'
    ):
        """
        Log metrics.
        
        Args:
            metrics: Dictionary of metric values
            step: Current step/epoch
            phase: Phase name (train/val/test)
        """
        if phase not in self.metrics:
            self.metrics[phase] = {}
        
        for name, value in metrics.items():
            if name not in self.metrics[phase]:
                self.metrics[phase][name] = []
            
            self.metrics[phase][name].append({
                'step': step,
                'value': value,
                'timestamp': time.time()
            })
    
    def log_text(self, text: str, filename: str = 'log.txt'):
        """Log text to file."""
        log_file = self.log_path / filename
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        with open(log_file, 'a') as f:
            f.write(f'[{timestamp}] {text}\n')
    
    def save_checkpoint(
        self,
        model_state: Dict,
        optimizer_state: Dict,
        epoch: int,
        metrics: Dict[str, float],
        is_best: bool = False
    ):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model_state,
            'optimizer_state_dict': optimizer_state,
            'metrics': metrics,
            'timestamp': datetime.now().isoformat()
        }
        
        # Save regular checkpoint
        checkpoint_path = self.log_path / f'checkpoint_epoch_{epoch}.pth'
        torch.save(checkpoint, checkpoint_path)
        
        # Save best checkpoint
        if is_best:
            best_path = self.log_path / 'best_checkpoint.pth'
            torch.save(checkpoint, best_path)
        
        # Keep only last N checkpoints
        self._cleanup_checkpoints(keep_last=3, keep_best=True)
    
    def _cleanup_checkpoints(self, keep_last: int = 3, keep_best: bool = True):
        """Remove old checkpoints."""
        checkpoints = sorted(self.log_path.glob('checkpoint_epoch_*.pth'))
        
        if len(checkpoints) > keep_last:
            for checkpoint in checkpoints[:-keep_last]:
                checkpoint.unlink()
    
    def save_metrics(self):
        """Save all metrics to JSON file."""
        metrics_path = self.log_path / 'metrics.json'
        with open(metrics_path, 'w') as f:
            json.dump(self.metrics, f, indent=2)
    
    def get_summary(self) -> Dict[str, Any]:
        """Get experiment summary."""
        duration = time.time() - self.start_time
        
        summary = {
            'experiment_name': self.experiment_name,
            'duration_seconds': duration,
            'duration_formatted': self._format_duration(duration),
            'best_metrics': self._get_best_metrics(),
            'final_metrics': self._get_final_metrics()
        }
        
        return summary
    
    def _format_duration(self, seconds: float) -> str:
        """Format duration in human-readable format."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        seconds = int(seconds % 60)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    
    def _get_best_metrics(self) -> Dict[str, float]:
        """Get best metrics across all phases."""
        best_metrics = {}
        
        for phase, phase_metrics in self.metrics.items():
            for metric_name, values in phase_metrics.items():
                if values:
                    # Determine if higher is better
                    if 'loss' in metric_name.lower():
                        best_value = min(v['value'] for v in values)
                    else:
                        best_value = max(v['value'] for v in values)
                    
                    key = f'{phase}_{metric_name}_best'
                    best_metrics[key] = best_value
        
        return best_metrics
    
    def _get_final_metrics(self) -> Dict[str, float]:
        """Get final metrics."""
        final_metrics = {}
        
        for phase, phase_metrics in self.metrics.items():
            for metric_name, values in phase_metrics.items():
                if values:
                    key = f'{phase}_{metric_name}_final'
                    final_metrics[key] = values[-1]['value']
        
        return final_metrics


class TensorBoardLogger(Logger):
    """Logger with TensorBoard support."""
    
    def __init__(
        self,
        log_dir: str,
        experiment_name: str,
        config: Optional[Dict] = None
    ):
        super().__init__(log_dir, experiment_name, config)
        
        # Initialize TensorBoard writer
        self.writer = SummaryWriter(self.log_path)
        
        # Log configuration
        if config:
            self.writer.add_text('config', json.dumps(config, indent=2), 0)
    
    def log_metrics(
        self,
        metrics: Dict[str, float],
        step: int,
        phase: str = 'train'
    ):
        """Log metrics to TensorBoard."""
        super().log_metrics(metrics, step, phase)
        
        # Log to TensorBoard
        for name, value in metrics.items():
            self.writer.add_scalar(f'{phase}/{name}', value, step)
    
    def log_images(
        self,
        images: torch.Tensor,
        step: int,
        tag: str = 'images',
        num_images: int = 8
    ):
        """Log images to TensorBoard."""
        # Select subset of images
        if images.shape[0] > num_images:
            images = images[:num_images]
        
        # Make grid
        self.writer.add_images(tag, images, step)
    
    def log_attention_maps(
        self,
        attention_maps: torch.Tensor,
        step: int,
        tag: str = 'attention'
    ):
        """Log attention maps as heatmaps."""
        # Convert to numpy
        if torch.is_tensor(attention_maps):
            attention_maps = attention_maps.detach().cpu().numpy()
        
        # Create figure
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        axes = axes.flatten()
        
        for i, ax in enumerate(axes):
            if i < attention_maps.shape[0]:
                sns.heatmap(attention_maps[i], ax=ax, cmap='hot', cbar=True)
                ax.set_title(f'Head {i}')
            else:
                ax.axis('off')
        
        plt.tight_layout()
        
        # Log figure
        self.writer.add_figure(tag, fig, step)
        plt.close(fig)
    
    def log_histogram(
        self,
        values: torch.Tensor,
        step: int,
        tag: str
    ):
        """Log histogram to TensorBoard."""
        self.writer.add_histogram(tag, values, step)
    
    def log_model_graph(self, model: torch.nn.Module, input_shape: tuple):
        """Log model architecture graph."""
        dummy_input = torch.zeros(input_shape)
        self.writer.add_graph(model, dummy_input)
    
    def close(self):
        """Close TensorBoard writer."""
        self.writer.close()
        self.save_metrics()
        
        # Save summary
        summary = self.get_summary()
        summary_path = self.log_path / 'summary.json'
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)


class WandBLogger(Logger):
    """Logger with Weights & Biases support."""
    
    def __init__(
        self,
        log_dir: str,
        experiment_name: str,
        config: Optional[Dict] = None,
        project: Optional[str] = None,
        entity: Optional[str] = None
    ):
        super().__init__(log_dir, experiment_name, config)
        
        if not WANDB_AVAILABLE:
            raise ImportError("wandb is not installed. Install with: pip install wandb")
        
        # Initialize W&B
        self.run = wandb.init(
            project=project or 'mavl-lung-disease',
            entity=entity,
            name=experiment_name,
            config=config,
            dir=log_dir
        )
    
    def log_metrics(
        self,
        metrics: Dict[str, float],
        step: int,
        phase: str = 'train'
    ):
        """Log metrics to W&B."""
        super().log_metrics(metrics, step, phase)
        
        # Prepare metrics for W&B
        wandb_metrics = {
            f'{phase}/{name}': value 
            for name, value in metrics.items()
        }
        wandb_metrics['step'] = step
        
        # Log to W&B
        wandb.log(wandb_metrics)
    
    def log_images(
        self,
        images: torch.Tensor,
        step: int,
        tag: str = 'images',
        captions: Optional[List[str]] = None
    ):
        """Log images to W&B."""
        # Convert to numpy
        if torch.is_tensor(images):
            images = images.detach().cpu().numpy()
        
        # Create W&B images
        wandb_images = []
        for i, img in enumerate(images):
            caption = captions[i] if captions else f'Image {i}'
            wandb_images.append(wandb.Image(img, caption=caption))
        
        # Log to W&B
        wandb.log({tag: wandb_images, 'step': step})
    
    def log_table(
        self,
        data: Dict[str, List],
        tag: str = 'results'
    ):
        """Log table data to W&B."""
        table = wandb.Table(columns=list(data.keys()))
        
        # Add rows
        num_rows = len(next(iter(data.values())))
        for i in range(num_rows):
            row = [data[col][i] for col in data.keys()]
            table.add_data(*row)
        
        # Log table
        wandb.log({tag: table})
    
    def save_checkpoint(
        self,
        model_state: Dict,
        optimizer_state: Dict,
        epoch: int,
        metrics: Dict[str, float],
        is_best: bool = False
    ):
        """Save checkpoint and upload to W&B."""
        super().save_checkpoint(
            model_state, optimizer_state, epoch, metrics, is_best
        )
        
        # Upload checkpoint to W&B
        if is_best:
            checkpoint_path = self.log_path / 'best_checkpoint.pth'
            wandb.save(str(checkpoint_path))
    
    def close(self):
        """Close W&B run."""
        self.save_metrics()
        
        # Save summary
        summary = self.get_summary()
        wandb.log(summary)
        
        # Finish run
        wandb.finish()


def get_logger(
    log_dir: str,
    experiment_name: str,
    config: Optional[Dict] = None,
    use_tensorboard: bool = True,
    use_wandb: bool = False,
    **kwargs
) -> Logger:
    """
    Get appropriate logger based on configuration.
    
    Args:
        log_dir: Directory for logs
        experiment_name: Experiment name
        config: Configuration dictionary
        use_tensorboard: Whether to use TensorBoard
        use_wandb: Whether to use W&B
        **kwargs: Additional arguments for specific loggers
        
    Returns:
        Logger instance
    """
    if use_wandb:
        return WandBLogger(log_dir, experiment_name, config, **kwargs)
    elif use_tensorboard:
        return TensorBoardLogger(log_dir, experiment_name, config)
    else:
        return Logger(log_dir, experiment_name, config)