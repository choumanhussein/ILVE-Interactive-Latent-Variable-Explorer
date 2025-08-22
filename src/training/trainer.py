"""
VAE Training Infrastructure with Weights & Biases integration.

This module provides a comprehensive training framework for VAE models including:
- Training and validation loops
- Loss tracking and visualization
- Model checkpointing
- Experiment logging with W&B
- Real-time monitoring of latent space evolution
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import wandb
import os
import time
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import yaml
from typing import Dict, Optional, Tuple

from ..models.base_vae import VAE, vae_loss
from ..evaluation.visualization import plot_latent_space, plot_reconstructions, plot_generated_samples


class VAETrainer:
    """
    Comprehensive trainer for VAE models with experiment tracking.
    """
    
    def __init__(
        self,
        model: VAE,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: Dict,
        device: str = 'cpu',
        use_wandb: bool = True
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        self.use_wandb = use_wandb
        
        # Training components
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()
        
        # Tracking
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.train_losses = []
        self.val_losses = []
        
        # Paths
        self.checkpoint_dir = config['paths']['checkpoint_dir']
        self.results_dir = config['paths']['results_dir']
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Initialize wandb
        if self.use_wandb:
            self._init_wandb()
    
    def _create_optimizer(self) -> optim.Optimizer:
        """Create optimizer based on config."""
        optimizer_name = self.config['training']['optimizer'].lower()
        lr = float(self.config['training']['learning_rate'])  # Ensure it's a float
        
        if optimizer_name == 'adam':
            return optim.Adam(self.model.parameters(), lr=lr)
        elif optimizer_name == 'adamw':
            return optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-4)
        elif optimizer_name == 'sgd':
            return optim.SGD(self.model.parameters(), lr=lr, momentum=0.9)
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")
    
    def _create_scheduler(self) -> Optional[optim.lr_scheduler._LRScheduler]:
        """Create learning rate scheduler based on config."""
        scheduler_name = self.config['training'].get('scheduler', None)
        
        if scheduler_name == 'reduce_on_plateau':
            return optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, 
                mode='min', 
                factor=0.5, 
                patience=int(self.config['training'].get('patience', 10))
                # Removed 'verbose' parameter for compatibility
            )
        elif scheduler_name == 'cosine':
            return optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, 
                T_max=int(self.config['training']['epochs'])
            )
        else:
            return None
    
    def _init_wandb(self):
        """Initialize Weights & Biases logging."""
        wandb.init(
            project=self.config['logging']['wandb_project'],
            name=f"vae_latent_{self.config['model']['latent_dim']}d_beta_{self.config['loss']['beta']}",
            config=self.config,
            tags=['vae', 'continuous-latent', 'mnist']
        )
        
        # Watch model for gradient tracking
        wandb.watch(self.model, log='all', log_freq=100)
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        epoch_losses = {'total': 0, 'reconstruction': 0, 'kl': 0}
        num_batches = len(self.train_loader)
        
        progress_bar = tqdm(self.train_loader, desc=f'Epoch {self.current_epoch}')
        
        for batch_idx, (data, _) in enumerate(progress_bar):
            data = data.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            reconstruction, mu, logvar, z = self.model(data)
            
            # Calculate loss
            total_loss, recon_loss, kl_loss = vae_loss(
                reconstruction, data, mu, logvar, 
                beta=float(self.config['loss']['beta'])  # Ensure float
            )
            
            # Backward pass
            total_loss.backward()
            self.optimizer.step()
            
            # Track losses
            batch_size = data.size(0)
            epoch_losses['total'] += total_loss.item()
            epoch_losses['reconstruction'] += recon_loss.item()
            epoch_losses['kl'] += kl_loss.item()
            
            # Update progress bar
            progress_bar.set_postfix({
                'Loss': f"{total_loss.item()/batch_size:.4f}",
                'Recon': f"{recon_loss.item()/batch_size:.4f}",
                'KL': f"{kl_loss.item()/batch_size:.4f}"
            })
            
            # Log to wandb
            if self.use_wandb and batch_idx % self.config['logging']['log_interval'] == 0:
                wandb.log({
                    'train/batch_loss': total_loss.item() / batch_size,
                    'train/batch_recon_loss': recon_loss.item() / batch_size,
                    'train/batch_kl_loss': kl_loss.item() / batch_size,
                    'train/lr': self.optimizer.param_groups[0]['lr'],
                    'epoch': self.current_epoch,
                    'batch': batch_idx
                })
        
        # Average losses over epoch
        for key in epoch_losses:
            epoch_losses[key] /= len(self.train_loader.dataset)
        
        return epoch_losses
    
    def validate_epoch(self) -> Dict[str, float]:
        """Validate for one epoch."""
        self.model.eval()
        epoch_losses = {'total': 0, 'reconstruction': 0, 'kl': 0}
        
        with torch.no_grad():
            for data, _ in self.val_loader:
                data = data.to(self.device)
                
                # Forward pass
                reconstruction, mu, logvar, z = self.model(data)
                
                # Calculate loss
                total_loss, recon_loss, kl_loss = vae_loss(
                    reconstruction, data, mu, logvar,
                    beta=float(self.config['loss']['beta'])  # Ensure float
                )
                
                # Track losses
                epoch_losses['total'] += total_loss.item()
                epoch_losses['reconstruction'] += recon_loss.item()
                epoch_losses['kl'] += kl_loss.item()
        
        # Average losses over epoch
        for key in epoch_losses:
            epoch_losses[key] /= len(self.val_loader.dataset)
        
        return epoch_losses
    
    def save_checkpoint(self, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': self.config
        }
        
        # Save latest checkpoint
        latest_path = os.path.join(self.checkpoint_dir, 'latest.pth')
        torch.save(checkpoint, latest_path)
        
        # Save best checkpoint
        if is_best:
            best_path = os.path.join(self.checkpoint_dir, 'best.pth')
            torch.save(checkpoint, best_path)
            print(f"✓ New best model saved with val_loss: {self.best_val_loss:.4f}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model from checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.best_val_loss = checkpoint['best_val_loss']
        
        print(f"✓ Loaded checkpoint from epoch {self.current_epoch}")
    
    def visualize_progress(self, val_losses: Dict[str, float]):
        """Create and log visualizations during training."""
        self.model.eval()
        
        with torch.no_grad():
            # Get a batch of validation data
            val_data, val_labels = next(iter(self.val_loader))
            val_data = val_data.to(self.device)
            
            # Generate reconstructions
            reconstruction, mu, logvar, z = self.model(val_data[:16])
            
            # Generate new samples
            generated = self.model.generate(num_samples=16, device=self.device)
            
            # Create visualizations
            fig_recon = plot_reconstructions(
                val_data[:16].cpu(), 
                reconstruction.cpu(),
                title=f'Reconstructions - Epoch {self.current_epoch}'
            )
            
            fig_generated = plot_generated_samples(
                generated.cpu(),
                title=f'Generated Samples - Epoch {self.current_epoch}'
            )
            
            fig_latent = None
            if mu.shape[1] == 2:  # Only create latent space plot for 2D
                fig_latent = plot_latent_space(
                    mu.cpu(), 
                    val_labels[:16],
                    title=f'Latent Space - Epoch {self.current_epoch}'
                )
            
            # Log to wandb
            if self.use_wandb:
                wandb_log = {
                    'visualizations/reconstructions': wandb.Image(fig_recon),
                    'visualizations/generated': wandb.Image(fig_generated),
                    'epoch': self.current_epoch
                }
                if fig_latent is not None:
                    wandb_log['visualizations/latent_space'] = wandb.Image(fig_latent)
                wandb.log(wandb_log)
            
            # Save locally
            results_epoch_dir = os.path.join(self.results_dir, f'epoch_{self.current_epoch:03d}')
            os.makedirs(results_epoch_dir, exist_ok=True)
            
            fig_recon.savefig(os.path.join(results_epoch_dir, 'reconstructions.png'), dpi=150, bbox_inches='tight')
            fig_generated.savefig(os.path.join(results_epoch_dir, 'generated.png'), dpi=150, bbox_inches='tight')
            if fig_latent is not None:
                fig_latent.savefig(os.path.join(results_epoch_dir, 'latent_space.png'), dpi=150, bbox_inches='tight')
            
            plt.close(fig_recon)
            plt.close(fig_generated) 
            if fig_latent is not None:
                plt.close(fig_latent)
    
    def train(self, num_epochs: Optional[int] = None) -> Dict[str, list]:
        """Main training loop."""
        if num_epochs is None:
            num_epochs = self.config['training']['epochs']
        
        print(f"Starting training for {num_epochs} epochs...")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"Training on: {self.device}")
        
        start_time = time.time()
        
        for epoch in range(num_epochs):
            self.current_epoch = epoch
            
            # Training
            train_losses = self.train_epoch()
            self.train_losses.append(train_losses)
            
            # Validation
            val_losses = self.validate_epoch()
            self.val_losses.append(val_losses)
            
            # Learning rate scheduling
            if self.scheduler:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_losses['total'])
                else:
                    self.scheduler.step()
            
            # Check if best model
            is_best = val_losses['total'] < self.best_val_loss
            if is_best:
                self.best_val_loss = val_losses['total']
            
            # Logging
            print(f"Epoch {epoch}/{num_epochs-1}")
            print(f"  Train Loss: {train_losses['total']:.4f} (Recon: {train_losses['reconstruction']:.4f}, KL: {train_losses['kl']:.4f})")
            print(f"  Val Loss:   {val_losses['total']:.4f} (Recon: {val_losses['reconstruction']:.4f}, KL: {val_losses['kl']:.4f})")
            
            if self.use_wandb:
                wandb.log({
                    'epoch': epoch,
                    'train/total_loss': train_losses['total'],
                    'train/recon_loss': train_losses['reconstruction'],
                    'train/kl_loss': train_losses['kl'],
                    'val/total_loss': val_losses['total'],
                    'val/recon_loss': val_losses['reconstruction'],
                    'val/kl_loss': val_losses['kl'],
                    'val/best_loss': self.best_val_loss
                })
            
            # Save checkpoint
            if epoch % self.config['logging']['save_interval'] == 0 or is_best:
                self.save_checkpoint(is_best)
            
            # Visualizations
            if epoch % (self.config['logging']['save_interval'] * 2) == 0:
                self.visualize_progress(val_losses)
        
        training_time = time.time() - start_time
        print(f"\n✓ Training completed in {training_time/60:.1f} minutes")
        print(f"Best validation loss: {self.best_val_loss:.4f}")
        
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'best_val_loss': self.best_val_loss
        }