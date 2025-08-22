"""
Training Script for Basic VAE on MNIST

This script trains a standard VAE on MNIST and demonstrates:
- Continuous latent variable learning
- Reconstruction and generation capabilities
- Latent space organization
- Real-time training monitoring with W&B
"""

import torch
import yaml
import sys
import os
import argparse
import wandb

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.data.mnist_loader import MNISTDataModule
from src.models.base_vae import VAE
from src.training.trainer import VAETrainer
from src.evaluation.visualization import plot_loss_curves, plot_latent_space_grid


def load_config(config_path: str = 'configs/base_config.yaml'):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def main():
    """Main training function."""
    print("🚀 Starting VAE Training on MNIST")
    print("=" * 50)
    
    # Parse arguments
    parser = argparse.ArgumentParser(description='Train VAE on MNIST')
    parser.add_argument('--config', type=str, default='configs/base_config.yaml',
                       help='Path to config file')
    parser.add_argument('--no-wandb', action='store_true',
                       help='Disable Weights & Biases logging')
    parser.add_argument('--epochs', type=int, default=None,
                       help='Number of training epochs (overrides config)')
    parser.add_argument('--latent-dim', type=int, default=None,
                       help='Latent dimension (overrides config)')
    parser.add_argument('--beta', type=float, default=None,
                       help='Beta value for β-VAE (overrides config)')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    print(f"✓ Loaded config: {config['project']['name']}")
    
    # Override config with command line arguments
    if args.epochs is not None:
        config['training']['epochs'] = args.epochs
    if args.latent_dim is not None:
        config['model']['latent_dim'] = args.latent_dim
    if args.beta is not None:
        config['loss']['beta'] = args.beta
    
    use_wandb = not args.no_wandb
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"✓ Using device: {device}")
    
    # Initialize data
    print("\n📊 Preparing data...")
    data_module = MNISTDataModule(
        batch_size=config['data']['batch_size'],
        num_workers=config['data']['num_workers'],
        pin_memory=config['data']['pin_memory']
    )
    
    data_module.prepare_data()
    data_module.setup()
    
    train_loader = data_module.train_dataloader()
    val_loader = data_module.val_dataloader()
    
    print(f"✓ Training samples: {len(data_module.train_dataset):,}")
    print(f"✓ Validation samples: {len(data_module.val_dataset):,}")
    print(f"✓ Batch size: {config['data']['batch_size']}")
    
    # Initialize model
    print("\n🧠 Creating model...")
    model = VAE(
        input_dim=config['model']['input_dim'],
        hidden_dims=config['model']['hidden_dims'],
        latent_dim=config['model']['latent_dim'],
        beta=config['loss']['beta']
    )
    
    print(f"✓ Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    print(f"✓ Latent dimensions: {config['model']['latent_dim']}")
    print(f"✓ β value: {config['loss']['beta']}")
    
    # Initialize trainer
    print("\n🏋️ Setting up trainer...")
    trainer = VAETrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=device,
        use_wandb=use_wandb
    )
    
    if use_wandb:
        print("✓ Weights & Biases initialized")
        print(f"✓ Project: {config['logging']['wandb_project']}")
    else:
        print("✓ Training without W&B logging")
    
    # Start training
    print(f"\n🚂 Starting training for {config['training']['epochs']} epochs...")
    print("-" * 50)
    
    try:
        results = trainer.train()
        
        print("\n🎉 Training completed successfully!")
        print(f"✓ Best validation loss: {results['best_val_loss']:.4f}")
        
        # Create final visualizations
        print("\n📊 Creating final visualizations...")
        
        # Loss curves
        fig_losses = plot_loss_curves(
            results['train_losses'],
            results['val_losses'],
            title="VAE Training Progress"
        )
        fig_losses.savefig('training_curves.png', dpi=150, bbox_inches='tight')
        print("✓ Saved training curves to 'training_curves.png'")
        
        # Latent space grid (only for 2D latent space)
        if config['model']['latent_dim'] == 2:
            model.to(device)
            fig_grid = plot_latent_space_grid(
                model, device=device,
                title="Final Latent Space Grid"
            )
            fig_grid.savefig('latent_space_grid.png', dpi=150, bbox_inches='tight')
            print("✓ Saved latent space grid to 'latent_space_grid.png'")
        
        # Log final results to wandb
        if use_wandb:
            wandb.log({
                "final/training_curves": wandb.Image(fig_losses),
                "final/best_val_loss": results['best_val_loss']
            })
            
            if config['model']['latent_dim'] == 2:
                wandb.log({"final/latent_space_grid": wandb.Image(fig_grid)})
            
            wandb.finish()
        
        # Save final model
        final_model_path = os.path.join(config['paths']['checkpoint_dir'], 'final_model.pth')
        torch.save({
            'model_state_dict': model.state_dict(),
            'config': config,
            'results': results
        }, final_model_path)
        print(f"✓ Saved final model to '{final_model_path}'")
        
        print("\n🎯 Training Summary:")
        print(f"   • Epochs trained: {config['training']['epochs']}")
        print(f"   • Best validation loss: {results['best_val_loss']:.4f}")
        print(f"   • Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        print(f"   • Latent dimensions: {config['model']['latent_dim']}")
        print(f"   • β value: {config['loss']['beta']}")
        
    except KeyboardInterrupt:
        print("\n⚠️ Training interrupted by user")
        if use_wandb:
            wandb.finish()
        
    except Exception as e:
        print(f"\n❌ Training failed with error: {e}")
        if use_wandb:
            wandb.finish()
        raise


def quick_test():
    """Quick test to verify everything works before full training."""
    print("🔧 Running quick test...")
    
    # Load config
    config = load_config()
    
    # Quick data test
    data_module = MNISTDataModule(batch_size=32, num_workers=0, pin_memory=False)
    data_module.prepare_data()
    data_module.setup()
    
    # Quick model test
    model = VAE(latent_dim=2)
    train_loader = data_module.train_dataloader()
    batch = next(iter(train_loader))
    images, labels = batch
    
    reconstruction, mu, logvar, z = model(images)
    print(f"✓ Model forward pass successful: {reconstruction.shape}")
    
    # Quick trainer test
    trainer = VAETrainer(
        model=model,
        train_loader=train_loader,
        val_loader=data_module.val_dataloader(),
        config=config,
        device='cpu',
        use_wandb=False
    )
    
    print("✓ All components working correctly!")
    return True


if __name__ == "__main__":
    # Check if user wants to run a quick test first
    if len(sys.argv) > 1 and sys.argv[1] == '--test':
        quick_test()
    else:
        main()