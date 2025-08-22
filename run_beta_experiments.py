"""
β-VAE Experiments: Exploring Continuous Latent Variables and Disentanglement

This script runs comprehensive experiments comparing different β values to demonstrate:
- How β controls the trade-off between reconstruction and disentanglement
- The effect of continuous latent variable regularization
- Latent space organization and interpretability

Run this to see the full power of continuous latent variables!
"""

import torch
import yaml
import sys
import os
import argparse
import wandb
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.data.mnist_loader import MNISTDataModule
from src.models.beta_vae import BetaVAE, ControlledBetaVAE
from src.training.trainer import VAETrainer
from src.evaluation.disentanglement import (
    create_disentanglement_report, 
    plot_beta_comparison,
    plot_latent_traversal,
    compute_disentanglement_score
)
from src.evaluation.visualization import plot_latent_space_grid


def load_config(config_path: str = 'configs/base_config.yaml'):
    """Load and modify config for β-VAE experiments."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Modify for β-VAE experiments
    config['model']['latent_dim'] = 2   # Use 2D for easy visualization
    config['training']['epochs'] = 30   # Shorter training for comparisons
    config['data']['batch_size'] = 128
    
    return config


def train_beta_vae(
    beta: float,
    config: Dict,
    data_module: MNISTDataModule,
    device: str,
    use_wandb: bool = False,
    experiment_name: str = "beta_vae"
) -> Tuple[BetaVAE, Dict]:
    """
    Train a single β-VAE model.
    
    Args:
        beta: β value for the experiment
        config: Configuration dictionary
        data_module: Data module
        device: Device to train on
        use_wandb: Whether to use W&B logging
        experiment_name: Name for the experiment
        
    Returns:
        Trained model and training results
    """
    print(f"\n🧠 Training β-VAE with β = {beta}")
    print("-" * 40)
    
    # Create model
    model = BetaVAE(
        input_dim=config['model']['input_dim'],
        hidden_dims=config['model']['hidden_dims'],
        latent_dim=config['model']['latent_dim'],
        beta=beta
    )
    
    # Update config for this experiment
    exp_config = config.copy()
    exp_config['loss']['beta'] = beta
    
    # Create trainer
    trainer = VAETrainer(
        model=model,
        train_loader=data_module.train_dataloader(),
        val_loader=data_module.val_dataloader(),
        config=exp_config,
        device=device,
        use_wandb=use_wandb
    )
    
    # Override wandb run name if using wandb
    if use_wandb:
        wandb.run.name = f"{experiment_name}_beta_{beta}"
        wandb.run.tags = ['beta-vae', 'disentanglement', f'beta_{beta}']
    
    # Train
    results = trainer.train()
    
    print(f"✓ Training completed for β = {beta}")
    print(f"  Best validation loss: {results['best_val_loss']:.4f}")
    
    return model, results


def analyze_disentanglement(
    models: Dict[float, BetaVAE],
    data_module: MNISTDataModule,
    device: str,
    save_dir: str = "beta_analysis"
) -> Dict[float, Dict]:
    """
    Analyze disentanglement properties of trained models.
    
    Args:
        models: Dictionary mapping β values to trained models
        data_module: Data module
        device: Device to run on
        save_dir: Directory to save results
        
    Returns:
        Dictionary of analysis results for each β value
    """
    print("\n📊 Analyzing Disentanglement Properties")
    print("=" * 50)
    
    os.makedirs(save_dir, exist_ok=True)
    test_loader = data_module.test_dataloader()
    all_results = {}
    
    for beta, model in models.items():
        print(f"\nAnalyzing β = {beta}...")
        
        model.to(device)
        model.eval()
        
        # Create analysis directory for this β
        beta_dir = os.path.join(save_dir, f"beta_{beta}")
        os.makedirs(beta_dir, exist_ok=True)
        
        # Compute disentanglement metrics
        metrics = compute_disentanglement_score(model, test_loader, device=device)
        
        # Create comprehensive report
        if hasattr(model, 'latent_traversal'):
            # Convert to ControlledBetaVAE for traversal analysis
            controlled_model = ControlledBetaVAE(
                input_dim=model.input_dim,
                latent_dim=model.latent_dim,
                beta=beta
            )
            controlled_model.load_state_dict(model.state_dict())
            controlled_model.to(device)
            
            report = create_disentanglement_report(
                controlled_model, test_loader, device=device, save_path=beta_dir
            )
        else:
            report = create_disentanglement_report(
                model, test_loader, device=device, save_path=beta_dir
            )
        
        all_results[beta] = {
            'metrics': metrics,
            'report': report
        }
        
        print(f"✓ Analysis complete for β = {beta}")
        print(f"  Avg separability: {metrics.get('avg_separability', 0):.4f}")
        print(f"  Effective dimensions: {metrics.get('effective_dimensions', 0)}")
    
    return all_results


def create_comparison_visualizations(
    results: Dict[float, Dict],
    save_dir: str = "beta_analysis"
):
    """Create visualizations comparing different β values."""
    print("\n📈 Creating Comparison Visualizations")
    print("-" * 40)
    
    # Extract metrics for comparison
    beta_values = sorted(results.keys())
    
    # Plot comparison of key metrics
    metrics_to_compare = [
        'avg_separability',
        'max_separability', 
        'effective_dimensions',
        'variance_entropy'
    ]
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()
    
    for i, metric in enumerate(metrics_to_compare):
        if i >= len(axes):
            break
            
        values = []
        for beta in beta_values:
            metric_val = results[beta]['metrics'].get(metric, 0)
            values.append(metric_val)
        
        axes[i].plot(beta_values, values, 'bo-', linewidth=2, markersize=8)
        axes[i].set_xlabel('β Value')
        axes[i].set_ylabel(metric.replace('_', ' ').title())
        axes[i].set_title(f'{metric.replace("_", " ").title()} vs β')
        axes[i].grid(True, alpha=0.3)
        
        # Highlight standard VAE (β=1)
        if 1.0 in beta_values:
            idx = beta_values.index(1.0)
            axes[i].plot(1.0, values[idx], 'ro', markersize=12)
    
    plt.suptitle('β-VAE Disentanglement Comparison', fontsize=16)
    plt.tight_layout()
    
    comparison_path = os.path.join(save_dir, 'beta_comparison.png')
    plt.savefig(comparison_path, dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"✓ Saved comparison plot to {comparison_path}")
    
    # Create summary table
    print(f"\n{'β Value':<8} {'Avg Sep':<10} {'Max Sep':<10} {'Eff Dims':<10} {'Var Entropy':<12}")
    print("-" * 55)
    
    for beta in beta_values:
        metrics = results[beta]['metrics']
        print(f"{beta:<8.1f} {metrics.get('avg_separability', 0):<10.4f} "
              f"{metrics.get('max_separability', 0):<10.4f} "
              f"{metrics.get('effective_dimensions', 0):<10.0f} "
              f"{metrics.get('variance_entropy', 0):<12.4f}")


def main():
    """Main experiment function."""
    parser = argparse.ArgumentParser(description='Run β-VAE experiments')
    parser.add_argument('--beta-values', nargs='+', type=float, 
                       default=[1.0, 2.0, 4.0, 6.0],
                       help='β values to experiment with')
    parser.add_argument('--epochs', type=int, default=25,
                       help='Number of epochs to train each model')
    parser.add_argument('--no-wandb', action='store_true',
                       help='Disable W&B logging')
    parser.add_argument('--quick', action='store_true',
                       help='Quick test with fewer epochs and β values')
    parser.add_argument('--analysis-only', action='store_true',
                       help='Skip training, only run analysis (requires saved models)')
    
    args = parser.parse_args()
    
    if args.quick:
        args.beta_values = [1.0, 4.0]
        args.epochs = 10
        print("🏃 Running quick experiment mode")
    
    print("🚀 β-VAE Disentanglement Experiments")
    print("=" * 50)
    print(f"β values: {args.beta_values}")
    print(f"Epochs per model: {args.epochs}")
    print(f"W&B logging: {not args.no_wandb}")
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Load config
    config = load_config()
    config['training']['epochs'] = args.epochs
    
    # Setup data
    print("\n📊 Preparing data...")
    data_module = MNISTDataModule(
        batch_size=config['data']['batch_size'],
        num_workers=config['data']['num_workers'],
        pin_memory=config['data']['pin_memory']
    )
    data_module.prepare_data()
    data_module.setup()
    
    models = {}
    
    if not args.analysis_only:
        # Train models for different β values
        print(f"\n🏋️ Training {len(args.beta_values)} β-VAE models...")
        
        for i, beta in enumerate(args.beta_values):
            print(f"\n[{i+1}/{len(args.beta_values)}] Training β = {beta}")
            
            model, results = train_beta_vae(
                beta=beta,
                config=config,
                data_module=data_module,
                device=device,
                use_wandb=not args.no_wandb,
                experiment_name="beta_comparison"
            )
            
            models[beta] = model
            
            # Save model
            model_path = f"experiments/checkpoints/beta_vae_{beta}.pth"
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            torch.save({
                'model_state_dict': model.state_dict(),
                'beta': beta,
                'config': config,
                'results': results
            }, model_path)
            
            print(f"✓ Saved model to {model_path}")
            
            if not args.no_wandb:
                wandb.finish()
    
    else:
        # Load existing models
        print("\n📂 Loading existing models...")
        for beta in args.beta_values:
            model_path = f"experiments/checkpoints/beta_vae_{beta}.pth"
            if os.path.exists(model_path):
                checkpoint = torch.load(model_path, map_location=device)
                
                model = BetaVAE(
                    input_dim=config['model']['input_dim'],
                    hidden_dims=config['model']['hidden_dims'],
                    latent_dim=config['model']['latent_dim'],
                    beta=beta
                )
                model.load_state_dict(checkpoint['model_state_dict'])
                models[beta] = model
                
                print(f"✓ Loaded model for β = {beta}")
            else:
                print(f"❌ Model not found for β = {beta} at {model_path}")
    
    if not models:
        print("❌ No models available for analysis!")
        return
    
    # Analyze disentanglement
    print(f"\n🔬 Analyzing disentanglement for {len(models)} models...")
    analysis_results = analyze_disentanglement(
        models, data_module, device, save_dir="experiments/beta_analysis"
    )
    
    # Create comparison visualizations
    create_comparison_visualizations(
        analysis_results, save_dir="experiments/beta_analysis"
    )
    
    print("\n🎉 Experiment completed!")
    print("\nKey Insights:")
    print("• Higher β values encourage more disentangled representations")
    print("• β-VAE trades off reconstruction quality for interpretability")
    print("• Continuous latent variables enable smooth interpolation")
    print("• Different β values lead to different latent space organizations")
    
    print(f"\nResults saved in: experiments/beta_analysis/")
    print("Check the generated visualizations to see disentanglement in action!")


def quick_demo():
    """Quick demonstration of β-VAE capabilities."""
    print("🎯 Quick β-VAE Demo")
    print("=" * 30)
    
    # Create a simple model for demonstration
    model = ControlledBetaVAE(latent_dim=10, beta=4.0)
    
    # Demonstrate controlled generation
    print("Demonstrating controlled generation...")
    
    # Create base latent code
    base_z = torch.zeros(1, 10)
    values = torch.linspace(-2, 2, 5)
    
    # Generate by varying first dimension
    generated = model.controlled_generation(base_z, dim_to_vary=0, values=values)
    print(f"Generated {generated.shape[0]} images by varying dimension 0")
    
    # Demonstrate latent traversal
    print("Demonstrating latent traversal...")
    traversals = model.latent_traversal(num_samples=2, num_steps=5)
    print(f"Created traversals for {len(traversals)} dimensions")
    
    print("✓ Demo completed! This shows the power of continuous latent variables.")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == '--demo':
        quick_demo()
    else:
        main()