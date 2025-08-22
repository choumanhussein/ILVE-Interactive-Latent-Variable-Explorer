"""
β-VAE Implementation for Disentangled Representation Learning

β-VAE modifies the standard VAE loss by weighting the KL divergence term:
L = E[log p(x|z)] - β * KL(q(z|x) || p(z))

When β > 1, the model is encouraged to learn more disentangled representations
where each latent dimension captures a specific factor of variation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional

from .base_vae import VAE, Encoder, Decoder


class BetaVAE(VAE):
    """
    β-VAE model that extends the standard VAE with controllable disentanglement.
    
    The β parameter controls the trade-off between reconstruction quality and
    disentanglement. Higher β values encourage more disentangled representations
    but may reduce reconstruction quality.
    """
    
    def __init__(
        self, 
        input_dim: int = 784, 
        hidden_dims: List[int] = [512, 256], 
        latent_dim: int = 10,  # Use higher dim for disentanglement
        beta: float = 4.0,     # β > 1 for disentanglement
        capacity: float = 0.0, # Capacity parameter for β-VAE variants
        capacity_change_duration: int = 100000,
        capacity_max: float = 25.0
    ):
        super(BetaVAE, self).__init__(input_dim, hidden_dims, latent_dim, beta)
        
        self.capacity = capacity
        self.capacity_change_duration = capacity_change_duration
        self.capacity_max = capacity_max
        self.training_step = 0
        
    def beta_vae_loss(
        self, 
        reconstruction: torch.Tensor, 
        target: torch.Tensor, 
        mu: torch.Tensor, 
        logvar: torch.Tensor,
        beta: Optional[float] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        β-VAE loss with optional capacity scheduling.
        
        Args:
            reconstruction: Reconstructed input [batch_size, input_dim]
            target: Original input [batch_size, input_dim]
            mu: Latent mean [batch_size, latent_dim]
            logvar: Latent log variance [batch_size, latent_dim]
            beta: β weight for KL term (uses self.beta if None)
            
        Returns:
            total_loss: Combined loss
            recon_loss: Reconstruction loss
            kl_loss: KL divergence
            capacity_loss: Capacity-regularized KL loss
        """
        if beta is None:
            beta = self.beta
            
        batch_size = reconstruction.size(0)
        
        # Reconstruction loss
        recon_loss = F.binary_cross_entropy(
            reconstruction, target, reduction='sum'
        ) / batch_size
        
        # KL divergence
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / batch_size
        
        # Capacity scheduling (for β-VAE variants)
        if self.capacity > 0:
            # Linear capacity increase during training
            current_capacity = min(
                self.capacity_max,
                self.capacity + (self.capacity_max - self.capacity) * 
                self.training_step / self.capacity_change_duration
            )
            capacity_loss = beta * torch.abs(kl_loss - current_capacity)
        else:
            capacity_loss = beta * kl_loss
        
        total_loss = recon_loss + capacity_loss
        
        self.training_step += 1
        
        return total_loss, recon_loss, kl_loss, capacity_loss
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass through β-VAE."""
        return super().forward(x)


class ControlledBetaVAE(BetaVAE):
    """
    β-VAE with controlled generation capabilities.
    
    This variant allows for controlled manipulation of specific latent dimensions
    to analyze disentanglement properties.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def controlled_generation(
        self, 
        base_z: torch.Tensor, 
        dim_to_vary: int, 
        values: torch.Tensor,
        device: str = 'cpu'
    ) -> torch.Tensor:
        """
        Generate images by varying a specific latent dimension.
        
        Args:
            base_z: Base latent code [1, latent_dim]
            dim_to_vary: Which dimension to vary
            values: Values to use for the varying dimension [num_values]
            device: Device to run on
            
        Returns:
            Generated images [num_values, input_dim]
        """
        self.eval()
        
        with torch.no_grad():
            generated_images = []
            
            for value in values:
                # Copy base latent code and modify specific dimension
                z = base_z.clone()
                z[0, dim_to_vary] = value
                
                # Generate image
                generated = self.decode(z)
                generated_images.append(generated)
            
            return torch.cat(generated_images, dim=0)
    
    def latent_traversal(
        self, 
        num_samples: int = 8,
        num_steps: int = 11,
        latent_range: Tuple[float, float] = (-3, 3),
        device: str = 'cpu'
    ) -> Dict[int, torch.Tensor]:
        """
        Perform latent traversal to analyze disentanglement.
        
        For each latent dimension, generate images by varying that dimension
        while keeping others at their mean values.
        
        Args:
            num_samples: Number of base samples to use
            num_steps: Number of steps in traversal
            latent_range: Range of values to traverse
            device: Device to run on
            
        Returns:
            Dictionary mapping dimension index to generated images
        """
        self.eval()
        
        traversals = {}
        values = torch.linspace(latent_range[0], latent_range[1], num_steps)
        
        with torch.no_grad():
            # Generate base latent codes (zeros or random)
            base_codes = torch.zeros(num_samples, self.latent_dim, device=device)
            
            for dim in range(self.latent_dim):
                dim_traversals = []
                
                for sample_idx in range(num_samples):
                    base_z = base_codes[sample_idx:sample_idx+1]
                    generated = self.controlled_generation(
                        base_z, dim, values, device
                    )
                    dim_traversals.append(generated)
                
                traversals[dim] = torch.stack(dim_traversals)
        
        return traversals
    
    def analyze_disentanglement_mnist(
        self, 
        test_loader,
        device: str = 'cpu'
    ) -> Dict[str, float]:
        """
        Analyze disentanglement properties on MNIST.
        
        This function computes metrics to evaluate how well different
        latent dimensions capture different factors of variation.
        
        Args:
            test_loader: DataLoader with test data
            device: Device to run on
            
        Returns:
            Dictionary with disentanglement metrics
        """
        self.eval()
        
        # Collect latent representations for different digits
        latent_by_digit = {i: [] for i in range(10)}
        
        with torch.no_grad():
            for batch_idx, (data, labels) in enumerate(test_loader):
                if batch_idx > 20:  # Limit for efficiency
                    break
                    
                data = data.to(device)
                mu, _ = self.encode(data)
                
                for i, label in enumerate(labels):
                    latent_by_digit[label.item()].append(mu[i].cpu())
        
        # Convert to tensors
        for digit in latent_by_digit:
            if latent_by_digit[digit]:
                latent_by_digit[digit] = torch.stack(latent_by_digit[digit])
        
        # Compute basic disentanglement metrics
        metrics = {}
        
        # 1. Variance explained by each dimension for each digit
        dim_variances = {}
        for dim in range(self.latent_dim):
            dim_var_by_digit = []
            for digit in range(10):
                if len(latent_by_digit[digit]) > 0:
                    var = torch.var(latent_by_digit[digit][:, dim]).item()
                    dim_var_by_digit.append(var)
            
            if dim_var_by_digit:
                dim_variances[dim] = {
                    'mean_var': np.mean(dim_var_by_digit),
                    'std_var': np.std(dim_var_by_digit),
                    'max_var': np.max(dim_var_by_digit),
                    'min_var': np.min(dim_var_by_digit)
                }
        
        metrics['dimension_variances'] = dim_variances
        
        # 2. Mutual information between dimensions and digit labels (simplified)
        # This is a simplified version - full MI calculation would be more complex
        digit_means = {}
        for digit in range(10):
            if len(latent_by_digit[digit]) > 0:
                digit_means[digit] = torch.mean(latent_by_digit[digit], dim=0)
        
        if digit_means:
            # Calculate separation between digit means in each dimension
            dim_separations = []
            for dim in range(self.latent_dim):
                means_in_dim = [digit_means[d][dim].item() for d in digit_means.keys()]
                separation = np.std(means_in_dim)
                dim_separations.append(separation)
            
            metrics['dimension_separations'] = dim_separations
            metrics['avg_separation'] = np.mean(dim_separations)
            metrics['max_separation'] = np.max(dim_separations)
        
        return metrics


def create_beta_vae_variants() -> Dict[str, BetaVAE]:
    """Create different β-VAE variants for comparison."""
    variants = {
        'standard_vae': BetaVAE(latent_dim=10, beta=1.0),
        'beta_vae_2': BetaVAE(latent_dim=10, beta=2.0),
        'beta_vae_4': BetaVAE(latent_dim=10, beta=4.0),
        'beta_vae_8': BetaVAE(latent_dim=10, beta=8.0),
        'controlled_beta_vae': ControlledBetaVAE(latent_dim=10, beta=4.0)
    }
    return variants


def test_beta_vae():
    """Test β-VAE implementation."""
    print("Testing β-VAE implementation...")
    
    # Create model
    model = BetaVAE(latent_dim=10, beta=4.0)
    
    # Test forward pass
    batch_size = 16
    x = torch.rand(batch_size, 784)
    
    reconstruction, mu, logvar, z = model(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Reconstruction shape: {reconstruction.shape}")
    print(f"Latent shape: {z.shape}")
    
    # Test β-VAE loss
    total_loss, recon_loss, kl_loss, capacity_loss = model.beta_vae_loss(
        reconstruction, x, mu, logvar
    )
    
    print(f"β-VAE Loss components:")
    print(f"  Total: {total_loss.item():.4f}")
    print(f"  Reconstruction: {recon_loss.item():.4f}")
    print(f"  KL: {kl_loss.item():.4f}")
    print(f"  Capacity: {capacity_loss.item():.4f}")
    
    # Test controlled generation
    controlled_model = ControlledBetaVAE(latent_dim=10, beta=4.0)
    base_z = torch.zeros(1, 10)
    values = torch.linspace(-2, 2, 5)
    
    generated = controlled_model.controlled_generation(
        base_z, dim_to_vary=0, values=values
    )
    print(f"Controlled generation shape: {generated.shape}")
    
    # Test latent traversal
    traversals = controlled_model.latent_traversal(num_samples=2, num_steps=5)
    print(f"Latent traversal dimensions: {list(traversals.keys())}")
    print(f"Traversal shape for dim 0: {traversals[0].shape}")
    
    print("✓ β-VAE test passed!")
    
    return model


if __name__ == "__main__":
    model = test_beta_vae()