"""
Basic Variational Autoencoder implementation for MNIST.

This module implements the core VAE architecture with:
- Encoder network (recognition model)
- Decoder network (generative model) 
- Reparameterization trick for continuous latent variables
- VAE loss function (reconstruction + KL divergence)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from dataclasses import dataclass
from typing import List, Optional

@dataclass
class VAEConfig:
    """Configuration class for VAE models."""
    input_dim: int = 784
    latent_dim: int = 2
    hidden_dims: List[int] = None
    learning_rate: float = 1e-3
    beta: float = 1.0
    
    def __post_init__(self):
        if self.hidden_dims is None:
            self.hidden_dims = [512, 256]

class Encoder(nn.Module):
    """
    Encoder network that maps input to latent space parameters.
    
    Takes 784D input (flattened MNIST) and outputs mean and log-variance
    of the latent distribution.
    """
    
    def __init__(self, input_dim=784, hidden_dims=[512, 256], latent_dim=2):
        super(Encoder, self).__init__()
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        
        # Build encoder layers
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            prev_dim = hidden_dim
            
        self.encoder = nn.Sequential(*layers)
        
        # Latent space parameters
        self.fc_mu = nn.Linear(prev_dim, latent_dim)
        self.fc_logvar = nn.Linear(prev_dim, latent_dim)
        
    def forward(self, x):
        """
        Forward pass through encoder.
        
        Args:
            x: Input tensor [batch_size, input_dim]
            
        Returns:
            mu: Mean of latent distribution [batch_size, latent_dim]
            logvar: Log variance of latent distribution [batch_size, latent_dim]
        """
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        
        return mu, logvar


class Decoder(nn.Module):
    """
    Decoder network that maps latent codes back to data space.
    
    Takes latent codes and reconstructs the original input.
    """
    
    def __init__(self, latent_dim=2, hidden_dims=[256, 512], output_dim=784):
        super(Decoder, self).__init__()
        
        self.latent_dim = latent_dim
        self.output_dim = output_dim
        
        # Build decoder layers (reverse of encoder)
        layers = []
        prev_dim = latent_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            prev_dim = hidden_dim
            
        # Final output layer
        layers.append(nn.Linear(prev_dim, output_dim))
        layers.append(nn.Sigmoid())  # Output in [0, 1] range
        
        self.decoder = nn.Sequential(*layers)
        
    def forward(self, z):
        """
        Forward pass through decoder.
        
        Args:
            z: Latent codes [batch_size, latent_dim]
            
        Returns:
            reconstruction: Reconstructed input [batch_size, output_dim]
        """
        return self.decoder(z)


class VAE(nn.Module):
    """
    Complete Variational Autoencoder model.
    
    Combines encoder and decoder with reparameterization trick
    for training with continuous latent variables.
    """
    
    def __init__(self, input_dim=784, hidden_dims=[512, 256], latent_dim=2, beta=1.0):
        super(VAE, self).__init__()
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.beta = beta  # Weight for KL divergence (β-VAE)
        
        # Build encoder and decoder
        self.encoder = Encoder(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            latent_dim=latent_dim
        )
        
        self.decoder = Decoder(
            latent_dim=latent_dim,
            hidden_dims=list(reversed(hidden_dims)),
            output_dim=input_dim
        )
        
    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick: z = μ + σ * ε
        
        This allows gradients to flow through the sampling operation.
        
        Args:
            mu: Mean of latent distribution [batch_size, latent_dim]
            logvar: Log variance of latent distribution [batch_size, latent_dim]
            
        Returns:
            z: Sampled latent codes [batch_size, latent_dim]
        """
        if self.training:
            # Sample epsilon from standard normal
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            # During inference, use mean
            return mu
            
    def forward(self, x):
        """
        Full forward pass through VAE.
        
        Args:
            x: Input data [batch_size, input_dim]
            
        Returns:
            reconstruction: Reconstructed input [batch_size, input_dim]
            mu: Latent mean [batch_size, latent_dim]
            logvar: Latent log variance [batch_size, latent_dim]
            z: Sampled latent codes [batch_size, latent_dim]
        """
        # Encode
        mu, logvar = self.encoder(x)
        
        # Reparameterize
        z = self.reparameterize(mu, logvar)
        
        # Decode
        reconstruction = self.decoder(z)
        
        return reconstruction, mu, logvar, z
        
    def encode(self, x):
        """Encode input to latent space."""
        return self.encoder(x)
        
    def decode(self, z):
        """Decode latent codes to data space."""
        return self.decoder(z)
        
    def generate(self, num_samples=16, device='cpu'):
        """
        Generate new samples by sampling from prior.
        
        Args:
            num_samples: Number of samples to generate
            device: Device to generate on
            
        Returns:
            Generated samples [num_samples, input_dim]
        """
        with torch.no_grad():
            # Sample from standard normal prior
            z = torch.randn(num_samples, self.latent_dim, device=device)
            samples = self.decode(z)
            return samples
            
    def interpolate(self, x1, x2, num_steps=10):
        """
        Interpolate between two inputs in latent space.
        
        Args:
            x1, x2: Input tensors to interpolate between
            num_steps: Number of interpolation steps
            
        Returns:
            Interpolated samples
        """
        with torch.no_grad():
            # Encode both inputs
            mu1, _ = self.encode(x1)
            mu2, _ = self.encode(x2)
            
            # Linear interpolation in latent space
            alphas = torch.linspace(0, 1, num_steps)
            interpolations = []
            
            for alpha in alphas:
                z_interp = (1 - alpha) * mu1 + alpha * mu2
                reconstruction = self.decode(z_interp)
                interpolations.append(reconstruction)
                
            return torch.stack(interpolations)


def vae_loss(reconstruction, target, mu, logvar, beta=1.0):
    """
    VAE loss function: reconstruction loss + KL divergence.
    
    Args:
        reconstruction: Reconstructed input [batch_size, input_dim]
        target: Original input [batch_size, input_dim]
        mu: Latent mean [batch_size, latent_dim]
        logvar: Latent log variance [batch_size, latent_dim]
        beta: Weight for KL divergence
        
    Returns:
        total_loss: Combined VAE loss
        recon_loss: Reconstruction loss component
        kl_loss: KL divergence component
    """
    # Reconstruction loss (Binary Cross Entropy)
    recon_loss = F.binary_cross_entropy(reconstruction, target, reduction='sum')
    
    # KL divergence: KL(q(z|x) || p(z))
    # where q(z|x) is learned posterior, p(z) is standard normal prior
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    # Total loss
    total_loss = recon_loss + beta * kl_loss
    
    return total_loss, recon_loss, kl_loss


def test_vae():
    """Test function to verify VAE implementation."""
    print("Testing VAE implementation...")
    
    # Create model
    model = VAE(input_dim=784, latent_dim=2)
    
    # Test forward pass with proper data range [0, 1]
    batch_size = 32
    x = torch.rand(batch_size, 784)  # Use rand (0-1) instead of randn
    
    reconstruction, mu, logvar, z = model(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Input range: [{x.min():.3f}, {x.max():.3f}]")
    print(f"Reconstruction shape: {reconstruction.shape}")
    print(f"Reconstruction range: [{reconstruction.min():.3f}, {reconstruction.max():.3f}]")
    print(f"Latent mean shape: {mu.shape}")
    print(f"Latent logvar shape: {logvar.shape}")
    print(f"Latent sample shape: {z.shape}")
    
    # Test loss calculation
    loss, recon_loss, kl_loss = vae_loss(reconstruction, x, mu, logvar)
    print(f"Total loss: {loss.item():.4f}")
    print(f"Reconstruction loss: {recon_loss.item():.4f}")
    print(f"KL loss: {kl_loss.item():.4f}")
    
    # Test generation
    generated = model.generate(num_samples=8)
    print(f"Generated samples shape: {generated.shape}")
    print(f"Generated range: [{generated.min():.3f}, {generated.max():.3f}]")
    
    # Test interpolation
    x1, x2 = x[:1], x[1:2]
    interpolated = model.interpolate(x1, x2, num_steps=5)
    print(f"Interpolated samples shape: {interpolated.shape}")
    
    print("✓ VAE test passed!")
    
    return model


if __name__ == "__main__":
    model = test_vae()