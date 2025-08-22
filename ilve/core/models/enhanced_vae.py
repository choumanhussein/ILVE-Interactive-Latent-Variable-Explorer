"""
Enhanced VAE with flexible architecture that can match saved models.
This version can handle models with or without BatchNorm layers.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional
from dataclasses import dataclass

@dataclass
class EnhancedVAEConfig:
    """Enhanced configuration class for VAE models."""
    input_dim: int = 784
    latent_dim: int = 2
    hidden_dims: List[int] = None
    learning_rate: float = 1e-3
    beta: float = 1.0
    use_batch_norm: bool = False  
    use_dropout: bool = True      
    
    def __post_init__(self):
        if self.hidden_dims is None:
            self.hidden_dims = [512, 256]

class FlexibleEncoder(nn.Module):
    """Flexible encoder that can match different architectures."""
    
    def __init__(self, input_dim=784, hidden_dims=[512, 256], latent_dim=2, 
                 use_batch_norm=False, use_dropout=True):
        super(FlexibleEncoder, self).__init__()
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.use_batch_norm = use_batch_norm
        self.use_dropout = use_dropout
        
        # Build encoder layers with flexible architecture
        layers = []
        prev_dim = input_dim
        
        for i, hidden_dim in enumerate(hidden_dims):
            # Linear layer
            layers.append(nn.Linear(prev_dim, hidden_dim))
            
            # Optional BatchNorm
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            
            # Activation
            layers.append(nn.ReLU())
            
            # Optional Dropout
            if use_dropout:
                layers.append(nn.Dropout(0.2))
            
            prev_dim = hidden_dim
            
        self.encoder = nn.Sequential(*layers)
        
        # Latent space parameters
        self.fc_mu = nn.Linear(prev_dim, latent_dim)
        self.fc_logvar = nn.Linear(prev_dim, latent_dim)
        
    def forward(self, x):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

class FlexibleDecoder(nn.Module):
    """Flexible decoder that can match different architectures."""
    
    def __init__(self, latent_dim=2, hidden_dims=[256, 512], output_dim=784,
                 use_batch_norm=False, use_dropout=True):
        super(FlexibleDecoder, self).__init__()
        
        self.latent_dim = latent_dim
        self.output_dim = output_dim
        self.use_batch_norm = use_batch_norm
        self.use_dropout = use_dropout
        
        # Build decoder layers with flexible architecture
        layers = []
        prev_dim = latent_dim
        
        for i, hidden_dim in enumerate(hidden_dims):
            # Linear layer
            layers.append(nn.Linear(prev_dim, hidden_dim))
            
            # Optional BatchNorm
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            
            # Activation
            layers.append(nn.ReLU())
            
            # Optional Dropout
            if use_dropout:
                layers.append(nn.Dropout(0.2))
            
            prev_dim = hidden_dim
        
        # Final output layer
        layers.append(nn.Linear(prev_dim, output_dim))
        layers.append(nn.Sigmoid())
        
        self.decoder = nn.Sequential(*layers)
        
    def forward(self, z):
        return self.decoder(z)

class EnhancedVAE(nn.Module):
    """Enhanced VAE with flexible architecture matching."""
    
    def __init__(self, input_dim=784, hidden_dims=[512, 256], latent_dim=2, 
                 beta=1.0, use_batch_norm=False, use_dropout=True):
        super(EnhancedVAE, self).__init__()
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.beta = beta
        
        # Build encoder and decoder with flexible architecture
        self.encoder = FlexibleEncoder(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            latent_dim=latent_dim,
            use_batch_norm=use_batch_norm,
            use_dropout=use_dropout
        )
        
        self.decoder = FlexibleDecoder(
            latent_dim=latent_dim,
            hidden_dims=list(reversed(hidden_dims)),
            output_dim=input_dim,
            use_batch_norm=use_batch_norm,
            use_dropout=use_dropout
        )
    
    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            return mu
    
    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        reconstruction = self.decoder(z)
        return reconstruction, mu, logvar, z
    
    def encode(self, x):
        return self.encoder(x)
    
    def decode(self, z):
        return self.decoder(z)
    
    def generate(self, num_samples=16, device='cpu'):
        with torch.no_grad():
            z = torch.randn(num_samples, self.latent_dim, device=device)
            samples = self.decode(z)
            return samples

def detect_architecture_from_state_dict(state_dict):
    """Detect the architecture configuration from state_dict keys."""
    has_batch_norm = any('running_mean' in key or 'running_var' in key for key in state_dict.keys())
    has_dropout = False  
    
   
    encoder_linear_layers = [key for key in state_dict.keys() if key.startswith('encoder.encoder.') and key.endswith('.weight') and 'fc_' not in key]
    
    return {
        'use_batch_norm': has_batch_norm,
        'use_dropout': True,  
        'num_encoder_layers': len(encoder_linear_layers)
    }

# Enhanced Beta VAE
class EnhancedBetaVAE(EnhancedVAE):
    """Enhanced Beta VAE with flexible architecture."""
    
    def __init__(self, input_dim=784, hidden_dims=[512, 256], latent_dim=2, 
                 beta=4.0, use_batch_norm=False, use_dropout=True):
        super(EnhancedBetaVAE, self).__init__(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            latent_dim=latent_dim,
            beta=beta,
            use_batch_norm=use_batch_norm,
            use_dropout=use_dropout
        )