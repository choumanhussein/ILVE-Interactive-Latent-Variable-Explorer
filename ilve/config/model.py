# =============================================================================
# config/models.py
"""
Model configuration definitions for ILVE framework.
"""

from dataclasses import dataclass
from typing import Dict, Any, List, Optional

@dataclass
class ModelConfigurations:
    """Standard model configurations for different use cases."""
    
    @staticmethod
    def get_standard_configs() -> Dict[str, Dict[str, Any]]:
        """Get standard model configurations."""
        return {
            'mnist_2d': {
                'input_dim': 784,
                'hidden_dims': [512, 256],
                'latent_dim': 2,
                'image_shape': (28, 28, 1),
                'description': 'Standard 2D MNIST VAE for visualization'
            },
            
            'mnist_10d': {
                'input_dim': 784,
                'hidden_dims': [512, 256],
                'latent_dim': 10,
                'image_shape': (28, 28, 1),
                'description': 'Higher-dimensional MNIST VAE for detailed analysis'
            },
            
            'mnist_20d': {
                'input_dim': 784,
                'hidden_dims': [512, 256, 128],
                'latent_dim': 20,
                'image_shape': (28, 28, 1),
                'description': 'High-dimensional MNIST VAE for research'
            },
            
            'celeba_64d': {
                'input_dim': 12288,  # 64x64x3
                'hidden_dims': [1024, 512, 256],
                'latent_dim': 64,
                'image_shape': (64, 64, 3),
                'description': 'CelebA faces VAE for facial attribute analysis'
            },
            
            'beta_vae_disentangled': {
                'input_dim': 784,
                'hidden_dims': [512, 256],
                'latent_dim': 10,
                'beta': 4.0,
                'image_shape': (28, 28, 1),
                'description': 'β-VAE optimized for disentanglement'
            }
        }
    
    @staticmethod
    def get_beta_recommendations() -> Dict[str, Dict[str, Any]]:
        """Get recommended beta values for different use cases."""
        return {
            'visualization': {
                'beta_range': (0.5, 2.0),
                'recommended': 1.0,
                'description': 'Prioritize image quality for demonstrations'
            },
            
            'education': {
                'beta_range': (2.0, 6.0),
                'recommended': 4.0,
                'description': 'Balance quality and interpretability for learning'
            },
            
            'research': {
                'beta_range': (4.0, 16.0),
                'recommended': 8.0,
                'description': 'Emphasize disentanglement for analysis'
            },
            
            'production': {
                'beta_range': (1.0, 4.0),
                'recommended': 2.0,
                'description': 'Practical balance for real applications'
            }
        }
    
    @staticmethod
    def get_architecture_templates() -> Dict[str, Dict[str, Any]]:
        """Get architecture templates for different model sizes."""
        return {
            'lightweight': {
                'hidden_dims': [256, 128],
                'max_latent_dim': 16,
                'description': 'Fast training and inference'
            },
            
            'standard': {
                'hidden_dims': [512, 256],
                'max_latent_dim': 32,
                'description': 'Good balance of capacity and speed'
            },
            
            'large': {
                'hidden_dims': [1024, 512, 256],
                'max_latent_dim': 64,
                'description': 'High capacity for complex data'
            },
            
            'research': {
                'hidden_dims': [1024, 512, 256, 128],
                'max_latent_dim': 128,
                'description': 'Maximum flexibility for experiments'
            }
        }
