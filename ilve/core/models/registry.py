"""
Model registry for explicit model type registration.
Supports pluggable model architectures for future extensibility.
"""

from typing import Dict, Type, Callable, Any, Optional
from abc import ABC, abstractmethod
import torch
import torch.nn as nn

class BaseModelInterface(ABC):
    """Abstract base class for all models in the framework."""
    
    @property
    @abstractmethod
    def latent_dim(self) -> int:
        """Return the latent dimension of the model."""
        pass
    
    @abstractmethod
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input to latent space."""
        pass
    
    @abstractmethod
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent code to output."""
        pass
    
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the model."""
        pass

class ModelTypeInfo:
    """Information about a registered model type."""
    
    def __init__(self, 
                 model_class: Type[BaseModelInterface],
                 loader_func: Callable,
                 display_name: str,
                 description: str):
        self.model_class = model_class
        self.loader_func = loader_func
        self.display_name = display_name
        self.description = description

class ModelRegistry:
    """
    Registry for model types with explicit registration.
    Enables pluggable model architectures for future extensibility.
    """
    
    def __init__(self):
        self._registry: Dict[str, ModelTypeInfo] = {}
        self._default_loaders: Dict[str, Callable] = {}
        
    def register_model_type(self, 
                           model_key: str,
                           model_class: Type[BaseModelInterface],
                           loader_func: Callable,
                           display_name: str,
                           description: str = ""):
        """
        Register a new model type.
        
        Args:
            model_key: Unique identifier for the model type
            model_class: Model class that implements BaseModelInterface
            loader_func: Function to load/initialize this model type
            display_name: Human-readable name for the model
            description: Optional description of the model
        """
        self._registry[model_key] = ModelTypeInfo(
            model_class=model_class,
            loader_func=loader_func,
            display_name=display_name,
            description=description
        )
    
    def get_model_types(self) -> Dict[str, ModelTypeInfo]:
        """Get all registered model types."""
        return self._registry.copy()
    
    def get_model_info(self, model_key: str) -> Optional[ModelTypeInfo]:
        """Get information about a specific model type."""
        return self._registry.get(model_key)
    
    def is_registered(self, model_key: str) -> bool:
        """Check if a model type is registered."""
        return model_key in self._registry
    
    def load_model(self, model_key: str, *args, **kwargs) -> Optional[BaseModelInterface]:
        """
        Load a model of the specified type.
        
        Args:
            model_key: The registered model type key
            *args, **kwargs: Arguments to pass to the loader function
            
        Returns:
            Loaded model instance or None if failed
        """
        if model_key not in self._registry:
            raise ValueError(f"Model type '{model_key}' not registered. "
                           f"Available types: {list(self._registry.keys())}")
        
        model_info = self._registry[model_key]
        try:
            return model_info.loader_func(*args, **kwargs)
        except Exception as e:
            print(f"Failed to load model type '{model_key}': {e}")
            return None
    
    def detect_model_type(self, model_path: str, filename: str = None) -> str:
        """
        Detect model type from file path or filename.
        
        Args:
            model_path: Path to the model file
            filename: Optional filename (extracted from path if not provided)
            
        Returns:
            Detected model type key, defaults to 'vae' if uncertain
        """
        if filename is None:
            filename = model_path.split('/')[-1].lower()
        else:
            filename = filename.lower()
        
      
        if 'beta' in filename:
            if 'controlled' in filename:
                return 'controlled_beta_vae'
            return 'beta_vae'
        elif 'vae' in filename:
            return 'vae'
        else:
            
            return 'vae'

model_registry = ModelRegistry()

def register_default_models():
    """Register the default VAE model types."""
    try:
       
        from models.base_vae import VAE
        from models.beta_vae import BetaVAE, ControlledBetaVAE
        
        
        class VAEWrapper(VAE, BaseModelInterface):
            def encode(self, x: torch.Tensor) -> torch.Tensor:
                mu, logvar = super().encode(x)
                return mu  # 
            
            def decode(self, z: torch.Tensor) -> torch.Tensor:
                return super().decode(z)
        
        class BetaVAEWrapper(BetaVAE, BaseModelInterface):
            def encode(self, x: torch.Tensor) -> torch.Tensor:
                mu, logvar = super().encode(x)
                return mu  
            
            def decode(self, z: torch.Tensor) -> torch.Tensor:
                return super().decode(z)
        
        class ControlledBetaVAEWrapper(ControlledBetaVAE, BaseModelInterface):
            def encode(self, x: torch.Tensor) -> torch.Tensor:
                mu, logvar = super().encode(x)
                return mu  
            
            def decode(self, z: torch.Tensor) -> torch.Tensor:
                return super().decode(z)
        
        # Loader functions
        def load_vae(**kwargs):
            return VAEWrapper(**kwargs)
        
        def load_beta_vae(**kwargs):
            return BetaVAEWrapper(**kwargs)
        
        def load_controlled_beta_vae(**kwargs):
            return ControlledBetaVAEWrapper(**kwargs)
        
        # Register the model types
        model_registry.register_model_type(
            model_key='vae',
            model_class=VAEWrapper,
            loader_func=load_vae,
            display_name='Standard VAE',
            description='Basic Variational Autoencoder with β=1.0'
        )
        
        model_registry.register_model_type(
            model_key='beta_vae',
            model_class=BetaVAEWrapper,
            loader_func=load_beta_vae,
            display_name='β-VAE',
            description='Beta Variational Autoencoder with adjustable β parameter for disentanglement'
        )
        
        model_registry.register_model_type(
            model_key='controlled_beta_vae',
            model_class=ControlledBetaVAEWrapper,
            loader_func=load_controlled_beta_vae,
            display_name='Controlled β-VAE',
            description='Beta VAE with dimension-specific β control'
        )
        
    except ImportError as e:
        print(f"Warning: Could not register default models: {e}")
        print("Please ensure the models directory is accessible.")


register_default_models()

def get_registry() -> ModelRegistry:
    """Get the global model registry instance."""
    return model_registry