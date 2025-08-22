"""
Model inference functionality for ILVE framework.
Handles image generation and model inference operations.
"""

import torch
import numpy as np
from typing import Union, List, Tuple, Optional
import streamlit as st

class ModelInference:
    """
    Handles model inference operations including image generation.
    Extracted from the original generation logic for better modularity.
    """
    
    def __init__(self, model, device: str = 'cpu'):
        self.model = model
        self.device = device
        self.model.eval()
        
        # Cache model properties
        self._latent_dim = model.latent_dim
        self._image_shape = self._infer_image_shape()
        
        # Get image dimensions from session state if available
        if hasattr(st.session_state, 'model_image_dims'):
            self.image_dims = st.session_state.model_image_dims
            self.num_channels = st.session_state.model_num_channels
        else:
            # Fallback defaults
            self.image_dims = (28, 28)
            self.num_channels = 1
    
    def _infer_image_shape(self) -> Tuple[int, ...]:
        """Infer the output image shape from the model."""
        try:
            with torch.no_grad():
                test_z = torch.zeros(1, self._latent_dim).to(self.device)
                test_output = self.model.decode(test_z)
                
                # Handle different output formats
                if test_output.dim() == 4:  
                    return test_output.shape[1:]  
                elif test_output.dim() == 2:  
                    
                    total_pixels = test_output.shape[1]
                    if total_pixels == 784:  # MNIST
                        return (1, 28, 28)
                    else:
                        
                        side = int(np.sqrt(total_pixels))
                        return (1, side, side)
                else:
                    return (1, 28, 28)  
        except Exception:
            return (1, 28, 28)  
    
    def generate_image(self, latent_code: Union[List[float], torch.Tensor]) -> Optional[np.ndarray]:
        """
        Generate a single image from latent code.
        
        Args:
            latent_code: Latent code as list or tensor
            
        Returns:
            Generated image as numpy array or None if generation fails
        """
        try:
            with torch.no_grad():
                if isinstance(latent_code, list):
                    z = torch.tensor(latent_code, dtype=torch.float32).unsqueeze(0).to(self.device)
                elif len(latent_code.shape) == 1:
                    z = latent_code.unsqueeze(0).to(self.device)
                else:
                    z = latent_code.to(self.device)
                
                output = self.model.decode(z)
                
             
                if output.dim() == 4:  
                    image = output.squeeze(0).cpu().numpy()
                    if image.shape[0] == 1: 
                        image = image.squeeze(0)
                    elif image.shape[0] == 3:  
                        image = image.transpose(1, 2, 0)  
                elif output.dim() == 2:  
                    image = output.squeeze(0).cpu().numpy()
                    
                    h, w = self.image_dims
                    c = self.num_channels
                    if c == 1:
                        image = image.reshape(h, w)
                    else:
                        image = image.reshape(h, w, c)
                else:
                    image = output.squeeze().cpu().numpy()
                
                # Ensure values are in [0, 1] range
                image = np.clip(image, 0, 1)
                return image
                
        except Exception as e:
            print(f"Error generating image: {e}")
            return None
    
    def generate_batch(self, latent_codes: List[Union[List[float], torch.Tensor]]) -> List[Optional[np.ndarray]]:
        """
        Generate a batch of images from multiple latent codes.
        
        Args:
            latent_codes: List of latent codes
            
        Returns:
            List of generated images (some may be None if generation fails)
        """
        images = []
        for code in latent_codes:
            image = self.generate_image(code)
            images.append(image)
        return images
    
    def encode_image(self, image: np.ndarray) -> Optional[torch.Tensor]:
        """
        Encode an image to latent space (if encoder is available).
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Latent code as tensor or None if encoding fails
        """
        try:
            
            if len(image.shape) == 2:  
                image_tensor = torch.from_numpy(image).unsqueeze(0).unsqueeze(0).float()
            elif len(image.shape) == 3:  
                if image.shape[2] == 3:  
                    image_tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float()
                else:  
                    image_tensor = torch.from_numpy(image).unsqueeze(0).float()
            else:
                return None
            
            image_tensor = image_tensor.to(self.device)
            
            with torch.no_grad():
                if hasattr(self.model, 'encode'):
                    
                    mu, logvar = self.model.encode(image_tensor)
                    return mu.squeeze(0)  
                else:
                    
                    return None
                    
        except Exception as e:
            print(f"Error encoding image: {e}")
            return None
    
    def interpolate_latent_codes(self, 
                                code1: Union[List[float], torch.Tensor],
                                code2: Union[List[float], torch.Tensor],
                                num_steps: int = 10,
                                method: str = 'linear') -> List[torch.Tensor]:
        """
        Interpolate between two latent codes.
        
        Args:
            code1: First latent code
            code2: Second latent code
            num_steps: Number of interpolation steps
            method: Interpolation method ('linear' or 'slerp')
            
        Returns:
            List of interpolated latent codes
        """
        # Convert to tensors
        if isinstance(code1, list):
            z1 = torch.tensor(code1, dtype=torch.float32)
        else:
            z1 = code1.clone()
            
        if isinstance(code2, list):
            z2 = torch.tensor(code2, dtype=torch.float32)
        else:
            z2 = code2.clone()
        
        # Generate interpolation weights
        alphas = np.linspace(0, 1, num_steps)
        interpolated_codes = []
        
        for alpha in alphas:
            if method == 'linear':
                # Linear interpolation
                z_interp = (1 - alpha) * z1 + alpha * z2
            elif method == 'slerp':
                # Spherical linear interpolation
                z1_norm = z1 / (torch.norm(z1) + 1e-8)
                z2_norm = z2 / (torch.norm(z2) + 1e-8)
                
                omega = torch.acos(torch.clamp(torch.dot(z1_norm, z2_norm), -1, 1))
                
                if omega < 1e-6:
                    z_interp = z1
                else:
                    z_interp = (torch.sin((1-alpha)*omega)/torch.sin(omega) * z1 + 
                               torch.sin(alpha*omega)/torch.sin(omega) * z2)
            else:
                raise ValueError(f"Unknown interpolation method: {method}")
            
            interpolated_codes.append(z_interp)
        
        return interpolated_codes
    
    def sample_from_prior(self, 
                         num_samples: int = 1,
                         std: float = 1.0,
                         seed: Optional[int] = None) -> List[torch.Tensor]:
        """
        Sample latent codes from the prior distribution.
        
        Args:
            num_samples: Number of samples to generate
            std: Standard deviation for sampling
            seed: Random seed for reproducibility
            
        Returns:
            List of sampled latent codes
        """
        if seed is not None:
            torch.manual_seed(seed)
        
        samples = []
        for _ in range(num_samples):
            z = torch.randn(self._latent_dim) * std
            samples.append(z)
        
        return samples
    
    def get_model_info(self) -> dict:
        """Get information about the loaded model."""
        param_count = sum(p.numel() for p in self.model.parameters())
        
        return {
            'latent_dim': self._latent_dim,
            'parameter_count': param_count,
            'image_shape': self._image_shape,
            'image_dims': self.image_dims,
            'num_channels': self.num_channels,
            'device': self.device,
            'model_type': type(self.model).__name__
        }
    
    def calculate_reconstruction_loss(self, 
                                    original_image: np.ndarray,
                                    reconstructed_image: np.ndarray) -> float:
        """
        Calculate reconstruction loss between original and reconstructed images.
        
        Args:
            original_image: Original image
            reconstructed_image: Reconstructed image
            
        Returns:
            Reconstruction loss (MSE)
        """
        try:
            # Ensure same shape
            if original_image.shape != reconstructed_image.shape:
                return float('inf')
            
            # Calculate MSE
            mse = np.mean((original_image - reconstructed_image) ** 2)
            return float(mse)
            
        except Exception:
            return float('inf')
    
    def analyze_latent_space_coverage(self, 
                                    num_samples: int = 1000,
                                    std_range: Tuple[float, float] = (0.5, 2.0)) -> dict:
        """
        Analyze the coverage of the latent space by sampling.
        
        Args:
            num_samples: Number of samples to generate
            std_range: Range of standard deviations to test
            
        Returns:
            Dictionary with coverage analysis results
        """
        results = {
            'valid_generations': 0,
            'failed_generations': 0,
            'std_analysis': []
        }
        
        std_values = np.linspace(std_range[0], std_range[1], 5)
        
        for std in std_values:
            valid_count = 0
            samples_per_std = num_samples // len(std_values)
            
            for _ in range(samples_per_std):
                z = torch.randn(self._latent_dim) * std
                image = self.generate_image(z)
                
                if image is not None:
                    valid_count += 1
                    results['valid_generations'] += 1
                else:
                    results['failed_generations'] += 1
            
            results['std_analysis'].append({
                'std': float(std),
                'valid_rate': valid_count / samples_per_std,
                'samples_tested': samples_per_std
            })
        
        return results