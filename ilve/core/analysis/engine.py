"""
Core analysis engine for ILVE framework.
Implements the three main analysis methodologies from the research paper.
"""

import torch
import numpy as np
from typing import List, Dict, Any, Tuple, Optional, Union
from dataclasses import dataclass, field
from enum import Enum
import matplotlib.pyplot as plt

class AnalysisMethod(Enum):
    """Enumeration of available analysis methods."""
    INDIVIDUAL_TRAVERSAL = "individual_traversal"
    PAIR_INTERACTION = "pair_interaction"
    RANDOM_GENERATION = "random_generation"

@dataclass
class AnalysisResult:
    """Result container for analysis operations."""
    method: AnalysisMethod
    images: List[np.ndarray]
    latent_codes: List[List[float]]
    metadata: Dict[str, Any]
    metrics: Optional[Dict[str, float]] = None

@dataclass
class TraversalResult(AnalysisResult):
    """Result for individual dimension traversal."""
    dimension: int = 0
    values: List[float] = field(default_factory=list)
    variance_scores: List[float] = field(default_factory=list)

@dataclass
class InteractionResult(AnalysisResult):
    """Result for pair interaction analysis."""
    dim1: int = 0
    dim2: int = 0
    grid_values: List[List[float]] = field(default_factory=list)

@dataclass
class GenerationResult(AnalysisResult):
    """Result for random generation analysis."""
    sampling_std: float = 1.0
    num_samples: int = 8

class AnalysisEngine:
    """
    Main analysis engine implementing the three core methodologies:
    1. Individual Knob Traversal
    2. Knob Pair Interaction  
    3. Random Image Generation
    """
    
    def __init__(self, model, device: str = 'cpu'):
        self.model = model
        self.device = device
        self.model.eval()
        
        # Cache model properties
        self._latent_dim = model.latent_dim
        self._image_shape = self._infer_image_shape()
    
    def _infer_image_shape(self) -> Tuple[int, ...]:
        """Infer the output image shape from the model."""
        # Generate a test image to determine shape
        with torch.no_grad():
            test_z = torch.zeros(1, self._latent_dim).to(self.device)
            test_output = self.model.decode(test_z)
            
            # Handle different output formats
            if test_output.dim() == 4:  # (B, C, H, W)
                return test_output.shape[1:]  # (C, H, W)
            elif test_output.dim() == 2:  # (B, H*W*C) - flattened
                # Try to infer square image
                total_pixels = test_output.shape[1]
                if total_pixels == 784:  # MNIST
                    return (1, 28, 28)
                else:
                    # Assume square grayscale
                    side = int(np.sqrt(total_pixels))
                    return (1, side, side)
            else:
                return (1, 28, 28)  # Default fallback
    
    def _generate_image(self, latent_code: Union[List[float], torch.Tensor]) -> np.ndarray:
        """Generate a single image from latent code."""
        with torch.no_grad():
            if isinstance(latent_code, list):
                z = torch.tensor(latent_code, dtype=torch.float32).unsqueeze(0).to(self.device)
            elif len(latent_code.shape) == 1:
                z = latent_code.unsqueeze(0).to(self.device)
            else:
                z = latent_code.to(self.device)
            
            output = self.model.decode(z)
            
            # Convert to numpy and handle different formats
            if output.dim() == 4:  # (B, C, H, W)
                image = output.squeeze(0).cpu().numpy()
                if image.shape[0] == 1:  # Grayscale
                    image = image.squeeze(0)
                elif image.shape[0] == 3:  # RGB
                    image = image.transpose(1, 2, 0)  # (H, W, C)
            elif output.dim() == 2:  # (B, H*W*C) - flattened
                image = output.squeeze(0).cpu().numpy()
                # Reshape based on inferred shape
                if len(self._image_shape) == 3:
                    c, h, w = self._image_shape
                    image = image.reshape(h, w) if c == 1 else image.reshape(h, w, c)
                else:
                    # Fallback to square
                    side = int(np.sqrt(len(image)))
                    image = image.reshape(side, side)
            else:
                image = output.squeeze().cpu().numpy()
            
            # Ensure values are in [0, 1] range
            image = np.clip(image, 0, 1)
            return image
    
    def individual_traversal(self, 
                           dimension: int,
                           traversal_range: float = 2.5,
                           num_steps: int = 7,
                           base_latent: Optional[List[float]] = None) -> TraversalResult:
        """
        Perform individual dimension traversal analysis.
        
        Args:
            dimension: Which latent dimension to traverse
            traversal_range: Range of values to traverse [-range, +range]
            num_steps: Number of steps in the traversal
            base_latent: Base latent code (zeros if None)
            
        Returns:
            TraversalResult containing images and analysis
        """
        if dimension >= self._latent_dim:
            raise ValueError(f"Dimension {dimension} exceeds model latent_dim {self._latent_dim}")
        
        # Create base latent code
        if base_latent is None:
            base_z = torch.zeros(self._latent_dim)
        else:
            base_z = torch.tensor(base_latent[:self._latent_dim], dtype=torch.float32)
            if len(base_latent) < self._latent_dim:
                # Pad with zeros if needed
                padding = torch.zeros(self._latent_dim - len(base_latent))
                base_z = torch.cat([base_z, padding])
        
        # Generate traversal values
        values = np.linspace(-traversal_range, traversal_range, num_steps)
        
        images = []
        latent_codes = []
        variance_scores = []
        
        for value in values:
            # Create latent code with modified dimension
            z = base_z.clone()
            z[dimension] = value
            
            # Generate image
            image = self._generate_image(z)
            images.append(image)
            latent_codes.append(z.tolist())
            
            # Calculate variance score for this image
            variance_scores.append(float(np.var(image)))
        
        # Calculate metadata
        effect_strength = np.std(variance_scores)
        max_variance_idx = np.argmax(variance_scores)
        min_variance_idx = np.argmin(variance_scores)
        
        metadata = {
            'dimension': dimension,
            'traversal_range': traversal_range,
            'num_steps': num_steps,
            'effect_strength': effect_strength,
            'max_variance_value': values[max_variance_idx],
            'min_variance_value': values[min_variance_idx],
            'mean_variance': np.mean(variance_scores)
        }
        
        return TraversalResult(
            method=AnalysisMethod.INDIVIDUAL_TRAVERSAL,
            images=images,
            latent_codes=latent_codes,
            metadata=metadata,
            dimension=dimension,
            values=values.tolist(),
            variance_scores=variance_scores
        )
    
    def pair_interaction(self,
                        dim1: int,
                        dim2: int,
                        traversal_range: float = 2.5,
                        grid_size: int = 5,
                        base_latent: Optional[List[float]] = None) -> InteractionResult:
        """
        Perform pair interaction analysis between two dimensions.
        
        Args:
            dim1: First dimension to analyze
            dim2: Second dimension to analyze
            traversal_range: Range of values for both dimensions
            grid_size: Size of the interaction grid (grid_size x grid_size)
            base_latent: Base latent code (zeros if None)
            
        Returns:
            InteractionResult containing interaction analysis
        """
        if dim1 >= self._latent_dim or dim2 >= self._latent_dim:
            raise ValueError(f"Dimensions {dim1}, {dim2} exceed model latent_dim {self._latent_dim}")
        
        if dim1 == dim2:
            raise ValueError("Cannot analyze interaction between the same dimension")
        
        # Create base latent code
        if base_latent is None:
            base_z = torch.zeros(self._latent_dim)
        else:
            base_z = torch.tensor(base_latent[:self._latent_dim], dtype=torch.float32)
            if len(base_latent) < self._latent_dim:
                padding = torch.zeros(self._latent_dim - len(base_latent))
                base_z = torch.cat([base_z, padding])
        
        # Generate grid values
        values = np.linspace(-traversal_range, traversal_range, grid_size)
        
        images = []
        latent_codes = []
        grid_values = []
        
        for i, val1 in enumerate(values):
            row_images = []
            row_codes = []
            row_values = []
            
            for j, val2 in enumerate(values):
                # Create latent code with modified dimensions
                z = base_z.clone()
                z[dim1] = val1
                z[dim2] = val2
                
                # Generate image
                image = self._generate_image(z)
                row_images.append(image)
                row_codes.append(z.tolist())
                row_values.append([val1, val2])
            
            images.append(row_images)
            latent_codes.append(row_codes)
            grid_values.append(row_values)
        
        # Flatten for result storage
        flat_images = [img for row in images for img in row]
        flat_codes = [code for row in latent_codes for code in row]
        
        # Calculate interaction metrics
        variance_matrix = np.array([[np.var(img) for img in row] for row in images])
        interaction_strength = np.std(variance_matrix)
        
        metadata = {
            'dim1': dim1,
            'dim2': dim2,
            'traversal_range': traversal_range,
            'grid_size': grid_size,
            'interaction_strength': interaction_strength,
            'variance_matrix': variance_matrix.tolist()
        }
        
        return InteractionResult(
            method=AnalysisMethod.PAIR_INTERACTION,
            images=flat_images,
            latent_codes=flat_codes,
            metadata=metadata,
            dim1=dim1,
            dim2=dim2,
            grid_values=grid_values
        )
    
    def random_generation(self,
                         num_samples: int = 8,
                         sampling_std: float = 1.0,
                         seed: Optional[int] = None) -> GenerationResult:
        """
        Perform random generation analysis.
        
        Args:
            num_samples: Number of random samples to generate
            sampling_std: Standard deviation for random sampling
            seed: Random seed for reproducibility
            
        Returns:
            GenerationResult containing random samples
        """
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
        
        # Generate random latent codes
        random_codes = torch.randn(num_samples, self._latent_dim) * sampling_std
        
        images = []
        latent_codes = []
        
        for code in random_codes:
            image = self._generate_image(code)
            images.append(image)
            latent_codes.append(code.tolist())
        
        # Calculate diversity metrics
        variance_scores = [np.var(img) for img in images]
        diversity_score = np.std(variance_scores)
        
        metadata = {
            'num_samples': num_samples,
            'sampling_std': sampling_std,
            'diversity_score': diversity_score,
            'mean_variance': np.mean(variance_scores),
            'seed': seed
        }
        
        return GenerationResult(
            method=AnalysisMethod.RANDOM_GENERATION,
            images=images,
            latent_codes=latent_codes,
            metadata=metadata,
            sampling_std=sampling_std,
            num_samples=num_samples
        )
    
    def comprehensive_analysis(self,
                             selected_dims: List[int],
                             traversal_range: float = 2.5,
                             num_steps: int = 7) -> Dict[str, AnalysisResult]:
        """
        Perform comprehensive analysis on selected dimensions.
        
        Args:
            selected_dims: List of dimensions to analyze
            traversal_range: Range for traversal analysis
            num_steps: Number of steps for traversal
            
        Returns:
            Dictionary of analysis results by method
        """
        results = {}
        
        # Individual traversals
        individual_results = []
        for dim in selected_dims:
            result = self.individual_traversal(dim, traversal_range, num_steps)
            individual_results.append(result)
        
        results['individual_traversals'] = individual_results
        
        # Pair interactions (if multiple dimensions)
        if len(selected_dims) >= 2:
            pair_results = []
            for i in range(min(3, len(selected_dims))):  # Limit to avoid too many combinations
                for j in range(i + 1, min(3, len(selected_dims))):
                    result = self.pair_interaction(selected_dims[i], selected_dims[j], traversal_range)
                    pair_results.append(result)
            results['pair_interactions'] = pair_results
        
        # Random generation
        random_result = self.random_generation(num_samples=8, sampling_std=1.5)
        results['random_generation'] = random_result
        
        return results
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        param_count = sum(p.numel() for p in self.model.parameters())
        
        return {
            'latent_dim': self._latent_dim,
            'parameter_count': param_count,
            'image_shape': self._image_shape,
            'device': self.device,
            'model_type': type(self.model).__name__
        }