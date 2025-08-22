# =============================================================================
# core/analysis/interaction.py
"""
Pair interaction analysis for ILVE framework.
"""

import torch
import numpy as np
from typing import List, Optional, Dict, Any
from .engine import AnalysisResult, AnalysisMethod, InteractionResult  
class PairInteraction:
    """Handles pair interaction analysis between two latent dimensions."""
    
    def __init__(self, model, image_generator_func):
        self.model = model
        self.generate_image = image_generator_func
        self.latent_dim = model.latent_dim
    
    def analyze(self,
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
        if dim1 >= self.latent_dim or dim2 >= self.latent_dim:
            raise ValueError(f"Dimensions {dim1}, {dim2} exceed model latent_dim {self.latent_dim}")
        
        if dim1 == dim2:
            raise ValueError("Cannot analyze interaction between the same dimension")
        
        # Create base latent code
        if base_latent is None:
            base_z = torch.zeros(self.latent_dim)
        else:
            base_z = torch.tensor(base_latent[:self.latent_dim], dtype=torch.float32)
            if len(base_latent) < self.latent_dim:
                padding = torch.zeros(self.latent_dim - len(base_latent))
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
                image = self.generate_image(z)
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