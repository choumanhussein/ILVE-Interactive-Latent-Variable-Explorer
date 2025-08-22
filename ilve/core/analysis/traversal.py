# core/analysis/traversal.py
"""
Individual dimension traversal analysis for ILVE framework.
"""

import torch
import numpy as np
from typing import List, Optional, Dict, Any
from .engine import AnalysisResult, AnalysisMethod, TraversalResult 

class IndividualTraversal:
    """Handles individual latent dimension traversal analysis."""
    
    def __init__(self, model, image_generator_func):
        self.model = model
        self.generate_image = image_generator_func
        self.latent_dim = model.latent_dim
    
    def analyze(self,
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
        if dimension >= self.latent_dim:
            raise ValueError(f"Dimension {dimension} exceeds model latent_dim {self.latent_dim}")
        
        # Create base latent code
        if base_latent is None:
            base_z = torch.zeros(self.latent_dim)
        else:
            base_z = torch.tensor(base_latent[:self.latent_dim], dtype=torch.float32)
            if len(base_latent) < self.latent_dim:
                padding = torch.zeros(self.latent_dim - len(base_latent))
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
            image = self.generate_image(z)
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