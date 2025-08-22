# =============================================================================
# core/analysis/generation.py
"""
Random generation analysis for ILVE framework.
"""

import torch
import numpy as np
from typing import List, Optional, Dict, Any
from .engine import AnalysisResult, AnalysisMethod, GenerationResult 

class RandomGeneration:
    """Handles random generation analysis."""
    
    def __init__(self, model, image_generator_func):
        self.model = model
        self.generate_image = image_generator_func
        self.latent_dim = model.latent_dim
    
    def analyze(self,
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
        random_codes = torch.randn(num_samples, self.latent_dim) * sampling_std
        
        images = []
        latent_codes = []
        
        for code in random_codes:
            image = self.generate_image(code)
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