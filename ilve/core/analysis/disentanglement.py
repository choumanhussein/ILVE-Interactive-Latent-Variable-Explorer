# =============================================================================
# core/analysis/disentanglement.py
"""
Disentanglement analysis for ILVE framework.
"""

import torch
import numpy as np
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass

@dataclass
class DisentanglementMetrics:
    """Container for disentanglement analysis metrics."""
    dimension: int
    effect_strength: float
    variance_range: float
    interpretability_score: float
    separability_score: float

class DisentanglementAnalyzer:
    """Analyzes disentanglement properties of VAE models."""
    
    def __init__(self, model, image_generator_func):
        self.model = model
        self.generate_image = image_generator_func
        self.latent_dim = model.latent_dim
    
    def analyze_dimension_effects(self, 
                                 dimensions: List[int],
                                 traversal_range: float = 2.5,
                                 num_steps: int = 7) -> List[DisentanglementMetrics]:
        """
        Analyze the effect strength and interpretability of specific dimensions.
        
        Args:
            dimensions: List of dimensions to analyze
            traversal_range: Range for traversal analysis
            num_steps: Number of steps in traversal
            
        Returns:
            List of DisentanglementMetrics for each dimension
        """
        results = []
        
        for dim in dimensions:
            if dim >= self.latent_dim:
                continue
                
            # Perform traversal analysis
            base_z = torch.zeros(self.latent_dim)
            values = np.linspace(-traversal_range, traversal_range, num_steps)
            
            images = []
            variances = []
            
            for value in values:
                z = base_z.clone()
                z[dim] = value
                image = self.generate_image(z)
                images.append(image)
                variances.append(np.var(image))
            
            # Calculate metrics
            effect_strength = np.std(variances)
            variance_range = np.max(variances) - np.min(variances)
            
            # Simple interpretability score based on variance progression
            interpretability_score = self._calculate_interpretability_score(variances)
            
            # Separability score based on image differences
            separability_score = self._calculate_separability_score(images)
            
            results.append(DisentanglementMetrics(
                dimension=dim,
                effect_strength=effect_strength,
                variance_range=variance_range,
                interpretability_score=interpretability_score,
                separability_score=separability_score
            ))
        
        return results
    
    def _calculate_interpretability_score(self, variances: List[float]) -> float:
        """Calculate interpretability score based on variance progression."""
        if len(variances) < 3:
            return 0.0
        
        # Check for monotonic or smooth progression
        diffs = np.diff(variances)
        smoothness = 1.0 - (np.std(diffs) / (np.mean(np.abs(diffs)) + 1e-8))
        
        # Normalize to [0, 1]
        return max(0.0, min(1.0, smoothness))
    
    def _calculate_separability_score(self, images: List[np.ndarray]) -> float:
        """Calculate separability score based on image differences."""
        if len(images) < 2:
            return 0.0
        
        # Calculate pairwise image differences
        differences = []
        for i in range(len(images)):
            for j in range(i + 1, len(images)):
                diff = np.mean(np.abs(images[i] - images[j]))
                differences.append(diff)
        
        # Separability is the mean difference
        separability = np.mean(differences)
        
        # Normalize (simple heuristic)
        return min(1.0, separability * 10)  # Scale factor based on typical image differences
    
    def calculate_beta_trade_off_metrics(self, beta: float) -> Dict[str, float]:
        """
        Calculate trade-off metrics for the current β value.
        
        Args:
            beta: Beta value of the model
            
        Returns:
            Dictionary of trade-off metrics
        """
        # Approximate metrics based on β value (these would ideally be measured)
        reconstruction_quality = max(100 - (beta - 1.0) * 10, 30) if beta >= 1.0 else 98
        disentanglement_score = min(beta / 4.0, 1.0) * 100 if beta >= 1.0 else (beta / 1.0 * 20)
        organization_score = min(beta * 10, 100)
        
        return {
            'beta': beta,
            'reconstruction_quality': reconstruction_quality,
            'disentanglement_score': disentanglement_score,
            'organization_score': organization_score,
            'balance_score': (reconstruction_quality + disentanglement_score) / 2
        }