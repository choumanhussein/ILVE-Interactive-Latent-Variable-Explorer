# core/metrics/quantitative.py
"""
Quantitative metrics for ILVE framework analysis.
"""

import numpy as np
import torch
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass

@dataclass
class MetricsResult:
    """Container for quantitative metrics results."""
    metric_name: str
    value: float
    metadata: Dict[str, Any]

class QuantitativeMetrics:
    """Handles quantitative analysis of VAE representations and generations."""
    
    @staticmethod
    def pixel_variance_analysis(images: List[np.ndarray]) -> Dict[str, float]:
        """
        Analyze pixel variance across a set of images.
        
        Args:
            images: List of generated images
            
        Returns:
            Dictionary of variance metrics
        """
        if not images:
            return {'mean_variance': 0.0, 'std_variance': 0.0, 'max_variance': 0.0, 'min_variance': 0.0}
        
        variances = [np.var(img) for img in images]
        
        return {
            'mean_variance': np.mean(variances),
            'std_variance': np.std(variances),
            'max_variance': np.max(variances),
            'min_variance': np.min(variances),
            'variance_range': np.max(variances) - np.min(variances)
        }
    
    @staticmethod
    def dimension_effect_strength(images: List[np.ndarray], 
                                 latent_values: List[float]) -> float:
        """
        Calculate the effect strength of a latent dimension.
        
        Args:
            images: Images generated from traversing a dimension
            latent_values: Corresponding latent values
            
        Returns:
            Effect strength score
        """
        if len(images) < 2:
            return 0.0
        
        # Calculate image differences
        image_diffs = []
        for i in range(len(images) - 1):
            diff = np.mean(np.abs(images[i+1] - images[i]))
            image_diffs.append(diff)
        
        # Calculate latent differences
        latent_diffs = np.diff(latent_values)
        
        # Effect strength is the correlation between latent and image changes
        if len(image_diffs) > 1 and len(latent_diffs) > 1:
            correlation = np.corrcoef(image_diffs, latent_diffs[:-1] if len(latent_diffs) > len(image_diffs) else latent_diffs)[0, 1]
            if np.isnan(correlation):
                correlation = 0.0
            return abs(correlation)
        
        return np.mean(image_diffs)
    
    @staticmethod
    def calculate_separability_score(images: List[np.ndarray]) -> float:
        """
        Calculate separability score for a set of images.
        
        Args:
            images: List of images to analyze
            
        Returns:
            Separability score (higher = more separable)
        """
        if len(images) < 2:
            return 0.0
        
        # Calculate pairwise distances
        distances = []
        for i in range(len(images)):
            for j in range(i + 1, len(images)):
                distance = np.mean(np.abs(images[i] - images[j]))
                distances.append(distance)
        
        # Separability is the mean pairwise distance
        separability = np.mean(distances)
        
        # Normalize (heuristic scaling)
        return min(1.0, separability * 10)
    
    @staticmethod
    def diversity_score(images: List[np.ndarray]) -> float:
        """
        Calculate diversity score for generated images.
        
        Args:
            images: List of generated images
            
        Returns:
            Diversity score
        """
        if len(images) < 2:
            return 0.0
        
        # Calculate variance of image variances
        variances = [np.var(img) for img in images]
        diversity = np.std(variances)
        
        return diversity

class DisentanglementMetrics:
    """Specialized metrics for disentanglement analysis."""
    
    @staticmethod
    def calculate_mig_score(latent_codes: List[List[float]], 
                           factor_labels: Optional[List[int]] = None) -> float:
        """
        Calculate Mutual Information Gap (MIG) score.
        Simplified implementation for educational purposes.
        
        Args:
            latent_codes: List of latent codes
            factor_labels: Optional factor labels for supervision
            
        Returns:
            MIG score approximation
        """
        if not latent_codes or len(latent_codes) < 10:
            return 0.0
        
        # Convert to numpy array
        codes_array = np.array(latent_codes)
        
        # Calculate variance across dimensions
        dim_variances = np.var(codes_array, axis=0)
        
        # Simple MIG approximation: ratio of max to second max variance
        sorted_vars = np.sort(dim_variances)[::-1]
        if len(sorted_vars) < 2 or sorted_vars[1] == 0:
            return 0.0
        
        mig_approx = (sorted_vars[0] - sorted_vars[1]) / sorted_vars[0]
        return max(0.0, min(1.0, mig_approx))
    
    @staticmethod
    def calculate_sap_score(latent_codes: List[List[float]], 
                           images: List[np.ndarray]) -> float:
        """
        Calculate Separated Attribute Predictability (SAP) score approximation.
        
        Args:
            latent_codes: List of latent codes
            images: Corresponding generated images
            
        Returns:
            SAP score approximation
        """
        if len(latent_codes) != len(images) or len(latent_codes) < 5:
            return 0.0
        
        codes_array = np.array(latent_codes)
        
        # Calculate image features (simple features for approximation)
        image_features = []
        for img in images:
            features = [
                np.mean(img),  # Average brightness
                np.std(img),   # Contrast
                np.mean(np.abs(np.diff(img, axis=0))),  # Vertical edges
                np.mean(np.abs(np.diff(img, axis=1)))   # Horizontal edges
            ]
            image_features.append(features)
        
        features_array = np.array(image_features)
        
        # Calculate correlations between latent dims and image features
        correlations = []
        for i in range(codes_array.shape[1]):  # For each latent dimension
            dim_correlations = []
            for j in range(features_array.shape[1]):  # For each image feature
                try:
                    corr = np.corrcoef(codes_array[:, i], features_array[:, j])[0, 1]
                    if not np.isnan(corr):
                        dim_correlations.append(abs(corr))
                except:
                    dim_correlations.append(0.0)
            correlations.append(max(dim_correlations) if dim_correlations else 0.0)
        
        # SAP approximation: mean of highest correlations
        return np.mean(correlations)
    
    @staticmethod
    def beta_trade_off_analysis(beta: float) -> Dict[str, float]:
        """
        Analyze the theoretical trade-offs for a given beta value.
        
        Args:
            beta: Beta parameter value
            
        Returns:
            Dictionary of trade-off metrics
        """
        # Theoretical relationships (based on typical β-VAE behavior)
        reconstruction_quality = max(100 - (beta - 1.0) * 12, 20) if beta >= 1.0 else 95
        disentanglement_score = min(beta * 20, 95) if beta <= 4.0 else 95 - (beta - 4.0) * 2
        organization_score = min(beta * 15, 90)
        
        # Calculate balance score
        balance_score = 2 * (reconstruction_quality * disentanglement_score) / (reconstruction_quality + disentanglement_score + 1e-8)
        
        return {
            'beta': beta,
            'reconstruction_quality': reconstruction_quality,
            'disentanglement_score': disentanglement_score,
            'organization_score': organization_score,
            'balance_score': balance_score,
            'trade_off_ratio': disentanglement_score / (reconstruction_quality + 1e-8)
        }

