# =============================================================================
# ui/components/visualizations.py
"""
Visualization components for ILVE framework.
Handles charts, plots, and image displays.
"""

import streamlit as st
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from core.metrics.visualization import PlotGenerator

class VisualizationComponent:
    """Component for creating and displaying visualizations."""
    
    @staticmethod
    def render_image_grid(images: List[np.ndarray], 
                         titles: Optional[List[str]] = None,
                         cols: int = 4,
                         figsize: Tuple[int, int] = (12, 8)) -> None:
        """
        Render a grid of images.
        
        Args:
            images: List of images to display
            titles: Optional titles for each image
            cols: Number of columns in grid
            figsize: Figure size
        """
        if not images:
            st.warning("No images to display")
            return
        
        rows = (len(images) + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=figsize)
        
        if rows == 1:
            axes = axes.reshape(1, -1)
        elif cols == 1:
            axes = axes.reshape(-1, 1)
        
        for i, image in enumerate(images):
            row, col = i // cols, i % cols
            
            if len(image.shape) == 2: 
                axes[row, col].imshow(image, cmap='gray', interpolation='nearest')
            else: 
                axes[row, col].imshow(image, interpolation='nearest')
            
            axes[row, col].axis('off')
            
            if titles and i < len(titles):
                axes[row, col].set_title(titles[i], fontsize=10)
                
        for i in range(len(images), rows * cols):
            row, col = i // cols, i % cols
            axes[row, col].axis('off')
        
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
    
    @staticmethod
    def render_single_image(image: np.ndarray, 
                           title: str = "Generated Image",
                           figsize: Tuple[int, int] = (4, 4)) -> None:
        """
        Render a single image with styling.
        
        Args:
            image: Image to display
            title: Image title
            figsize: Figure size
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        if len(image.shape) == 2:  
            ax.imshow(image, cmap='gray', interpolation='nearest')
        else:
            ax.imshow(image, interpolation='nearest')
        
        ax.axis('off')
        ax.set_title(title, fontsize=16, pad=20, fontweight='bold')

        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(2)
            spine.set_edgecolor('#4CAF50')
        
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
    
    @staticmethod
    def render_variance_plot(values: List[float], 
                           variances: List[float],
                           dimension: int) -> None:
        """
        Render a variance analysis plot.
        
        Args:
            values: Latent values
            variances: Corresponding variance scores
            dimension: Dimension index
        """
        fig = PlotGenerator.create_variance_plot(values, variances, dimension)
        st.plotly_chart(fig, use_container_width=True)
    
    @staticmethod
    def render_2d_latent_plot(current_pos: Tuple[float, float],
                             history: Optional[List[Tuple[float, float]]] = None) -> None:
        """
        Render 2D latent space plot.
        
        Args:
            current_pos: Current position in latent space
            history: Optional history of positions
        """
        fig = PlotGenerator.create_2d_latent_space_plot(current_pos, history)
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
    
    @staticmethod
    def render_beta_tradeoff_plot(current_beta: float) -> None:
        """
        Render beta trade-off analysis plot.
        
        Args:
            current_beta: Current model's beta value
        """

        beta_values = [0.5, 1.0, 2.0, 4.0, 8.0, 16.0]
        reconstruction_scores = [98, 95, 90, 80, 60, 40]
        disentanglement_scores = [10, 30, 60, 80, 90, 95]
        
        fig = PlotGenerator.create_beta_tradeoff_plot(
            beta_values, reconstruction_scores, disentanglement_scores, current_beta
        )
        st.plotly_chart(fig, use_container_width=True)
    
    @staticmethod
    def render_analysis_summary(analysis_results: Dict[str, Any]) -> None:
        """
        Render summary of analysis results.
        
        Args:
            analysis_results: Dictionary of analysis results
        """
        st.markdown("#### 📊 Analysis Summary")
        
        cols = st.columns(3)
        

        total_images = sum(len(result.images) if hasattr(result, 'images') else 0 
                          for result in analysis_results.values())
        
        num_methods = len(analysis_results)
        

        avg_effect = 0.0
        effect_count = 0
        for result in analysis_results.values():
            if hasattr(result, 'metadata') and 'effect_strength' in result.metadata:
                avg_effect += result.metadata['effect_strength']
                effect_count += 1
        
        if effect_count > 0:
            avg_effect /= effect_count
        
        with cols[0]:
            st.metric("Images Generated", total_images)
        
        with cols[1]:
            st.metric("Analysis Methods", num_methods)
        
        with cols[2]:
            st.metric("Avg Effect Strength", f"{avg_effect:.3f}")

