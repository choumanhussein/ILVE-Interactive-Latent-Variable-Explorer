# =============================================================================
# core/metrics/visualization.py
"""
Visualization formatting for ILVE framework analysis results.
"""

import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from typing import List, Dict, Any, Optional, Tuple
import streamlit as st

class VisualizationFormatter:
    """Formats analysis results for visualization."""
    
    @staticmethod
    def format_traversal_results(images: List[np.ndarray], 
                                values: List[float],
                                dimension: int,
                                variance_scores: Optional[List[float]] = None) -> Dict[str, Any]:
        """
        Format traversal results for visualization.
        
        Args:
            images: Generated images from traversal
            values: Latent values used
            dimension: Dimension that was traversed
            variance_scores: Optional variance scores
            
        Returns:
            Formatted data for visualization
        """
        return {
            'images': images,
            'values': values,
            'dimension': dimension,
            'variance_scores': variance_scores or [np.var(img) for img in images],
            'num_steps': len(images),
            'value_range': (min(values), max(values)) if values else (0, 0)
        }
    
    @staticmethod
    def format_interaction_grid(images: List[List[np.ndarray]], 
                               values1: List[float],
                               values2: List[float],
                               dim1: int,
                               dim2: int) -> Dict[str, Any]:
        """
        Format interaction analysis results for grid visualization.
        
        Args:
            images: 2D grid of images
            values1: Values for first dimension
            values2: Values for second dimension
            dim1: First dimension index
            dim2: Second dimension index
            
        Returns:
            Formatted data for grid visualization
        """
        return {
            'image_grid': images,
            'values1': values1,
            'values2': values2,
            'dim1': dim1,
            'dim2': dim2,
            'grid_size': (len(images), len(images[0]) if images else 0)
        }

class PlotGenerator:
    """Generates plots for analysis results."""
    
    @staticmethod
    def create_variance_plot(values: List[float], 
                           variances: List[float],
                           dimension: int,
                           title: Optional[str] = None) -> go.Figure:
        """
        Create a variance plot for dimension traversal.
        
        Args:
            values: Latent values
            variances: Corresponding variance scores
            dimension: Dimension index
            title: Optional plot title
            
        Returns:
            Plotly figure
        """
        if not title:
            title = f"Effect of Dimension {dimension}"
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=values,
            y=variances,
            mode='lines+markers',
            name=f'Dimension {dimension}',
            line=dict(color='rgba(66, 165, 245, 0.8)', width=3),
            marker=dict(size=8)
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title=f'Latent Value (Dimension {dimension})',
            yaxis_title='Image Variance',
            font=dict(family="Inter, sans-serif"),
            plot_bgcolor='rgba(255, 255, 255, 0.9)',
            height=400
        )
        
        return fig
    
    @staticmethod
    def create_2d_latent_space_plot(current_position: Tuple[float, float],
                                   history_positions: Optional[List[Tuple[float, float]]] = None,
                                   title: str = "2D Latent Space") -> go.Figure:
        """
        Create a 2D latent space visualization.
        
        Args:
            current_position: Current position in latent space
            history_positions: Optional history of positions
            title: Plot title
            
        Returns:
            Plotly figure
        """
        fig = go.Figure()
        
        # Add grid lines
        grid_range = np.linspace(-3, 3, 7)
        for val in grid_range:
            fig.add_hline(y=val, line_dash="dot", line_color="lightgray", opacity=0.3)
            fig.add_vline(x=val, line_dash="dot", line_color="lightgray", opacity=0.3)
        
        # Add current position
        fig.add_trace(go.Scatter(
            x=[current_position[0]],
            y=[current_position[1]],
            mode='markers',
            marker=dict(
                size=25,
                color='rgba(66, 165, 245, 0.9)',
                line=dict(width=4, color='white'),
                symbol='circle'
            ),
            name='Current Position',
            hovertemplate=f'Dim 0: {current_position[0]:.2f}<br>Dim 1: {current_position[1]:.2f}<extra></extra>'
        ))
        
        # Add history if provided
        if history_positions and len(history_positions) > 0:
            hist_x = [pos[0] for pos in history_positions]
            hist_y = [pos[1] for pos in history_positions]
            
            fig.add_trace(go.Scatter(
                x=hist_x,
                y=hist_y,
                mode='markers+lines',
                marker=dict(size=10, color='rgba(156, 39, 176, 0.6)'),
                line=dict(color='rgba(156, 39, 176, 0.4)', width=2),
                name='History',
                hovertemplate='Previous position<extra></extra>'
            ))
        
        fig.update_layout(
            title=title,
            xaxis_title="Latent Dimension 0",
            yaxis_title="Latent Dimension 1",
            xaxis=dict(range=[-3.2, 3.2], gridcolor='rgba(76, 175, 80, 0.1)'),
            yaxis=dict(range=[-3.2, 3.2], gridcolor='rgba(76, 175, 80, 0.1)'),
            plot_bgcolor='rgba(255, 255, 255, 0.9)',
            font=dict(family="Inter, sans-serif"),
            showlegend=True,
            height=500
        )
        
        return fig
    
    @staticmethod
    def create_beta_tradeoff_plot(beta_values: List[float],
                                 reconstruction_scores: List[float],
                                 disentanglement_scores: List[float],
                                 current_beta: float) -> go.Figure:
        """
        Create a beta trade-off visualization.
        
        Args:
            beta_values: List of beta values
            reconstruction_scores: Corresponding reconstruction scores
            disentanglement_scores: Corresponding disentanglement scores
            current_beta: Current model's beta value
            
        Returns:
            Plotly figure
        """
        fig = go.Figure()
        
        # Add reconstruction quality line
        fig.add_trace(go.Scatter(
            x=beta_values,
            y=reconstruction_scores,
            mode='lines+markers',
            name='Image Quality Score',
            line=dict(color='rgba(239, 68, 68, 0.8)', width=3),
            marker=dict(size=8)
        ))
        
        # Add disentanglement score line
        fig.add_trace(go.Scatter(
            x=beta_values,
            y=disentanglement_scores,
            mode='lines+markers',
            name='Disentanglement Score',
            line=dict(color='rgba(34, 197, 94, 0.8)', width=3),
            marker=dict(size=8)
        ))
        
        # Highlight current beta
        current_recon = np.interp(current_beta, beta_values, reconstruction_scores)
        current_disent = np.interp(current_beta, beta_values, disentanglement_scores)
        
        fig.add_trace(go.Scatter(
            x=[current_beta, current_beta],
            y=[current_recon, current_disent],
            mode='markers',
            name=f'Your Model (β={current_beta})',
            marker=dict(size=14, color='rgba(66, 165, 245, 0.9)', 
                       line=dict(width=3, color='white'))
        ))
        
        fig.update_layout(
            title="β-VAE Trade-off: Quality vs. Disentanglement",
            xaxis_title="β Value",
            yaxis_title="Score (%)",
            xaxis=dict(type='log' if max(beta_values) > 10 else 'linear'),
            yaxis=dict(range=[0, 100]),
            font=dict(family="Inter, sans-serif"),
            plot_bgcolor='rgba(255, 255, 255, 0.9)',
            height=450,
            showlegend=True
        )
        
        return fig