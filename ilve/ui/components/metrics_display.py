# =============================================================================
# ui/components/metrics_display.py
"""
Metrics display components for ILVE framework.
Handles display of quantitative analysis results.
"""

import streamlit as st
from typing import Dict, Any, List, Optional
from core.metrics.quantitative import DisentanglementMetrics

class MetricsDisplayComponent:
    """Component for displaying analysis metrics."""
    
    @staticmethod
    def render_model_metrics(model, beta: float, config: Dict[str, Any]) -> None:
        """
        Render model information metrics.
        
        Args:
            model: Loaded model
            beta: Beta value
            config: Model configuration
        """
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Latent Dimensions</div>
                <div class="metric-value">{model.latent_dim}</div>
                <div style="color: #6b7280; font-size: 0.8rem;">Control knobs</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            param_count = sum(p.numel() for p in model.parameters())
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Parameters</div>
                <div class="metric-value">{param_count:,}</div>
                <div style="color: #6b7280; font-size: 0.8rem;">Model complexity</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">β Value</div>
                <div class="metric-value">{beta:.1f}</div>
                <div style="color: #6b7280; font-size: 0.8rem;">Organization effort</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            image_dims = config.get('image_dims', (28, 28))
            num_channels = config.get('num_channels', 1)
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Image Size</div>
                <div class="metric-value">{image_dims[0]}×{image_dims[1]}×{num_channels}</div>
                <div style="color: #6b7280; font-size: 0.8rem;">Output dimensions</div>
            </div>
            """, unsafe_allow_html=True)
    
    @staticmethod
    def render_disentanglement_metrics(beta: float) -> None:
        """
        Render disentanglement analysis metrics.
        
        Args:
            beta: Beta value for trade-off analysis
        """
        st.markdown("#### 📊 Disentanglement Analysis")
        
        metrics = DisentanglementMetrics.beta_trade_off_analysis(beta)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">β Value</div>
                <div class="metric-value">{metrics['beta']:.1f}</div>
                <div style="color: #6b7280; font-size: 0.8rem;">Organization effort</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            score = metrics['disentanglement_score']
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Disentanglement</div>
                <div class="metric-value">{score:.0f}%</div>
                <div style="color: #6b7280; font-size: 0.8rem;">Organization quality</div>
            </div>
            """, unsafe_allow_html=True)
            st.progress(score / 100)
        
        with col3:
            quality = metrics['reconstruction_quality']
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Image Quality</div>
                <div class="metric-value">{quality:.0f}%</div>
                <div style="color: #6b7280; font-size: 0.8rem;">Reconstruction fidelity</div>
            </div>
            """, unsafe_allow_html=True)
            st.progress(quality / 100)
        
        with col4:
            balance = metrics['balance_score']
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Balance Score</div>
                <div class="metric-value">{balance:.0f}%</div>
                <div style="color: #6b7280; font-size: 0.8rem;">Overall performance</div>
            </div>
            """, unsafe_allow_html=True)
            st.progress(balance / 100)
    
    @staticmethod
    def render_analysis_metrics(results: Dict[str, Any]) -> None:
        """
        Render analysis-specific metrics.
        
        Args:
            results: Analysis results dictionary
        """
        if not results:
            st.info("No analysis results to display")
            return
        
        st.markdown("#### 🔍 Analysis Metrics")
        
        for method_name, result in results.items():
            with st.expander(f"📊 {method_name.replace('_', ' ').title()}"):
                if hasattr(result, 'metadata'):
                    metadata = result.metadata
                    

                    metric_cols = st.columns(3)
                    
                    with metric_cols[0]:
                        if 'effect_strength' in metadata:
                            st.metric("Effect Strength", f"{metadata['effect_strength']:.3f}")
                    
                    with metric_cols[1]:
                        if 'num_steps' in metadata:
                            st.metric("Steps Analyzed", metadata['num_steps'])
                        elif 'num_samples' in metadata:
                            st.metric("Samples Generated", metadata['num_samples'])
                    
                    with metric_cols[2]:
                        if 'mean_variance' in metadata:
                            st.metric("Mean Variance", f"{metadata['mean_variance']:.4f}")
                        elif 'diversity_score' in metadata:
                            st.metric("Diversity Score", f"{metadata['diversity_score']:.3f}")

                    if len(metadata) > 3:
                        st.markdown("**Additional Metrics:**")
                        for key, value in metadata.items():
                            if key not in ['effect_strength', 'num_steps', 'num_samples', 'mean_variance', 'diversity_score']:
                                if isinstance(value, (int, float)):
                                    st.write(f"- **{key.replace('_', ' ').title()}**: {value:.4f}")
                                else:
                                    st.write(f"- **{key.replace('_', ' ').title()}**: {value}")
    
    @staticmethod
    def render_comparison_metrics(model1_metrics: Dict[str, float],
                                 model2_metrics: Dict[str, float],
                                 model1_name: str = "Model 1",
                                 model2_name: str = "Model 2") -> None:
        """
        Render comparison metrics between two models.
        
        Args:
            model1_metrics: Metrics for first model
            model2_metrics: Metrics for second model
            model1_name: Name of first model
            model2_name: Name of second model
        """
        st.markdown("#### ⚖️ Model Comparison")
        

        common_metrics = set(model1_metrics.keys()) & set(model2_metrics.keys())
        
        if not common_metrics:
            st.warning("No common metrics found for comparison")
            return
        
        for metric in common_metrics:
            col1, col2, col3 = st.columns([1, 1, 1])
            
            val1 = model1_metrics[metric]
            val2 = model2_metrics[metric]
            difference = val2 - val1
            
            with col1:
                st.metric(f"{model1_name}", f"{val1:.3f}")
            
            with col2:
                st.metric(f"{model2_name}", f"{val2:.3f}")
            
            with col3:
                delta_color = "normal" if abs(difference) < 0.01 else ("inverse" if difference < 0 else "normal")
                st.metric("Difference", f"{val2:.3f}", f"{difference:+.3f}")
            
            st.markdown("---")