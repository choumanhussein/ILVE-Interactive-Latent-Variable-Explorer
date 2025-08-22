"""
Model selector UI component for ILVE framework.
Extracted from the main app to enable reusability and better organization.
"""

import streamlit as st
from typing import Dict, Any, Optional, List
from core.models.loader import ModelLoader
from utils.state import get_session_state_manager

class ModelSelectorComponent:
    """Reusable model selector component."""
    
    def __init__(self):
        self.model_loader = ModelLoader()
        self.state_manager = get_session_state_manager()
    
    def render(self) -> Optional[Dict[str, Any]]:
        """
        Render the model selector component.
        
        Returns:
            Selected model info or None if no selection
        """
        st.markdown("### 📦 Pick Your AI Model!")
        st.markdown("""
        <div class="help-text">
            This is where you choose which "AI artist" you want to explore! Each model has learned to create images in a slightly different way.
        </div>
        """, unsafe_allow_html=True)
        

        available_models = self.model_loader.get_available_models()
        
        if not available_models:
            self._render_no_models_message()
            return None

        self._handle_default_model_selection(available_models)

        selected_model = self._render_model_selection_grid(available_models)
        
        return selected_model
    
    def _render_no_models_message(self):
        """Render message when no models are found."""
        st.markdown("""
        <div class="info-card">
            <h4>🚫 No Models Found</h4>
            <p>It looks like you haven't trained any VAE models yet. To get started, please run one of these commands in your terminal:</p>
            <pre><code>python train_basic_vae.py --epochs 15 --latent-dim 2 --no-wandb
python run_beta_experiments.py</code></pre>
            <p>Once trained, the model files will appear here!</p>
        </div>
        """, unsafe_allow_html=True)
    
    def _handle_default_model_selection(self, available_models: List[Dict[str, Any]]):
        """Handle automatic default model selection."""
        if (not self.state_manager.model.default_model_set and 
            available_models and 
            not self.state_manager.model.current_model_path):

            default_candidate = self._find_best_default_model(available_models)
            
            if default_candidate:
                self.state_manager.update_model_state(
                    current_model_path=default_candidate['path'],
                    default_model_set=True
                )
                st.info(f"Automatically selected **{default_candidate['display_name']}** for you to start exploring!")
                st.rerun()
    
    def _find_best_default_model(self, available_models: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Find the best default model to auto-select."""

        preferences = [
            lambda m: "beta" in m['name'].lower() and "latent_dim_2" in m['name'].lower() and m['beta'] == 4.0,
            lambda m: "beta" in m['name'].lower() and "latent_dim_2" in m['name'].lower(),
            lambda m: "latent_dim_2" in m['name'].lower(),
            lambda m: True 
        ]
        
        for preference in preferences:
            for model in available_models:
                if preference(model):
                    return model
        
        return available_models[0] if available_models else None
    
    def _render_model_selection_grid(self, available_models: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Render the model selection grid."""
        st.markdown("#### Here are the AI Models we found:")
        

        display_models = available_models[:10]
        cols = st.columns(min(len(display_models), 3))
        selected_model_info = None
        
        for i, model in enumerate(display_models):
            col_idx = i % 3
            with cols[col_idx]:

                is_current = (self.state_manager.model.current_model_path == model['path'])
                

                button_label = "✅ Selected" if is_current else "Select This Model"
                
                if st.button(
                    button_label,
                    key=f"select_model_{i}",
                    help=f"Load {model['name']}",
                    disabled=is_current
                ):
                    selected_model_info = model
                    self.state_manager.update_model_state(
                        current_model_path=selected_model_info['path'],
                        default_model_set=True
                    )
                    st.rerun()
                

                self._render_model_card(model, is_current)
        

        if self.state_manager.model.current_model_path:
            for model_item in available_models:
                if model_item['path'] == self.state_manager.model.current_model_path:
                    return model_item
        
        return None
    
    def _render_model_card(self, model: Dict[str, Any], is_current: bool):
        """Render individual model card."""
        card_border_style = "border: 2px solid #4CAF50;" if is_current else ""
        

        display_beta = model['beta']
        if is_current and hasattr(self.state_manager.model, 'loaded_model_beta'):
            display_beta = self.state_manager.model.loaded_model_beta
        
        st.markdown(f"""
        <div class="metric-card" style="{card_border_style}">
            <div class="metric-label">{model['type']}</div>
            <div style="font-weight: 600; margin: 0.5rem 0; font-size: 0.9rem;">
                {model['name'][:30]}{'...' if len(model['name']) > 30 else ''}
            </div>
            <div style="color: #6b7280; font-size: 0.8rem;">
                Size: {model['size_mb']:.1f} MB<br>
                Trained: {model['modified'].strftime('%Y-%m-%d')}<br>
                β: {display_beta:.1f}
            </div>
        </div>
        """, unsafe_allow_html=True)

class ModelInfoDisplay:
    """Component for displaying loaded model information."""
    
    @staticmethod
    def render_model_metrics(model, beta: float, config: Dict[str, Any]):
        """Render model information metrics."""
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Latent Dimensions</div>
                <div class="metric-value">{model.latent_dim}</div>
                <div style="color: #6b7280; font-size: 0.8rem;">(Your AI's "control knobs")</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            param_count = sum(p.numel() for p in model.parameters())
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Parameters</div>
                <div class="metric-value">{param_count:,}</div>
                <div style="color: #6b7280; font-size: 0.8rem;">(Complexity of the AI)</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">β Value</div>
                <div class="metric-value">{beta:.1f}</div>
                <div style="color: #6b7280; font-size: 0.8rem;">(How "organized" the knobs are)</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            image_dims = config.get('image_dims', (28, 28))
            num_channels = config.get('num_channels', 1)
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Image Size</div>
                <div class="metric-value">{image_dims[0]}x{image_dims[1]}x{num_channels}</div>
                <div style="color: #6b7280; font-size: 0.8rem;">(Size of images it creates)</div>
            </div>
            """, unsafe_allow_html=True)
    
    @staticmethod
    def render_sidebar_info(selected_model: Dict[str, Any], model, beta: float):
        """Render model information in sidebar."""
        st.sidebar.markdown("---")
        st.sidebar.markdown("### 📊 Your Current AI")
        st.sidebar.success(f"✅ Model: **{selected_model['type'] if selected_model else 'Generic VAE'}**")
        st.sidebar.info(f"📏 Control Knobs: **{model.latent_dim}**")
        st.sidebar.info(f"⚙️ Organization Effort (β): **{beta:.1f}**")
        

        state_manager = get_session_state_manager()
        image_dims = state_manager.model.model_image_dims
        num_channels = state_manager.model.model_num_channels
        st.sidebar.info(f"🖼️ Image Size: **{image_dims[0]}x{image_dims[1]}x{num_channels}**")

def render_model_selector() -> Optional[Dict[str, Any]]:
    """Convenience function to render model selector component."""
    selector = ModelSelectorComponent()
    return selector.render()

def render_model_info(model, beta: float, config: Dict[str, Any]):
    """Convenience function to render model information."""
    ModelInfoDisplay.render_model_metrics(model, beta, config)