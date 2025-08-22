# ui/components/latent_controls.py
"""
Latent control components for ILVE framework.
Handles slider controls and latent space manipulation.
"""

import streamlit as st
import numpy as np
from typing import List, Dict, Any, Optional, Callable
from utils.state import get_session_state_manager

class LatentControlsComponent:
    """Component for latent space control sliders."""
    
    def __init__(self, model, state_manager=None):
        self.model = model
        self.state_manager = state_manager or get_session_state_manager()
        self.latent_dim = model.latent_dim
    
    def render_control_panel(self, 
                           key_prefix: str = "latent",
                           max_sliders: Optional[int] = None,
                           on_change: Optional[Callable] = None) -> List[float]:
        """
        Render the latent control panel with sliders.
        
        Args:
            key_prefix: Prefix for slider keys
            max_sliders: Maximum number of sliders to show
            on_change: Callback function when values change
            
        Returns:
            Current latent values
        """
        st.markdown("""
        <div class="control-panel">
            <h4>🎛️ Latent Space Controls</h4>
        """, unsafe_allow_html=True)
        

        state_key = f"{key_prefix}_values"
        if state_key not in st.session_state:
            st.session_state[state_key] = [0.0] * self.latent_dim
        
        
        show_advanced = self.state_manager.ui.user_preferences.get('show_advanced', False)
        num_sliders = max_sliders or (self.latent_dim if show_advanced else min(8, self.latent_dim))
        

        col1, col2 = st.columns(2)
        with col1:
            if st.button("🎲 Randomize", key=f"{key_prefix}_random"):
                new_values = [float(np.random.normal(0, 1.0)) for _ in range(self.latent_dim)]
                st.session_state[state_key] = new_values

                for i in range(self.latent_dim):
                    slider_key = f"{key_prefix}_slider_{i}"
                    if slider_key in st.session_state:
                        st.session_state[slider_key] = new_values[i]
                if on_change:
                    on_change(new_values)
                st.rerun()
        
        with col2:
            if st.button("🔄 Reset", key=f"{key_prefix}_reset"):
                new_values = [0.0] * self.latent_dim
                st.session_state[state_key] = new_values

                for i in range(self.latent_dim):
                    slider_key = f"{key_prefix}_slider_{i}"
                    if slider_key in st.session_state:
                        st.session_state[slider_key] = new_values[i]
                if on_change:
                    on_change(new_values)
                st.rerun()
        
        st.markdown("---")
        

        current_values = st.session_state[state_key].copy()
        

        values_changed = False
        

        for i in range(min(num_sliders, self.latent_dim)):
            slider_key = f"{key_prefix}_slider_{i}"
            

            if slider_key not in st.session_state:
                st.session_state[slider_key] = current_values[i]
            

            val = st.slider(
                f"Dimension {i}",
                min_value=-3.0,
                max_value=3.0,
                value=float(st.session_state[slider_key]), 
                step=0.05,
                key=slider_key,
                help=f"Control latent dimension {i}"
            )
            

            if abs(val - current_values[i]) > 1e-6:
                current_values[i] = val
                values_changed = True
        

        if values_changed:
            st.session_state[state_key] = current_values
            if on_change:
                on_change(current_values)
        

        if self.latent_dim > num_sliders:
            st.info(f"Showing {num_sliders} of {self.latent_dim} dimensions. Enable 'Show Advanced' to see all.")
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        return current_values
    
    def render_preset_controls(self, key_prefix: str = "preset") -> Optional[List[float]]:
        """
        Render preset latent configurations.
        
        Args:
            key_prefix: Prefix for control keys
            
        Returns:
            Selected preset values or None
        """
        st.markdown("#### 🎯 Preset Configurations")
        
        presets = {
            "Origin (All Zeros)": [0.0] * self.latent_dim,
            "Random Normal": [float(np.random.normal(0, 1)) for _ in range(self.latent_dim)],
            "Positive Bias": [0.5] * self.latent_dim,
            "Negative Bias": [-0.5] * self.latent_dim,
            "Extreme Positive": [2.0] * self.latent_dim,
            "Extreme Negative": [-2.0] * self.latent_dim
        }
        
        selected_preset = st.selectbox(
            "Choose a preset configuration:",
            list(presets.keys()),
            key=f"{key_prefix}_selector"
        )
        
        if st.button("Apply Preset", key=f"{key_prefix}_apply"):
            return presets[selected_preset]
        
        return None