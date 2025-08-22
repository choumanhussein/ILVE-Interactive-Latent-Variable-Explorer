# =============================================================================
# ui/fragments/interactive_explorer.py
"""
Interactive explorer fragment for ILVE framework.
Real-time latent space manipulation using Streamlit fragments.
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from typing import Callable, List
import time

@st.fragment
def interactive_explorer_fragment(model, beta: float, generate_image_func: Callable, state_manager):
    """
    Fragment for real-time interactive exploration.
    This fragment updates independently for better performance.
    
    Args:
        model: Loaded VAE model
        beta: Beta value of the model
        generate_image_func: Function to generate images from latent codes
        state_manager: Session state manager
    """

    col_control, col_output = st.columns([1, 1])


    with col_control:
        st.markdown("""
        <div class="control-panel">
            <h4>🎛️ Your AI's Control Knobs</h4>
        """, unsafe_allow_html=True)
        st.markdown("""
        <div class="help-text">
            Think of these sliders as **"control knobs"** for your AI artist! Each knob might change a different aspect of the image, like thickness, slant, or even what digit it is.
            <br><br>
            **Drag a slider** to see the image change instantly!
        </div>
        """, unsafe_allow_html=True)

        if 'fragment_latent_values' not in st.session_state or \
           len(st.session_state.fragment_latent_values) != model.latent_dim:
            st.session_state.fragment_latent_values = [
                float(np.random.normal(0, 1.0))
                for _ in range(model.latent_dim)
            ]


        btn_col1, btn_col2 = st.columns(2)
        with btn_col1:
            if st.button("🎲 Randomize Knobs", key="fragment_random"):
                st.session_state.fragment_latent_values = [
                    float(np.random.normal(0, 1.0))
                    for _ in range(model.latent_dim)
                ]
                st.rerun()
        with btn_col2:
            if st.button("🔄 Reset Knobs to Center", key="fragment_reset"):
                st.session_state.fragment_latent_values = [0.0] * model.latent_dim
                st.rerun()

        st.markdown("---") 


        show_advanced = state_manager.ui.user_preferences.get('show_advanced', False)
        num_sliders = model.latent_dim if show_advanced else min(8, model.latent_dim)
        
        for i in range(num_sliders):
            val = st.slider(
                f"Knob {i+1} (Dimension {i})",
                min_value=-3.0,
                max_value=3.0,
                value=st.session_state.fragment_latent_values[i],
                step=0.05,
                key=f"fragment_slider_{i}",
                help=f"Adjust 'Knob {i+1}' to see how this specific aspect changes."
            )
            st.session_state.fragment_latent_values[i] = val

        if model.latent_dim > num_sliders:
            st.info(f"Showing {num_sliders} of {model.latent_dim} control knobs. Check 'Show Advanced' to see all.")
            
        st.markdown("</div>", unsafe_allow_html=True)


    with col_output:
        st.markdown("""
        <div class="image-display">
            <h4>🖼️ What the AI Creates!</h4>
        """, unsafe_allow_html=True)
        st.markdown("""
        <div class="help-text">
            This is the image your AI model generates based on the **"control knob"** settings you chose. Notice how it changes in real-time as you move the sliders!
        </div>
        """, unsafe_allow_html=True)

        full_latent = st.session_state.fragment_latent_values + [0.0] * (model.latent_dim - len(st.session_state.fragment_latent_values))
        
        try:
            image = generate_image_func(full_latent)
            
            if image is not None:
                img_h, img_w = state_manager.model.model_image_dims
                num_channels = state_manager.model.model_num_channels

                fig, ax = plt.subplots(figsize=(4, 4))
                
                if num_channels == 1:
                    ax.imshow(image, cmap='gray', interpolation='nearest')
                elif num_channels == 3:
                    ax.imshow(image, interpolation='nearest')
                else:
                    ax.imshow(image, cmap='gray', interpolation='nearest')
                    st.warning(f"Image has {num_channels} channels. Displaying as grayscale.")

                ax.axis('off')
                ax.set_title(f'Generated Image (β={beta})', fontsize=16, pad=20, fontweight='bold')
                

                for spine in ax.spines.values():
                    spine.set_visible(True)
                    spine.set_linewidth(2)
                    spine.set_edgecolor('#4CAF50')
                
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)

                if st.button("💾 Save Current Image", key="fragment_save"):

                    current_history = state_manager.analysis.generation_history
                    current_history.append({
                        'latent_code': full_latent,
                        'timestamp': time.time(),
                        'image_hash': hash(image.tobytes())
                    })
                    

                    if len(current_history) > 10:
                        current_history.pop(0)
                    
                    state_manager.update_analysis_state(generation_history=current_history)
                    st.success("Image successfully saved to history!")
            else:
                st.error("Failed to generate image. Please check your model.")
                
        except Exception as e:
            st.error(f"Error generating image: {e}")

        st.markdown("</div>", unsafe_allow_html=True)


    state_manager.update_model_state(latent_values=full_latent)