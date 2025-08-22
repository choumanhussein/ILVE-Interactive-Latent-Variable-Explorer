"""
Interactive Explorer page for ILVE framework.
Real-time latent space exploration with Streamlit fragments.
"""

import streamlit as st
import torch
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from typing import Any, Dict
import time

from core.analysis.engine import AnalysisEngine
from ui.styles.css import apply_component_css

@st.fragment
def interactive_explorer_fragment(model, beta: float, analysis_engine: AnalysisEngine, state_manager):
    """
    Fragment for real-time interactive exploration.
    This fragment updates independently for better performance.
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


        for i in range(model.latent_dim):
            key = f"explorer_latent_dim_{i}"
            if key not in st.session_state:
                st.session_state[key] = float(np.random.normal(0, 1.0))


        btn_col1, btn_col2 = st.columns(2)
        with btn_col1:
            if st.button("🎲 Randomize Knobs", key="explorer_fragment_random", help="Set all control knobs to random values. See what new image appears!"):
                for i in range(model.latent_dim):
                    st.session_state[f"explorer_latent_dim_{i}"] = float(np.random.normal(0, 1.0))
                st.rerun()
        with btn_col2:
            if st.button("🔄 Reset Knobs to Center", key="explorer_fragment_reset", help="Set all control knobs to their neutral (zero) position."):
                for i in range(model.latent_dim):
                    st.session_state[f"explorer_latent_dim_{i}"] = 0.0
                st.rerun()

        st.markdown("---")  

       
        num_sliders = min(model.latent_dim, 8) if not state_manager.ui.user_preferences.get('show_advanced', False) else model.latent_dim
        
        current_latent_values = []
        for i in range(model.latent_dim):
            if i < num_sliders:

                val = st.slider(
                    f"Knob {i+1} (Dimension {i})",
                    min_value=-3.0,
                    max_value=3.0,
                    value=st.session_state[f"explorer_latent_dim_{i}"],
                    step=0.05,
                    key=f"explorer_slider_{i}",
                    help=f"Adjust 'Knob {i+1}' to see how this specific aspect of the generated image changes."
                )

                st.session_state[f"explorer_latent_dim_{i}"] = val
                current_latent_values.append(val)
            else:
                
                current_latent_values.append(st.session_state[f"explorer_latent_dim_{i}"])

        if model.latent_dim > num_sliders:
            st.info(f"Showing {num_sliders} of {model.latent_dim} control knobs. To see all of them, check 'Show Advanced Controls' in the sidebar.")
            
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

      
        try:

            image = analysis_engine._generate_image(current_latent_values)
            
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
                    st.warning(f"Image has {num_channels} channels. Displaying as grayscale for preview.")

                ax.axis('off')
                ax.set_title(f'Generated Image (β={beta})', fontsize=16, pad=20, fontweight='bold')
                for spine in ax.spines.values():
                    spine.set_visible(True)
                    spine.set_linewidth(2)
                    spine.set_edgecolor('#4CAF50')  
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)


                if st.button("💾 Save Current Image", key="explorer_fragment_save", help="Save the current image to your generation history for later review."):
                   
                    current_history = state_manager.analysis.generation_history
                    current_history.append({
                        'latent_code': current_latent_values,
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


    current_latent_values = [st.session_state[f"explorer_latent_dim_{i}"] for i in range(model.latent_dim)]
    state_manager.update_model_state(latent_values=current_latent_values)

def render_2d_visualization(model, state_manager):
    """Render 2D latent space visualization if applicable."""
    if model.latent_dim == 2:
        st.markdown("---")
        st.markdown("### 🗺️ Your Path in the AI's Imagination Map (2D Latent Space)")
        st.markdown("""
        <div class="help-text">
            Since your current AI model uses <strong>2 main "control knobs"</strong>, we can draw a map of its entire imagination! Every point on this map represents a unique image the AI can create.
            <br><br>
            The <strong>large blue dot</strong> shows your current "control knob" settings. The smaller purple dots are images you've recently generated.
        </div>
        """, unsafe_allow_html=True)

        fig = go.Figure()
        
        
        grid_range = np.linspace(-3, 3, 7)
        for val in grid_range:
            fig.add_hline(y=val, line_dash="dot", line_color="lightgray", opacity=0.3)
            fig.add_vline(x=val, line_dash="dot", line_color="lightgray", opacity=0.3)
        

        current_z1 = state_manager.model.latent_values[0] if len(state_manager.model.latent_values) > 0 else 0
        current_z2 = state_manager.model.latent_values[1] if len(state_manager.model.latent_values) > 1 else 0

        fig.add_trace(go.Scatter(
            x=[current_z1],
            y=[current_z2],
            mode='markers',
            marker=dict(
                size=25,  
                color='rgba(66, 165, 245, 0.9)',  
                line=dict(width=4, color='white'),
                symbol='circle'
            ),
            name='Current Spot',
            hovertemplate='Knob 1: %{x:.2f}<br>Knob 2: %{y:.2f}<extra></extra>'
        ))
        

        if len(state_manager.analysis.generation_history) > 1:
            history_z1 = [h['latent_code'][0] for h in state_manager.analysis.generation_history[-5:]]
            history_z2 = [h['latent_code'][1] for h in state_manager.analysis.generation_history[-5:]]
            
            fig.add_trace(go.Scatter(
                x=history_z1,
                y=history_z2,
                mode='markers+lines',
                marker=dict(size=10, color='rgba(156, 39, 176, 0.6)'), 
                line=dict(color='rgba(156, 39, 176, 0.4)', width=2),
                name='Recent Stops',
                hovertemplate='Previous spot<extra></extra>'
            ))
        
        fig.update_layout(
            title="Your Current Spot in the AI's Imagination Map",
            xaxis_title="Control Knob 1 (Latent Dimension 0)",
            yaxis_title="Control Knob 2 (Latent Dimension 1)",
            xaxis=dict(
                range=[-3.2, 3.2],
                gridcolor='rgba(76, 175, 80, 0.1)',  
                zerolinecolor='rgba(76, 175, 80, 0.3)',
                zerolinewidth=2
            ),
            yaxis=dict(
                range=[-3.2, 3.2],
                gridcolor='rgba(76, 175, 80, 0.1)',
                zerolinecolor='rgba(76, 175, 80, 0.3)',
                zerolinewidth=2
            ),
            plot_bgcolor='rgba(255, 255, 255, 0.9)',
            paper_bgcolor='rgba(255, 255, 255, 0)',
            font=dict(family="Inter, sans-serif", size=12),
            showlegend=True,
            legend=dict(
                x=0.02,
                y=0.98,
                bgcolor="rgba(255, 255, 255, 0.8)",
                bordercolor="rgba(76, 175, 80, 0.2)",
                borderwidth=1
            ),
            width=None,
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
    else:
        st.markdown("---")
        st.info("ℹ️ Your current AI model has more than 2 'control knobs', so we can't draw a simple 2D map. Try the 'Latent Space Analysis' tab for other ways to explore its higher-dimensional imagination!")

def render_interactive_explorer(model, beta: float, analysis_engine: AnalysisEngine, state_manager):
    """
    Main function to render the interactive explorer page.
    
    Args:
        model: Loaded VAE model
        beta: Beta value of the model
        analysis_engine: Core analysis engine
        state_manager: Session state manager
    """

    apply_component_css("interactive_explorer")
    
    st.markdown("""
    <div class="section-header">
        🎮 Interactive Latent Space Explorer
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-card">
        <h4>🚀 Real-time Image Creation! (It's like magic, powered by Streamlit Fragments)</h4>
        <p>This is where you can play directly with the AI's "imagination"! Adjust the **control knobs** (latent dimensions) on the left and see the new images appear instantly on the right. Every little tweak creates a new, unique image!</p>
    </div>
    """, unsafe_allow_html=True)
    

    interactive_explorer_fragment(model, beta, analysis_engine, state_manager)
    

    render_2d_visualization(model, state_manager)