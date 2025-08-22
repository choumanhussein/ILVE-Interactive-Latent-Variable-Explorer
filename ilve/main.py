"""
ILVE Framework - Modular Entry Point (Fixed Image Paths with Debug)
"""

import streamlit as st
import sys
from pathlib import Path
import base64
import os

# Add the ilve directory to Python path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

# Import modular components
from core.models.loader import ModelLoader
from core.analysis.engine import AnalysisEngine
from ui.components.model_selector import render_model_selector, ModelInfoDisplay
from ui.styles.css import apply_custom_css
from ui.pages.explorer import render_interactive_explorer
from ui.pages.analysis import render_latent_space_analysis
from ui.pages.disentanglement import render_disentanglement_analysis
from ui.pages.summary import render_project_summary
from utils.state import init_session_state, get_session_state_manager
from config.settings import get_config_manager

# Page configuration with modern settings
st.set_page_config(
    page_title="Friendly VAE Explorer",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/choumanhussein/ILVE-Interactive-Latent-Variable-Explorer',
        'Report a bug': 'https://github.com/choumanhussein/ILVE-Interactive-Latent-Variable-Explorer',
        'About': "# Friendly VAE Explorer\nAn easy-to-understand tool for learning about VAE and β-VAE models."
    }
)

def get_image_base64(image_path):
    """Convert image to base64 for embedding in HTML."""
    try:
        if os.path.exists(image_path):
            with open(image_path, "rb") as img_file:
                return base64.b64encode(img_file.read()).decode()
        return None
    except Exception as e:
        st.error(f"Error loading image {image_path}: {e}")
        return None



def main():
    """Main application entry point."""
    

    state_manager = init_session_state()
    apply_custom_css()
    
    
    st.markdown("""
    <style>
    /* Force full width layout */
    .main .block-container {
        max-width: 100% !important;
        padding-left: 1rem !important;
        padding-right: 1rem !important;
        padding-top: 1rem !important;
    }
    
    /* Ensure tabs use full width */
    .stTabs [data-baseweb="tab-list"] {
        width: 100%;
    }
    
    .stTabs [data-baseweb="tab-panel"] {
        width: 100%;
    }
    
    /* Fix any constrained containers */
    div[data-testid="stVerticalBlock"] {
        width: 100%;
    }
    
    /* Make sure columns use available space */
    div[data-testid="column"] {
        width: 100% !important;
        flex: 1 1 auto !important;
    }
    </style>
    """, unsafe_allow_html=True)
    
    state_manager.model_loader = ModelLoader()
    
   
    st.markdown("""
    <div class="main-header">🧠 Friendly VAE Explorer</div>
    <div class="subtitle">Your Easy Guide to AI's Creative Imagination</div>
    """, unsafe_allow_html=True)
    

    st.markdown("""
    <div class="info-card">
        <h4>👋 Welcome, Future AI Explorer!</h4>
        <p>Ever wondered how AI can **create new images** or **understand complex ideas**? This app is your playground to find out!</p>
        <p>We'll help you play with **AI's "imagination"** in real-time, learn what its "control knobs" do, and understand the core ideas behind modern generative AI.</p>
        <p><strong>🎯 Ready to get started?</strong> First, let's pick an AI model to play with!</p>
    </div>
    """, unsafe_allow_html=True)
    
    
    st.sidebar.markdown("""
    <div style="text-align: center; margin-bottom: 2rem;">
        <h2 style="color: #4CAF50;">🎛️ Your AI Toolkit</h2>
        <p style="color: #666;">Control your exploration here</p>
    </div>
    """, unsafe_allow_html=True)
    
    model_loader = ModelLoader()
    selected_model = render_model_selector()
    
    model, beta, config = None, 1.0, {}
    if state_manager.model.current_model_path:
        try:
            model, beta, config = model_loader.load_model(state_manager.model.current_model_path)
            if model is not None:
                state_manager.update_model_state(loaded_model_beta=beta)
        except Exception as e:
            st.error(f"Failed to load model: {e}")
            return
    

    if model is None:
        st.warning("⚠️ No AI model is currently loaded. Please select one from the list above to enable the interactive features.")
        return
    
    ModelInfoDisplay.render_sidebar_info(selected_model, model, beta)
    

    with st.sidebar.expander("⚙️ App Settings", expanded=False):
        show_advanced = st.checkbox(
            "Show All Control Knobs (Advanced)", 
            value=state_manager.ui.user_preferences.get('show_advanced', False),
            key="pref_show_advanced",
            help="By default, we show fewer control knobs for simplicity. Check this to see all of them for models with many dimensions."
        )
        

        state_manager.update_ui_state(
            user_preferences={
                **state_manager.ui.user_preferences,
                'show_advanced': show_advanced
            }
        )
    
    
    analysis_engine = AnalysisEngine(model)
    
    with st.container():
        tabs = st.tabs([
            "🎮 Interactive Explorer", 
            "🔬 Latent Space Analysis", 
            "📈 Understanding 'Organized Knobs'",
            "🎓 Learning Hub", 
            "📋 Project Summary"
        ])
        
        with tabs[0]:
            with st.container():
                render_interactive_explorer(model, beta, analysis_engine, state_manager)
        
        with tabs[1]:
            with st.container():
                render_latent_space_analysis(model, analysis_engine, state_manager)
        
        with tabs[2]:
            with st.container():
                render_disentanglement_analysis(model, beta, analysis_engine, state_manager, model_loader)
        
        with tabs[3]:
            with st.container():
                st.info("🎓 Educational content coming soon!")
        
        with tabs[4]:
            with st.container():
                render_project_summary()
    
   
    st.markdown("---")
    
    
    current_dir = Path(__file__).parent
    naist_logo_path = current_dir / "naist_logo.avif"
    lab_logo_path = current_dir / "lab_logo.png"
    
  
    naist_logo_b64 = get_image_base64(naist_logo_path) if naist_logo_path.exists() else None
    lab_logo_b64 = get_image_base64(lab_logo_path) if lab_logo_path.exists() else None
    
    lab_logo_html = f'<img src="data:image/png;base64,{lab_logo_b64}" alt="UCSL Lab Logo">' if lab_logo_b64 else '<span style="color: #999;">[Lab Logo]</span>'
    naist_logo_html = f'<img src="data:image/png;base64,{naist_logo_b64}" alt="NAIST Logo">' if naist_logo_b64 else '<span style="color: #999;">[NAIST Logo]</span>'
    
    st.markdown(
        f"""
    <style>
    .affil-footer {{ text-align:center; color:#444; margin-top:1.5rem; font-size:0.95rem; line-height:1.6; }}
    .affil-footer .title {{ font-size:1.1rem; font-weight:700; margin-bottom:0.25rem; }}
    .affil-footer p {{ margin:0.25rem 0; }}
    .affil-footer .logos {{ margin-top:0.75rem; display:flex; justify-content:center; gap:1.25rem; align-items:center; flex-wrap:wrap; }}
    .affil-footer .logos img {{ height:36px; opacity:0.95; }}
    .affil-footer .logos img:hover {{ opacity:1; }}
    .affil-footer .logos span {{ font-style:italic; padding:0.5rem; }}
    </style>

    <div class="affil-footer">
    <div class="title">Friendly VAE Explorer</div>

    <p><em>By Chouman Hussein</em></p>

    <p><em>Ubiquitous Computing Systems Laboratory,<br>
    Nara Institute of Science and Technology (NAIST)</em></p>

    <div class="logos">
        <a href="https://ubi-lab.naist.jp/" target="_blank" rel="noopener">
        {lab_logo_html}
        </a>
        <a href="https://www.naist.jp/en/" target="_blank" rel="noopener">
        {naist_logo_html}
        </a>
    </div>
    </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()