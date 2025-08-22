# =============================================================================
# ui/pages/disentanglement.py
"""
Disentanglement Analysis page for ILVE framework.
β-VAE analysis and disentanglement metrics with real model comparison.
"""

import streamlit as st
import torch
import numpy as np
import plotly.graph_objects as go
from typing import Dict, Any, List, Optional
from core.analysis.engine import AnalysisEngine
from core.analysis.disentanglement import DisentanglementAnalyzer
from ui.components.visualizations import VisualizationComponent
from ui.components.metrics_display import MetricsDisplayComponent
from core.models.loader import ModelLoader

def render_disentanglement_analysis(model, beta: float, analysis_engine: AnalysisEngine, state_manager, model_loader=None):
    """
    Render the disentanglement analysis page.
    
    Args:
        model: Loaded VAE model
        beta: Beta value of the model
        analysis_engine: Core analysis engine
        state_manager: Session state manager
        model_loader: Optional ModelLoader instance (will create if not provided)
    """
    st.markdown("""
    <div class="section-header">
        📈 Understanding "Organized Knobs" (Disentanglement)
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown(f"""
    <div class="info-card">
        <h4>🧬 What is Disentanglement?</h4>
        <p>Imagine each "control knob" (latent dimension) changes **only one specific feature** of the image, like *just* the thickness of a digit, or *just* its slant. When this happens, we say the latent space is **"disentangled"** (or organized).</p>
        <p>Your current AI model is a **β-VAE** with a **β value of {beta}**. The β (beta) value controls how hard the AI tries to make its knobs disentangled. Higher β means more organized knobs, but sometimes the images might look a bit worse. It's a trade-off!</p>
        <p><strong>Heads Up:</strong> To see the detailed analysis grids, you'll need to click the "Generate" buttons.</p>
    </div>
    """, unsafe_allow_html=True)
    

    st.markdown("#### 📊 Quick Check: How Organized are Your AI's Knobs?")
    render_disentanglement_metrics_dashboard(beta)
    

    render_beta_comparison_section(beta)
    
    
    add_real_beta_comparison_to_page(model, beta, state_manager, model_loader)
    

    render_dimension_analysis_section(model, analysis_engine)

def render_disentanglement_metrics_dashboard(beta: float):
    """Render the metrics dashboard for disentanglement analysis."""

    interpretability_score = min(beta / 4.0, 1.0) * 100 if beta >= 1.0 else (beta / 1.0 * 20)
    reconstruction_quality = max(100 - (beta - 1.0) * 10, 30) if beta >= 1.0 else 98
    sparsity_score = min(beta * 10, 100)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">β Value</div>
            <div class="metric-value">{beta:.1f}</div>
            <div style="color: #6b7280; font-size: 0.8rem;">(Your model's "organization effort")</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Knob Organization</div>
            <div class="metric-value">{interpretability_score:.0f}%</div>
            <div style="color: #6b7280; font-size: 0.8rem;">(How well each knob controls one feature)</div>
        </div>
        """, unsafe_allow_html=True)
        st.progress(interpretability_score / 100)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Image Quality</div>
            <div class="metric-value">{reconstruction_quality:.0f}%</div>
            <div style="color: #6b7280; font-size: 0.8rem;">(How clear and realistic the images are)</div>
        </div>
        """, unsafe_allow_html=True)
        st.progress(reconstruction_quality / 100)
    
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Knob Usage</div>
            <div class="metric-value">{sparsity_score:.0f}%</div>
            <div style="color: #6b7280; font-size: 0.8rem;">(How many knobs are actually doing something useful)</div>
        </div>
        """, unsafe_allow_html=True)
        st.progress(sparsity_score / 100)

def render_beta_comparison_section(beta: float):
    """Render beta value comparison and trade-off analysis."""
    st.markdown("#### 🔄 The Great β-Value Trade-off (Theoretical)")
    st.markdown("""
    <div class="help-text">
        This chart shows you the typical relationship between the **β value** (how much the AI tries to organize its knobs) and the resulting **image quality** vs. **knob organization**.
        <br><br>
        <strong>Move the slider below</strong> to see how these two factors change with different hypothetical β values.
    </div>
    """, unsafe_allow_html=True)
    
    
    beta_illustrative = st.slider(
        "Explore different hypothetical β values", 
        0.5, 16.0, 4.0, 0.1,
        help="Drag to see how image quality and knob organization typically change.",
        key="beta_illustrative_slider"
    )
    

    render_beta_tradeoff_plot(beta, beta_illustrative)
    

    render_beta_description(beta_illustrative, beta)

def render_beta_tradeoff_plot(current_beta: float, illustrative_beta: float):
    """Render the interactive beta trade-off plot."""
    
    beta_data = {
        'β': [0.5, 1.0, 2.0, 4.0, 8.0, 16.0],
        'Reconstruction': [98, 95, 90, 80, 60, 40],
        'Disentanglement': [10, 30, 60, 80, 90, 95]
    }
    
    fig_beta = go.Figure()
    

    fig_beta.add_trace(go.Scatter(
        x=beta_data['β'],
        y=beta_data['Reconstruction'],
        mode='lines+markers',
        name='Image Quality Score',
        line=dict(color='rgba(239, 68, 68, 0.8)', width=3),
        marker=dict(size=8),
        hovertemplate='β: %{x}<br>Quality: %{y:.0f}%<extra></extra>'
    ))
    
    fig_beta.add_trace(go.Scatter(
        x=beta_data['β'],
        y=beta_data['Disentanglement'],
        mode='lines+markers',
        name='Knob Organization Score',
        line=dict(color='rgba(34, 197, 94, 0.8)', width=3),
        marker=dict(size=8),
        hovertemplate='β: %{x}<br>Organization: %{y:.0f}%<extra></extra>'
    ))
    
   
    current_recon_model = np.interp(current_beta, beta_data['β'], beta_data['Reconstruction'])
    current_disent_model = np.interp(current_beta, beta_data['β'], beta_data['Disentanglement'])
    
    fig_beta.add_trace(go.Scatter(
        x=[current_beta, current_beta],
        y=[current_recon_model, current_disent_model],
        mode='markers',
        name=f'Your Model (β={current_beta})',
        marker=dict(size=14, color='rgba(66, 165, 245, 0.9)', 
                     line=dict(width=3, color='white')),
        hovertemplate=f'Your Model at β={current_beta:.1f}<br>Quality: {current_recon_model:.0f}%<br>Organization: {current_disent_model:.0f}%<extra></extra>'
    ))

    
    illustrative_recon = np.interp(illustrative_beta, beta_data['β'], beta_data['Reconstruction'])
    illustrative_disent = np.interp(illustrative_beta, beta_data['β'], beta_data['Disentanglement'])

    fig_beta.add_trace(go.Scatter(
        x=[illustrative_beta],
        y=[illustrative_recon],
        mode='markers',
        name='Illustrative Point (Quality)',
        marker=dict(size=10, color='red', symbol='square', opacity=0.8),
        hovertemplate=f'Illustrative β={illustrative_beta:.1f}<br>Quality: {illustrative_recon:.0f}%<extra></extra>'
    ))
    
    fig_beta.add_trace(go.Scatter(
        x=[illustrative_beta],
        y=[illustrative_disent],
        mode='markers',
        name='Illustrative Point (Organization)',
        marker=dict(size=10, color='green', symbol='square', opacity=0.8),
        hovertemplate=f'Illustrative β={illustrative_beta:.1f}<br>Organization: {illustrative_disent:.0f}%<extra></extra>'
    ))
    
    fig_beta.update_layout(
        title="The Trade-off: Image Quality vs. Knob Organization",
        xaxis_title="β Value (How hard the AI tries to organize its knobs)",
        yaxis_title="Score (%)",
        xaxis=dict(type='log', dtick=np.log10(2), showgrid=True),
        yaxis=dict(range=[0, 100], showgrid=True),
        hovermode='closest',
        font=dict(family="Inter, sans-serif"),
        plot_bgcolor='rgba(255, 255, 255, 0.9)',
        height=450,
        legend=dict(x=0.01, y=0.99, bgcolor="rgba(255,255,255,0.7)", bordercolor="#ddd", borderwidth=1),
        margin=dict(t=50)
    )
    
    st.plotly_chart(fig_beta, use_container_width=True)

def render_beta_description(illustrative_beta: float, current_beta: float):
    """Render description for the selected beta value."""
    beta_data = {
        'β': [0.5, 1.0, 2.0, 4.0, 8.0, 16.0],
        'Description': [
            'Images are very clear, but knobs are mixed up.',
            'Standard VAE: Balanced quality and some organization.',
            'Images still good, knobs start to organize more.',
            'Good organization, images might show minor blur.',
            'Very organized knobs, but images might look blurry.',
            'Max organization, but images can look very abstract.'
        ]
    }

    interpolated_description = np.interp(illustrative_beta, beta_data['β'], range(len(beta_data['Description'])))
    closest_index = int(round(interpolated_description))
    closest_index = min(closest_index, len(beta_data['Description'])-1)
    
    st.markdown(f"""
    <div class="help-text">
        <p><strong>At this β value ({illustrative_beta:.1f}), generally:</strong> {beta_data['Description'][closest_index]}</p>
        <p>This shows the balance. Your loaded model is at β={current_beta:.1f}.</p>
    </div>
    """, unsafe_allow_html=True)

def add_real_beta_comparison_to_page(current_model, current_beta: float, state_manager, model_loader=None):
    """
    Call this function from your main disentanglement page to add the real comparison section.
    Insert it after your existing beta slider section.
    """

    if model_loader is None:
        if not hasattr(st.session_state, 'model_loader_instance'):
            st.session_state.model_loader_instance = ModelLoader()
        
    
    render_multi_beta_comparison_section(current_model, current_beta, model_loader)

def diagnose_model_loading_issue(model_loader, current_beta):
    """Diagnostic function to understand the model loading issue."""
    st.markdown("---")
    st.markdown("#### 🔍 Diagnostic Information")
    
    try:
        
        available_models = model_loader.get_available_models()
        st.write(f"**Total available models found:** {len(available_models)}")
        
        if len(available_models) == 0:
            st.error("❌ No models found! Check your checkpoint directory.")
            st.info("Expected directories: experiments/checkpoints, checkpoints, ., ..")
            return
        

        st.write("**Available models (first 5):**")
        for i, model_info in enumerate(available_models[:5]):
            st.write(f"  {i+1}. {model_info}")
        
        
        beta_models = {}
        all_model_types = []
        for model_info in available_models:
            all_model_types.append(model_info['type'])
            if model_info['type'] == 'β-VAE':
                beta_val = model_info['beta']
                if beta_val not in beta_models:
                    beta_models[beta_val] = model_info
        
        st.write(f"**Model types found:** {set(all_model_types)}")
        st.write(f"**β-VAE models found:** {len(beta_models)}")
        st.write(f"**Available β values:** {sorted(beta_models.keys())}")
        st.write(f"**Current β:** {current_beta}")
        
        if len(beta_models) < 2:
            st.warning("⚠️ Need at least 2 β-VAE models for comparison")
            st.info("Your β-VAE models:")
            for beta, model_info in beta_models.items():
                st.write(f"  - β={beta}: {model_info['name']}")
        else:
            st.success(f"✅ Found {len(beta_models)} β-VAE models - comparison should work!")
        

        if beta_models:
            test_beta, test_model_info = list(beta_models.items())[0]
            st.write(f"**Testing load of β={test_beta} model...**")
            try:
                model, beta, config = model_loader.load_model(test_model_info['path'])
                if model is not None:
                    st.success(f"✅ Successfully loaded test model: β={beta}, latent_dim={model.latent_dim}")
                else:
                    st.error("❌ Model loading returned None")
            except Exception as e:
                st.error(f"❌ Failed to load test model: {str(e)}")
        
    except Exception as e:
        st.error(f"❌ Diagnostic failed: {str(e)}")
        st.exception(e)

def render_multi_beta_comparison_section(current_model, current_beta: float, model_loader):
    """
    Full width version of the multi-beta comparison section.
    """
    st.markdown("---")
    st.markdown("#### 🖼️ Real β Trade-off: See It in Your Actual Models!")
    
   
    available_models = model_loader.get_available_models()
    

    beta_models = {}
    for model_info in available_models:
        if model_info['type'] == 'β-VAE':
            beta_val = model_info['beta']
            if beta_val not in beta_models:
                beta_models[beta_val] = model_info
    
    available_betas = sorted(beta_models.keys())
    
    if len(available_betas) < 2:
        st.info(f"""
        🔍 **Found {len(available_betas)} β-VAE model(s):** {available_betas if available_betas else 'None'}
        
        To see the real β trade-off comparison, you need at least 2 models with different β values.
        Your available models: {[m['display_name'] for m in available_models[:3]]}...
        """)
        return
    
    st.markdown(f"""
    <div class="info-card">
        <h4>🎯 Amazing! You have multiple β-VAE models!</h4>
        <p><strong>Available β values:</strong> {', '.join(map(str, available_betas))}</p>
        <p><strong>Currently selected:</strong> β = {current_beta}</p>
        <p>Let's generate the <em>exact same image</em> using each model to see the real trade-off!</p>
    </div>
    """, unsafe_allow_html=True)
    
    
    with st.container():

        comparison_type = st.selectbox(
            "What type of comparison?",
            ["Same Random Sample", "Latent Traversal", "Quality Analysis"],
            help="Choose how to compare your models"
        )
        
       
        if comparison_type == "Same Random Sample":
            render_same_sample_comparison(beta_models, available_betas, current_beta, current_model, model_loader)
        elif comparison_type == "Latent Traversal":
            render_traversal_comparison(beta_models, available_betas, current_beta, current_model, model_loader)
        else:
            render_quality_analysis_comparison(beta_models, available_betas, current_beta, current_model, model_loader)

def render_same_sample_comparison(beta_models: Dict, available_betas: List, current_beta: float, current_model, model_loader):
    """Generate the same sample across all beta models."""
    st.markdown("##### 🎲 Same Latent Code → Different β Models")
    
    
    st.markdown("""
    <style>
    .full-width-container {
        width: 100% !important;
        max-width: 100% !important;
        margin: 0 auto;
        padding: 0;
    }
    </style>
    """, unsafe_allow_html=True)
    

    with st.container():
        st.markdown("**🎛️ Controls**")

        seed = st.number_input(
            "Random Seed",
            value=42,
            help="Same seed = reproducible comparison"
        )

        latent_strength = st.slider(
            "Sample Intensity",
            0.5, 2.0, 1.0, 0.1,
            help="Higher = more extreme/interesting samples"
        )

        st.markdown("<br>", unsafe_allow_html=True)  

        if st.button("🎨 Generate Real Comparison", type="primary", use_container_width=True):
            generate_real_beta_comparison(
                beta_models, available_betas, current_beta,
                current_model, model_loader, seed, latent_strength
            )






def load_models_with_minimal_output(beta_models: Dict, available_betas: List, current_beta: float, current_model, model_loader):
    """Load models with minimal UI output - just show which ones succeeded."""
    loaded_models = {}
    model_latent_dims = {}
    

    loaded_models[current_beta] = current_model
    model_latent_dims[current_beta] = current_model.latent_dim
    

    progress_container = st.container()
    
    with progress_container:
        st.markdown("**Loading your β-VAE models...**")

        for i, beta in enumerate(available_betas):
            if beta != current_beta:
                model_info = beta_models[beta]
                
               
                with st.expander(f"Loading β={beta} model...", expanded=False):
                    try:
                        model, _, config = model_loader.load_model(model_info['path'])
                        
                        if model is not None:
                            loaded_models[beta] = model
                            model_latent_dims[beta] = model.latent_dim
                            st.success(f"✅ β={beta} model loaded successfully!")
                        else:
                            st.error(f"❌ β={beta} model failed to load")
                    except Exception as e:
                        st.error(f"❌ β={beta} model loading failed: {str(e)}")
        
        # Summary
        successful_betas = list(loaded_models.keys())
        st.markdown(f"**Loading complete!** Successfully loaded {len(successful_betas)} models: β = {successful_betas}")
    
    return loaded_models, model_latent_dims

def generate_real_beta_comparison(beta_models: Dict, available_betas: List, current_beta: float, 
                                current_model, model_loader, seed: int, strength: float):
    """Generate images from the same latent code across all your trained models."""
    
   
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    
    loaded_models, model_latent_dims = load_models_with_minimal_output(beta_models, available_betas, current_beta, current_model, model_loader)
    
    if len(loaded_models) < 2:
        st.error(f"Could not load enough models for comparison. Only loaded {len(loaded_models)} models: {list(loaded_models.keys())}")
        return
    
    st.markdown("**🔍 Comparing Your Different β Models:**")
    
   
    unique_latent_dims = set(model_latent_dims.values())
    
    if len(unique_latent_dims) > 1:
        st.warning(f"""
        ⚠️ **Different Latent Dimensions Detected!**
        
        Your models have different latent space sizes:
        {', '.join([f'β={beta}: {dim}D' for beta, dim in model_latent_dims.items()])}
        
        We'll generate separate random samples for each model since they can't share the same latent code.
        """)
        

        generate_separate_samples_comparison(loaded_models, model_latent_dims, seed, strength, current_beta)
    else:

        common_latent_dim = list(unique_latent_dims)[0]
        latent_code = torch.randn(1, common_latent_dim) * strength
        
        st.info(f"✅ All models have {common_latent_dim}D latent space - using identical latent codes!")
        generate_same_latent_comparison(loaded_models, latent_code, current_beta)

def generate_separate_samples_comparison(loaded_models: Dict, model_latent_dims: Dict, 
                                       seed: int, strength: float, current_beta: float):
    """Generate separate random samples for models with different latent dimensions."""
    
    st.markdown("#### 📸 Comparison Results (Separate Samples):")
    
    cols = st.columns(len(loaded_models))
    images = {}
    quality_scores = {}
    
    for i, beta in enumerate(sorted(loaded_models.keys())):
        model = loaded_models[beta]
        latent_dim = model_latent_dims[beta]
        
        with cols[i]:
            try:
                
                torch.manual_seed(seed + int(beta * 10)) 
                latent_code = torch.randn(1, latent_dim) * strength
                
                with torch.no_grad():
                    model.eval()
                    generated_img = model.decode(latent_code)
                    
                    img_array = process_model_output(generated_img)
                

                images[beta] = img_array
                quality_scores[beta] = calculate_image_quality(img_array)
                

                st.image(img_array, caption=f"β = {beta}", use_container_width=True)
                st.caption(f"Latent Dim: {latent_dim}D")
                

                if abs(beta - current_beta) < 0.01:
                    st.markdown("** Your Selected Model**")
                

                organization_score = min(beta * 15, 95)
                st.markdown(f"""
                📊 **Quality:** {quality_scores[beta]:.1f}  
                🎯 **Organization:** {organization_score:.0f}%
                """)
                

                if beta <= 1.0:
                    st.markdown("🔴 Lower β: Clearer but less organized")
                elif beta <= 4.0:
                    st.markdown("🟡 Medium β: Balanced quality/organization")
                else:
                    st.markdown("🟢 High β: Organized but may be blurrier")
                
            except Exception as e:
                st.error(f"Error with β={beta}: {str(e)}")
    

    st.info("""
    **Note:** Since your models have different latent dimensions, each image was generated from a 
    separate (but similarly seeded) random sample. The comparison shows the general quality trends 
    rather than identical transformations.
    """)
    

    if len(images) >= 2:
        render_real_quality_analysis(images, quality_scores, list(loaded_models.keys()), current_beta)

def generate_same_latent_comparison(loaded_models: Dict, latent_code: torch.Tensor, current_beta: float):
    """Generate images using the same latent code (when all models have same latent dim)."""
    
    st.markdown("#### 📸 Comparison Results (Identical Latent Codes):")
    
    cols = st.columns(len(loaded_models))
    images = {}
    quality_scores = {}
    
    for i, beta in enumerate(sorted(loaded_models.keys())):
        model = loaded_models[beta]
        
        with cols[i]:
            try:
                with torch.no_grad():
                    model.eval()
                    generated_img = model.decode(latent_code)
                    
                    img_array = process_model_output(generated_img)
                

                images[beta] = img_array
                quality_scores[beta] = calculate_image_quality(img_array)
                

                st.image(img_array, caption=f"β = {beta}", use_container_width=True)
                

                if abs(beta - current_beta) < 0.01:
                    st.markdown("**Your Selected Model**")
                

                organization_score = min(beta * 15, 95)
                st.markdown(f"""
                📊 **Quality:** {quality_scores[beta]:.1f}  
                🎯 **Organization:** {organization_score:.0f}%
                """)
                

                if beta <= 1.0:
                    st.markdown("🔴 Lower β: Clearer but less organized")
                elif beta <= 4.0:
                    st.markdown("🟡 Medium β: Balanced quality/organization")
                else:
                    st.markdown("🟢 High β: Organized but may be blurrier")
                
            except Exception as e:
                st.error(f"Error with β={beta}: {str(e)}")
    

    if len(images) >= 2:
        render_real_quality_analysis(images, quality_scores, list(loaded_models.keys()), current_beta)

def process_model_output(generated_img):
    """Process model output tensor into displayable numpy array."""
    try:
        if isinstance(generated_img, torch.Tensor):
            img_array = generated_img.cpu().numpy().squeeze()
            

            if len(img_array.shape) == 4:  
                img_array = img_array[0]  
            
            if len(img_array.shape) == 3:
                if img_array.shape[0] in [1, 3]:  
                    img_array = np.transpose(img_array, (1, 2, 0))  
                elif img_array.shape[-1] in [1, 3]:  
                    pass  
                elif img_array.shape[0] == img_array.shape[1]:  
                    pass  
                else:

                    total_pixels = np.prod(img_array.shape)
                    side_length = int(np.sqrt(total_pixels))
                    if side_length * side_length == total_pixels:
                        img_array = img_array.reshape(side_length, side_length)
            

            elif len(img_array.shape) == 1:
                total_pixels = img_array.shape[0]
                side_length = int(np.sqrt(total_pixels))
                if side_length * side_length == total_pixels:
                    img_array = img_array.reshape(side_length, side_length)
                else:

                    if total_pixels == 784:  
                        img_array = img_array.reshape(28, 28)
                    elif total_pixels == 1024:  
                        img_array = img_array.reshape(32, 32)
                    elif total_pixels == 4096:  
                        img_array = img_array.reshape(64, 64)
                    else:
                        raise ValueError(f"Cannot reshape 1D array of size {total_pixels} to image")
            

            if len(img_array.shape) == 3 and img_array.shape[-1] == 1:
                img_array = img_array.squeeze(-1)
            

            if len(img_array.shape) not in [2, 3]:
                raise ValueError(f"Final image shape {img_array.shape} is not 2D or 3D")
            

            if img_array.max() > 1.0 or img_array.min() < 0.0:
                img_array = np.clip((img_array - img_array.min()) / (img_array.max() - img_array.min()), 0, 1)
            
            return img_array
        else:
            return generated_img
            
    except Exception as e:

        if isinstance(generated_img, torch.Tensor):
            shape_info = f"Tensor shape: {generated_img.shape}"
        else:
            shape_info = f"Type: {type(generated_img)}"
        
        raise ValueError(f"Image processing failed - {shape_info}. Error: {str(e)}")


def render_traversal_comparison(beta_models: Dict, available_betas: List, current_beta: float, 
                              current_model, model_loader):
    """Show latent traversal across different beta models."""
    st.markdown("##### 🚶 Latent Walk: How Smooth is Each Model?")

    st.markdown("""
    <style>
    .full-width-container {
        width: 100% !important;
        max-width: 100% !important;
        margin: 0 auto;
        padding: 0;
    }
    </style>
    """, unsafe_allow_html=True)
 
    with st.container():
        st.markdown('<div class="full-width-container">', unsafe_allow_html=True)
        
        st.markdown("**🎛️ Traversal Controls**")

        dimension = st.selectbox(
            "Dimension to traverse", 
            range(min(8, current_model.latent_dim)), 
            help="Which latent dimension to vary"
        )

        traversal_range = st.slider("Range", 1.0, 3.0, 2.0, 0.5)

        num_steps = st.slider("Steps", 5, 9, 7)

        st.markdown("<br>", unsafe_allow_html=True)  

        if st.button("🚶 Start Real Traversal", use_container_width=True):
            generate_real_traversal_comparison(
                beta_models, available_betas, current_beta, 
                current_model, model_loader, dimension, traversal_range, num_steps
            )

def generate_real_traversal_comparison(beta_models: Dict, available_betas: List, current_beta: float,
                                     current_model, model_loader, dimension: int, traversal_range: float, num_steps: int):
    """Generate traversal comparison using your actual models."""
    

    latent_dim = current_model.latent_dim
    traversal_values = np.linspace(-traversal_range, traversal_range, num_steps)
    

    loaded_models, _ = load_models_with_minimal_output(beta_models, available_betas, current_beta, current_model, model_loader)
    

    for beta in sorted(loaded_models.keys()):
        model = loaded_models[beta]
        
        st.markdown(f"#### β = {beta} {'👈 Your Model' if abs(beta - current_beta) < 0.01 else ''}")
        
        images = []
        
        for val in traversal_values:
            latent = torch.zeros(1, model.latent_dim) 
            

            if dimension >= model.latent_dim:
                st.warning(f"⚠️ Dimension {dimension} doesn't exist in β={beta} model (has {model.latent_dim} dims). Using dimension 0 instead.")
                actual_dim = 0
            else:
                actual_dim = dimension
                
            latent[0, actual_dim] = val
            
            with torch.no_grad():
                model.eval()
                img = model.decode(latent)
                
                try:
                    img_array = process_model_output(img)
                    

                    if len(img_array.shape) not in [2, 3]:
                        st.error(f"❌ β={beta}, val={val:.1f}: Processed shape {img_array.shape} is invalid")
                        continue
                        
                    images.append(img_array)
                    
                except Exception as e:
                    st.error(f"❌ β={beta}, val={val:.1f}: Processing failed - {str(e)}")

                    placeholder = np.zeros((28, 28))
                    images.append(placeholder)
        

        if images and len(images) == num_steps:
            cols = st.columns(num_steps)
            for i, (img, val) in enumerate(zip(images, traversal_values)):
                with cols[i]:
                    try:
                        st.image(img, caption=f"{val:.1f}", use_container_width=True)
                    except Exception as e:
                        st.error(f"Display error: {str(e)}")
                        st.write(f"Image shape: {img.shape if hasattr(img, 'shape') else 'No shape'}")
        else:
            st.error(f"❌ Could not generate valid images for β={beta}")
        

        if len(images) >= 2:
            try:
                smoothness = calculate_traversal_smoothness(images)
                
                if beta <= 1.0:
                    interpretation = "🔴 Lower β: May show abrupt changes or mixed features"
                elif beta <= 4.0:
                    interpretation = "🟡 Medium β: Good balance of smoothness and detail"
                else:
                    interpretation = "🟢 High β: Very smooth, highly organized features"
                
                st.markdown(f"**Smoothness Score:** {smoothness:.3f} | {interpretation}")
            except Exception as e:
                st.warning(f"Could not calculate smoothness: {str(e)}")
        
        st.markdown("---")

def render_quality_analysis_comparison(beta_models: Dict, available_betas: List, current_beta: float,
                                     current_model, model_loader):
    """Analyze quality differences across your models."""
    st.markdown("##### 📊 Model Quality Analysis")
    

    with st.container():
        st.markdown("**Ready to analyze all your β-VAE models?**")
        st.markdown(
            "This will generate multiple samples from each model and compare their quality metrics."
        )

        st.markdown("<br>", unsafe_allow_html=True) 

        if st.button("🔬 Analyze All Models", type="primary", use_container_width=True):
            analyze_all_models_quality(
                beta_models, available_betas, current_beta,
                current_model, model_loader
            )

def analyze_all_models_quality(beta_models: Dict, available_betas: List, current_beta: float,
                              current_model, model_loader):
    """Comprehensive quality analysis of all models."""
    

    loaded_models, _ = load_models_with_minimal_output(beta_models, available_betas, current_beta, current_model, model_loader)
    

    num_samples = 10
    all_results = {beta: {'quality': [], 'sharpness': [], 'variance': []} for beta in loaded_models.keys()}
    
    progress_bar = st.progress(0)
    
    for sample_idx in range(num_samples):
        torch.manual_seed(sample_idx + 100)  
        latent_code = torch.randn(1, current_model.latent_dim)
        
        for beta, model in loaded_models.items():
            with torch.no_grad():
                model.eval()
                img = model.decode(latent_code)
                
                if isinstance(img, torch.Tensor):
                    img_array = img.cpu().numpy().squeeze()
                    if len(img_array.shape) == 3 and img_array.shape[0] in [1, 3]:
                        img_array = np.transpose(img_array, (1, 2, 0))
                    if len(img_array.shape) == 3 and img_array.shape[-1] == 1:
                        img_array = img_array.squeeze(-1)
                    if img_array.max() > 1.0 or img_array.min() < 0.0:
                        img_array = np.clip((img_array - img_array.min()) / (img_array.max() - img_array.min()), 0, 1)
                    
                    all_results[beta]['quality'].append(calculate_image_quality(img_array))
                    all_results[beta]['sharpness'].append(calculate_image_sharpness(img_array))
                    all_results[beta]['variance'].append(np.var(img_array))
        
        progress_bar.progress((sample_idx + 1) / num_samples)

    render_comprehensive_quality_plot(all_results, current_beta)

def render_comprehensive_quality_plot(results: Dict, current_beta: float):
    """Create comprehensive quality comparison plot."""
    
    betas = sorted(results.keys())
    

    quality_means = [np.mean(results[beta]['quality']) for beta in betas]
    quality_stds = [np.std(results[beta]['quality']) for beta in betas]
    sharpness_means = [np.mean(results[beta]['sharpness']) for beta in betas]
    
    fig = go.Figure()
    

    fig.add_trace(go.Scatter(
        x=betas,
        y=quality_means,
        error_y=dict(type='data', array=quality_stds, visible=True),
        mode='lines+markers',
        name='Image Quality',
        line=dict(color='rgba(239, 68, 68, 0.8)', width=3),
        marker=dict(size=10)
    ))

    fig.add_trace(go.Scatter(
        x=betas,
        y=sharpness_means,
        mode='lines+markers',
        name='Image Sharpness',
        line=dict(color='rgba(34, 197, 94, 0.8)', width=3),
        marker=dict(size=10),
        yaxis='y2'
    ))
    

    current_quality = np.mean(results[current_beta]['quality'])
    current_sharpness = np.mean(results[current_beta]['sharpness'])
    
    fig.add_trace(go.Scatter(
        x=[current_beta],
        y=[current_quality],
        mode='markers',
        name=f'Your Model (β={current_beta})',
        marker=dict(size=15, color='rgba(66, 165, 245, 0.9)', 
                   line=dict(width=3, color='white')),
        showlegend=True
    ))
    
    fig.update_layout(
        title="Real Quality Analysis: Your Trained Models",
        xaxis_title="β Value (Your Actual Models)",
        yaxis=dict(title="Quality Score", side="left"),
        yaxis2=dict(title="Sharpness Score", side="right", overlaying="y"),
        height=500,
        hovermode='x unified'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Summary insights
    best_quality_beta = betas[np.argmax(quality_means)]
    best_sharpness_beta = betas[np.argmax(sharpness_means)]
    
    st.markdown(f"""
    ### 🎯 Analysis Results:
    - **Best Quality:** β = {best_quality_beta} (score: {max(quality_means):.1f})
    - **Sharpest Images:** β = {best_sharpness_beta} (score: {max(sharpness_means):.3f})
    - **Your Model:** β = {current_beta} (quality: {current_quality:.1f})
    - **Trade-off Confirmed:** {'✅' if best_quality_beta < best_sharpness_beta else '❓'} Lower β = better quality, higher β = more organized
    """)

def calculate_image_quality(image: np.ndarray) -> float:
    """Calculate image quality score."""

    contrast = np.std(image)
    

    dynamic_range = np.max(image) - np.min(image)
    

    if len(image.shape) == 2:
        edges = np.gradient(image)
        edge_strength = np.mean(np.sqrt(edges[0]**2 + edges[1]**2))
    else:
        gray = np.mean(image, axis=2) if len(image.shape) == 3 else image
        edges = np.gradient(gray)
        edge_strength = np.mean(np.sqrt(edges[0]**2 + edges[1]**2))
    

    quality = contrast * 30 + dynamic_range * 40 + edge_strength * 100
    return min(quality, 100)

def calculate_image_sharpness(image: np.ndarray) -> float:
    """Calculate image sharpness using Laplacian variance."""
    try:

        if len(image.shape) == 3:
            if image.shape[-1] == 3:  
                gray = np.mean(image, axis=2)
            elif image.shape[-1] == 1:  
                gray = image.squeeze(-1)
            else:  
                gray = np.mean(image, axis=2)
        elif len(image.shape) == 2:
            gray = image
        elif len(image.shape) == 1:

            total_pixels = image.shape[0]
            side_length = int(np.sqrt(total_pixels))
            if side_length * side_length == total_pixels:
                gray = image.reshape(side_length, side_length)
            else:

                if total_pixels == 784:
                    gray = image.reshape(28, 28)
                elif total_pixels == 1024:
                    gray = image.reshape(32, 32)
                else:
                    return 0.0 
        else:
            return 0.0  
        

        if len(gray.shape) != 2:
            return 0.0
        
        h, w = gray.shape
        

        if h < 3 or w < 3:
            return 0.0
        

        laplacian = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
        

        result = 0
        for i in range(1, h-1):
            for j in range(1, w-1):
                patch = gray[i-1:i+2, j-1:j+2]
                result += abs(np.sum(patch * laplacian))
        
        return result / ((h-2) * (w-2))
        
    except Exception as e:

        return 0.0

def calculate_traversal_smoothness(images: List[np.ndarray]) -> float:
    """Calculate traversal smoothness."""
    if len(images) < 2:
        return 0.0
    
    differences = []
    for i in range(1, len(images)):
        diff = np.mean(np.abs(images[i] - images[i-1]))
        differences.append(diff)
    
  
    smoothness = 1.0 / (1.0 + np.var(differences))
    return smoothness

def render_real_quality_analysis(images: Dict, quality_scores: Dict, available_betas: List, current_beta: float):
    """Render analysis of the generated comparison images."""
    st.markdown("---")
    st.markdown("#### 📊 Your Models' Performance Analysis")
    

    fig = go.Figure()
    
    betas = sorted(images.keys())
    qualities = [quality_scores[beta] for beta in betas]
    organization_scores = [min(beta * 15, 95) for beta in betas]
    

    fig.add_trace(go.Scatter(
        x=betas,
        y=qualities,
        mode='lines+markers',
        name='Measured Quality',
        line=dict(color='rgba(239, 68, 68, 0.8)', width=3),
        marker=dict(size=10)
    ))
    
   
    fig.add_trace(go.Scatter(
        x=betas,
        y=organization_scores,
        mode='lines+markers',
        name='Expected Organization',
        line=dict(color='rgba(34, 197, 94, 0.8)', width=3),
        marker=dict(size=10)
    ))
    

    if current_beta in quality_scores:
        fig.add_trace(go.Scatter(
            x=[current_beta, current_beta],
            y=[quality_scores[current_beta], min(current_beta * 15, 95)],
            mode='markers',
            name=f'Your Model (β={current_beta})',
            marker=dict(size=15, color='rgba(66, 165, 245, 0.9)', 
                       line=dict(width=3, color='white'))
        ))
    
    fig.update_layout(
        title="Trade-off Confirmed: Your Real Models",
        xaxis_title="β Value",
        yaxis_title="Score",
        height=400,
        hovermode='x unified'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    

    best_quality_beta = max(quality_scores.keys(), key=lambda k: quality_scores[k])
    highest_beta = max(betas)
    
    st.markdown(f"""
    ### 🏆 Results from YOUR Models:
    - **Best Image Quality:** β = {best_quality_beta} (score: {quality_scores[best_quality_beta]:.1f})
    - **Most Organized:** β = {highest_beta} (expected organization: {min(highest_beta * 15, 95):.0f}%)
    - **Trade-off Visible:** {'✅ Yes' if best_quality_beta < highest_beta else '❓ Unclear'} - Lower β gives better quality
    - **Your Choice:** β = {current_beta} balances quality ({quality_scores.get(current_beta, 'N/A')}) and organization
    """)

def render_dimension_analysis_section(model, analysis_engine: AnalysisEngine):
    """Render detailed per-dimension analysis section."""
    st.markdown("---")
    st.markdown("#### 🎯 Discover What Each Knob Controls (Per-Dimension Analysis)")
    st.markdown("""
    <div class="help-text">
        This section helps you figure out what each individual "control knob" in your AI model is responsible for. 
        We change one knob at a time, keeping all others neutral, and see how the image changes.
    </div>
    """, unsafe_allow_html=True)
    

    analysis_mode = st.selectbox(
        "Which type of analysis?",
        ["Comprehensive Grid (All Key Knobs)", "Individual Knob Focus"],
        help="Choose 'Comprehensive Grid' to see changes for multiple knobs at once, or 'Individual Knob Focus' to zoom in on just one."
    )
    
    if analysis_mode == "Comprehensive Grid (All Key Knobs)":
        st.markdown("**🔬 See Multiple Knobs at Once**")
        
        with st.form(key="comprehensive_grid_form"):
            max_dims = max(2, min(8, model.latent_dim))
            default_dims = min(4, model.latent_dim)
            
            if model.latent_dim < 1:
                st.warning("⚠️ Your model has 0 control knobs. This analysis needs at least one.")
                num_dims_to_show = 0
            elif model.latent_dim == 1:
                st.info("ℹ️ Your model has 1 control knob. We'll analyze just that one.")
                num_dims_to_show = 1
            else:
                num_dims_to_show = st.slider(
                    "How many key knobs to analyze (top ones)?", 
                    min_value=1, 
                    max_value=max_dims,
                    value=default_dims,
                    help="We'll show you the effect of moving the first few control knobs."
                )
            
            traversal_range = st.slider("How much to move the knob?", 1.0, 3.0, 2.0, 0.5)
            
            submit_comprehensive = st.form_submit_button("🎨 Generate Comprehensive Analysis")
            
            if submit_comprehensive:
                if num_dims_to_show > 0:
                    generate_comprehensive_grid_analysis(analysis_engine, num_dims_to_show, traversal_range)
                else:
                    st.error("Cannot perform analysis: Please select at least 1 dimension to analyze.")
    
    elif analysis_mode == "Individual Knob Focus":
        st.markdown("**🎯 Zoom in on a Single Knob**")
        
        with st.form(key="individual_knob_form"):
            focus_dim = st.selectbox(
                "Which Knob to Focus On?", 
                range(model.latent_dim),
                format_func=lambda x: f"Knob {x}"
            )
            
            analysis_detail = st.selectbox(
                "How detailed should the analysis be?", 
                ["Quick (7 steps)", "Standard (11 steps)", "Detailed (15 steps)"]
            )
            
            num_steps = {"Quick (7 steps)": 7, "Standard (11 steps)": 11, "Detailed (15 steps)": 15}[analysis_detail]
            
            submit_individual = st.form_submit_button("🔍 Analyze Individual Knob")
            
            if submit_individual:
                generate_individual_knob_analysis(analysis_engine, focus_dim, num_steps)

def generate_comprehensive_grid_analysis(analysis_engine: AnalysisEngine, num_dims: int, traversal_range: float):
    """Generate comprehensive grid analysis for multiple dimensions."""
    with st.spinner(f"Analyzing how {num_dims} knobs change images..."):
        try:
            results = {}
            
            for dim in range(num_dims):
                result = analysis_engine.individual_traversal(
                    dimension=dim,
                    traversal_range=traversal_range,
                    num_steps=7
                )
                
                results[f"dimension_{dim}"] = result
                
                
                st.markdown(f"**Knob {dim} Analysis:**")
                

                effect_strength = result.metadata.get('effect_strength', 0)
                if effect_strength > 0.015:
                    category = "Strong Effect (Likely Disentangled)"
                    color = "🟢"
                elif effect_strength > 0.005:
                    category = "Moderate Effect"
                    color = "🟡"
                else:
                    category = "Weak Effect (Redundant)"
                    color = "🔴"
                
                st.markdown(f"{color} **{category}** - Effect Strength: {effect_strength:.4f}")
                

                if result.images and hasattr(VisualizationComponent, 'render_image_grid'):
                    VisualizationComponent.render_image_grid(
                        result.images,
                        [f'{val:.1f}' for val in result.values],
                        cols=len(result.images),
                        figsize=(len(result.images) * 1.2, 1.5)
                    )
                else:
                    
                    cols = st.columns(len(result.images))
                    for i, (img, val) in enumerate(zip(result.images, result.values)):
                        with cols[i]:
                            st.image(img, caption=f'{val:.1f}', use_container_width=True)
                
                st.markdown("---")
            

            st.success(f"✅ Analysis complete for {num_dims} knobs!")
            
           
            if hasattr(MetricsDisplayComponent, 'render_analysis_metrics'):
                MetricsDisplayComponent.render_analysis_metrics(results)
            
        except Exception as e:
            st.error(f"Analysis failed: {str(e)}")
            with st.expander("Error Details"):
                st.exception(e)

def generate_individual_knob_analysis(analysis_engine: AnalysisEngine, focus_dim: int, num_steps: int):
    """Generate detailed analysis for individual dimension."""
    with st.spinner(f"Deep analysis of Knob {focus_dim} in progress..."):
        try:
            result = analysis_engine.individual_traversal(
                dimension=focus_dim,
                traversal_range=3.0,
                num_steps=num_steps
            )
            
            st.markdown(f"**Detailed Analysis of Knob {focus_dim}:**")
            
            if hasattr(VisualizationComponent, 'render_image_grid'):
                VisualizationComponent.render_image_grid(
                    result.images,
                    [f'Value: {val:.1f}\nVar: {var:.3f}' for val, var in zip(result.values, result.variance_scores)],
                    cols=min(5, len(result.images)),
                    figsize=(min(10, len(result.images) * 2), 6)
                )
            else:
                cols = st.columns(min(5, len(result.images)))
                for i, (img, val, var) in enumerate(zip(result.images, result.values, result.variance_scores)):
                    with cols[i % len(cols)]:
                        st.image(img, caption=f'Val: {val:.1f}\nVar: {var:.3f}', use_container_width=True)
            
            st.markdown("##### How Much Does This Knob Change the Image?")
            if hasattr(VisualizationComponent, 'render_variance_plot'):
                VisualizationComponent.render_variance_plot(
                    result.values, result.variance_scores, focus_dim
                )
            else:
                import matplotlib.pyplot as plt
                fig, ax = plt.subplots(figsize=(8, 4))
                ax.plot(result.values, result.variance_scores, 'o-')
                ax.set_xlabel(f'Knob {focus_dim} Value')
                ax.set_ylabel('Image Change (Variance)')
                ax.set_title(f'Effect of Knob {focus_dim}')
                st.pyplot(fig)
            
            if hasattr(MetricsDisplayComponent, 'render_analysis_metrics'):
                MetricsDisplayComponent.render_analysis_metrics({f"knob_{focus_dim}": result})
            
        except Exception as e:
            st.error(f"Individual analysis failed: {str(e)}")
            with st.expander("Error Details"):
                st.exception(e)