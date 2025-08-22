# ui/pages/analysis.py
"""
Latent Space Analysis page for ILVE framework.
Comprehensive analysis tools and visualizations.
"""

import streamlit as st
import numpy as np
from typing import Dict, Any
from core.analysis.engine import AnalysisEngine
from ui.components.visualizations import VisualizationComponent
from ui.components.metrics_display import MetricsDisplayComponent
from ui.styles.css import apply_component_css

def render_latent_space_analysis(model, analysis_engine: AnalysisEngine, state_manager):
    """
    Render the latent space analysis page.
    
    Args:
        model: Loaded VAE model
        analysis_engine: Core analysis engine
        state_manager: Session state manager
    """
    apply_component_css("analysis_results")
    
    st.markdown("""
    <div class="section-header">
        🔬 Deep Dive into AI's Imagination
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-card">
        <h4>🧪 Uncover Hidden Patterns in the AI's "Thought Space"</h4>
        <p>This section lets you see how your AI model organizes its understanding of images. It's like looking at the underlying rules it learned! You can see how smoothly it transforms images and even create animations between different concepts.</p>
        <p><strong>Heads Up:</strong> To generate these detailed analyses, you'll need to click the "Generate" buttons.</p>
    </div>
    """, unsafe_allow_html=True)
    
    if model.latent_dim == 2:
        render_2d_analysis(model, analysis_engine, state_manager)
    else:
        render_high_dimensional_analysis(model, analysis_engine, state_manager)

def render_2d_analysis(model, analysis_engine: AnalysisEngine, state_manager):
    """Render 2D-specific analysis tools."""
    st.markdown("#### 🗺️ The Complete 2D Imagination Map")
    st.markdown("""
    <div class="help-text">
        This map shows every possible image your AI can create when using only two "control knobs." 
        Think of it as a grid of all the AI's ideas!
    </div>
    """, unsafe_allow_html=True)
    
    with st.form(key="2d_map_form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            grid_resolution = st.selectbox(
                "Grid Detail (Images per row/column)", 
                [6, 8, 10, 12], 
                index=1,
                help="Choose how many images to show in each row and column."
            )
        with col2:
            latent_range = st.slider(
                "Exploration Range", 
                1.5, 4.0, 2.5, 0.5,
                help="How 'far out' in the AI's imagination map to explore."
            )
        with col3:
            show_grid_numbers = st.checkbox(
                "Show Values on Images", 
                False,
                help="Display the exact 'control knob' values for each image."
            )
        
        submit_map = st.form_submit_button("🎨 Generate Imagination Map")
        
        if submit_map:
            generate_2d_map(model, analysis_engine, grid_resolution, latent_range, show_grid_numbers)
    
    render_interpolation_studio(model, analysis_engine)

def render_interpolation_studio(model, analysis_engine: AnalysisEngine):
    """Render interpolation analysis for 2D models."""
    st.markdown("---")
    st.markdown("#### 🎯 Create a Smooth Morphing Animation (Interpolation Studio)")
    st.markdown("""
    <div class="help-text">
        See how your AI model smoothly transforms one idea into another! Pick a **Starting Point** and an 
        **Ending Point** on the imagination map, and the AI will show you the step-by-step morphing journey.
    </div>
    """, unsafe_allow_html=True)
    
    with st.form(key="interpolation_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**🎯 Start Image**")
            start_z1 = st.slider("Knob 1 (Start)", -3.0, 3.0, -2.0, 0.1, key="interp_start_z1")
            start_z2 = st.slider("Knob 2 (Start)", -3.0, 3.0, -2.0, 0.1, key="interp_start_z2")
            
            start_image = analysis_engine._generate_image([start_z1, start_z2])
            if start_image is not None:
                VisualizationComponent.render_single_image(start_image, "Start", (3, 3))
        
        with col2:
            st.markdown("**🏁 End Image**")
            end_z1 = st.slider("Knob 1 (End)", -3.0, 3.0, 2.0, 0.1, key="interp_end_z1")
            end_z2 = st.slider("Knob 2 (End)", -3.0, 3.0, 2.0, 0.1, key="interp_end_z2")
            
            end_image = analysis_engine._generate_image([end_z1, end_z2])
            if end_image is not None:
                VisualizationComponent.render_single_image(end_image, "End", (3, 3))
        
        col_int1, col_int2, col_int3 = st.columns(3)
        with col_int1:
            num_steps = st.slider("Number of Morphing Steps", 5, 20, 10)
        with col_int2:
            interpolation_method = st.selectbox(
                "Morphing Method", 
                ["Linear (Direct Path)", "Spherical (Curved Path)"]
            )
        with col_int3:
            show_path = st.checkbox("Show Path on Map", True)
        
        submit_interpolation = st.form_submit_button("🔄 Start Morphing!")
        
        if submit_interpolation:
            generate_interpolation_sequence(
                analysis_engine, [start_z1, start_z2], [end_z1, end_z2], 
                num_steps, interpolation_method, show_path
            )

def render_high_dimensional_analysis(model, analysis_engine: AnalysisEngine, state_manager):
    """Render analysis tools for high-dimensional models."""
    st.markdown("#### 🔢 Explore AI's Imagination with Many Knobs (Higher-Dimensional Analysis)")
    st.markdown(f"""
    <div class="help-text">
        Your AI model has **{model.latent_dim} control knobs**! We can't draw a simple 2D map for all of them, 
        but we can still explore how each knob changes the image, or try generating totally random images 
        to see the AI's full range of creativity.
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("**🔬 Choose Your Exploration Method**")
        selected_dims = st.multiselect(
            "Which Knobs to Focus On?",
            range(model.latent_dim),
            default=list(range(min(4, model.latent_dim))),
            format_func=lambda x: f"Knob {x}",
            help="Pick which specific control knobs you want to analyze."
        )
        
        analysis_type = st.radio(
            "How do you want to explore?",
            ["Individual Knob Traversal", "Knob Pair Interaction", "Random Image Generation"],
            help="Choose a method: Change one knob at a time, see how two knobs work together, or get random images."
        )
    
    with col2:
        if analysis_type == "Individual Knob Traversal":
            with st.form(key="individual_traversal_form"):
                st.markdown("**📊 Individual Knob Analysis Settings**")
                num_steps = st.slider("Number of steps per knob", 5, 15, 7)
                traversal_range = st.slider("Range to explore", 1.0, 4.0, 2.5)
                submit_individual = st.form_submit_button("📊 Start Individual Analysis!")
                
                if submit_individual and selected_dims:
                    generate_individual_analysis(analysis_engine, selected_dims, num_steps, traversal_range)
        
        elif analysis_type == "Knob Pair Interaction":
            with st.form(key="pair_interaction_form"):
                st.markdown("**🔄 Knob Pair Analysis Settings**")
                grid_size = st.slider("Grid size", 3, 7, 5)
                traversal_range = st.slider("Range to explore", 1.0, 4.0, 2.5)
                submit_pair = st.form_submit_button("🔄 Start Pair Analysis!")
                
                if submit_pair and len(selected_dims) >= 2:
                    generate_pair_analysis(analysis_engine, selected_dims, grid_size, traversal_range)
        
        elif analysis_type == "Random Image Generation":
            with st.form(key="random_generation_form"):
                st.markdown("**🎲 Random Generation Settings**")
                num_samples = st.slider("How many random images?", 4, 16, 8)
                sampling_std = st.slider("How 'wild' should the randomness be?", 0.5, 3.0, 1.5)
                submit_random = st.form_submit_button("✨ Generate Random Batch!")
                
                if submit_random:
                    generate_random_analysis(analysis_engine, num_samples, sampling_std)

def generate_individual_analysis(analysis_engine, selected_dims, num_steps, traversal_range):
    """Generate individual knob traversal analysis."""
    with st.spinner("Analyzing individual knobs..."):
        try:
            st.markdown("**📊 How Each Knob Changes the Image**")
            
            for dim in selected_dims[:4]:  
                result = analysis_engine.individual_traversal(dim, num_steps=num_steps, traversal_range=traversal_range)
                
                st.markdown(f"**Exploring Knob {dim}**")
                VisualizationComponent.render_image_grid(
                    result.images, 
                    [f'{val:.1f}' for val in result.values],
                    cols=len(result.images),
                    figsize=(len(result.images) * 1.2, 1.5)
                )
                
                VisualizationComponent.render_variance_plot(
                    result.values, result.variance_scores, dim
                )
            
            st.success("Individual knob analysis complete!")
            
        except Exception as e:
            st.error(f"Analysis failed: {str(e)}")

def generate_pair_analysis(analysis_engine, selected_dims, grid_size, traversal_range):
    """Generate knob pair interaction analysis."""
    with st.spinner("Analyzing knob pair interactions..."):
        try:
            st.markdown("**🔄 How Two Knobs Work Together**")
            
            dim1, dim2 = selected_dims[0], selected_dims[1]
            result = analysis_engine.pair_interaction(dim1, dim2, grid_size=grid_size, traversal_range=traversal_range)
            
            grid_images = []
            for i in range(grid_size):
                row = result.images[i*grid_size:(i+1)*grid_size]
                grid_images.extend(row)
            
            VisualizationComponent.render_image_grid(
                grid_images, cols=grid_size, figsize=(6, 6)
            )
            
            st.success(f"Interaction analysis between Knob {dim1} and Knob {dim2} complete!")
            
        except Exception as e:
            st.error(f"Analysis failed: {str(e)}")

def generate_random_analysis(analysis_engine, num_samples, sampling_std):
    """Generate random image analysis."""
    with st.spinner("Generating random images..."):
        try:
            st.markdown("**🎲 Generate Completely Random Images**")
            
            result = analysis_engine.random_generation(num_samples, sampling_std)
            
            VisualizationComponent.render_image_grid(
                result.images,
                [f'Random {i+1}' for i in range(len(result.images))],
                cols=4,
                figsize=(8, 6)
            )
            
            st.success("Random images generated!")
            MetricsDisplayComponent.render_analysis_metrics({'random_generation': result})
            
        except Exception as e:
            st.error(f"Analysis failed: {str(e)}")

def generate_2d_map(model, analysis_engine, grid_resolution, latent_range, show_grid_numbers):
    """Generate and display 2D imagination map."""
    with st.spinner(f"Creating your {grid_resolution}×{grid_resolution} imagination map..."):
        try:
            x = np.linspace(-latent_range, latent_range, grid_resolution)
            y = np.linspace(-latent_range, latent_range, grid_resolution)
            
            images = []
            titles = []
            
            for i, yi in enumerate(y):
                for j, xi in enumerate(x):
                    image = analysis_engine._generate_image([xi, yi])
                    if image is not None:
                        images.append(image)
                        if show_grid_numbers:
                            titles.append(f'({xi:.1f},{yi:.1f})')
                        else:
                            titles.append("")
            
            if images:
                VisualizationComponent.render_image_grid(
                    images, titles if show_grid_numbers else None, 
                    cols=grid_resolution, figsize=(grid_resolution * 1.2, grid_resolution * 1.2)
                )
                st.success(f"✅ Your map with {len(images)} images is ready!")
            else:
                st.error("Failed to generate images for the map.")
                
        except Exception as e:
            st.error(f"Error generating map: {str(e)}")

def generate_interpolation_sequence(analysis_engine, start_point, end_point, num_steps, method, show_path):
    """Generate and display interpolation sequence."""
    with st.spinner("Creating your smooth morphing sequence..."):
        try:
            from core.models.inference import ModelInference
            inference = ModelInference(analysis_engine.model)
            
            interpolation_codes = inference.interpolate_latent_codes(
                start_point, end_point, num_steps, 
                'linear' if 'Linear' in method else 'slerp'
            )
            
            images = []
            titles = []
            
            for i, code in enumerate(interpolation_codes):
                image = analysis_engine._generate_image(code)
                if image is not None:
                    images.append(image)
                    alpha = i / (num_steps - 1)
                    titles.append(f'Step {i+1}\n(α={alpha:.2f})')
            
            if images:
                VisualizationComponent.render_image_grid(
                    images, titles, cols=min(10, len(images)), 
                    figsize=(max(12, len(images) * 1.5), 4)
                )
                
                if show_path:
                    st.markdown("#### 🗺️ See the Morphing Path on the Map")
                    path_points = [(code[0].item(), code[1].item()) for code in interpolation_codes]
                    VisualizationComponent.render_2d_latent_plot(
                        path_points[-1], path_points[:-1]
                    )
            else:
                st.error("Failed to generate interpolation sequence.")
                
        except Exception as e:
            st.error(f"Error generating interpolation: {str(e)}")

def generate_high_dimensional_analysis(analysis_engine, selected_dims, analysis_type, total_dims):
    """Generate high-dimensional analysis results (legacy function kept for compatibility)."""
    pass