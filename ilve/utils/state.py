"""
Organized session state management for ILVE framework.
Maintains backward compatibility while providing better organization.
"""

import streamlit as st
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime

@dataclass
class ModelState:
    """State related to loaded models and their properties."""
    model_loaded: bool = False
    current_model_path: Optional[str] = None
    loaded_model_beta: float = 1.0
    model_image_dims: tuple = (28, 28)
    model_num_channels: int = 1
    latent_values: List[float] = field(default_factory=lambda: [0.0] * 10)
    fragment_latent_values: List[float] = field(default_factory=lambda: [0.0] * 10)
    default_model_set: bool = False

@dataclass
class UIState:
    """State related to UI preferences and settings."""
    user_preferences: Dict[str, Any] = field(default_factory=lambda: {
        'theme': 'modern',
        'show_advanced': False,
        'auto_generate': True
    })
    is_preview_generation: bool = False

@dataclass
class EducationState:
    """State related to educational progress and learning paths."""
    learning_progress: Dict[str, List[str]] = field(default_factory=lambda: {
        'beginner': [],
        'intermediate': [],
        'advanced': [],
        'expert': []
    })
    current_advanced_step: int = 1
    random_latent_point: Optional[List[float]] = None

@dataclass
class AnalysisState:
    """State related to analysis results and generation history."""
    generation_history: List[Dict[str, Any]] = field(default_factory=list)

class SessionStateManager:
    """
    Centralized session state manager that maintains backward compatibility
    with the original st.session_state key structure.
    """
    
    def __init__(self):
        self._initialize_states()
    
    def _initialize_states(self):
        """Initialize all state categories if they don't exist."""
        if 'model_state' not in st.session_state:
            st.session_state.model_state = ModelState()
        
        if 'ui_state' not in st.session_state:
            st.session_state.ui_state = UIState()
        
        if 'education_state' not in st.session_state:
            st.session_state.education_state = EducationState()
        
        if 'analysis_state' not in st.session_state:
            st.session_state.analysis_state = AnalysisState()
        

        self._sync_to_legacy_keys()
    
    def _sync_to_legacy_keys(self):
        """Sync organized state to legacy session state keys for backward compatibility."""
        model_state = st.session_state.model_state
        ui_state = st.session_state.ui_state
        education_state = st.session_state.education_state
        analysis_state = st.session_state.analysis_state
        
        st.session_state.model_loaded = model_state.model_loaded
        st.session_state.current_model_path = model_state.current_model_path
        st.session_state.loaded_model_beta = model_state.loaded_model_beta
        st.session_state.model_image_dims = model_state.model_image_dims
        st.session_state.model_num_channels = model_state.model_num_channels
        st.session_state.latent_values = model_state.latent_values
        st.session_state.fragment_latent_values = model_state.fragment_latent_values
        st.session_state.default_model_set = model_state.default_model_set
        
        # UI-related keys
        st.session_state.user_preferences = ui_state.user_preferences
        st.session_state.is_preview_generation = ui_state.is_preview_generation
        
        # Education-related keys
        st.session_state.learning_progress = education_state.learning_progress
        st.session_state.current_advanced_step = education_state.current_advanced_step
        if education_state.random_latent_point is not None:
            st.session_state.random_latent_point = education_state.random_latent_point
        
        # Analysis-related keys
        st.session_state.generation_history = analysis_state.generation_history
    
    def _sync_from_legacy_keys(self):
        """Sync legacy session state keys back to organized state."""
        # This handles cases where the legacy keys are modified directly
        if hasattr(st.session_state, 'model_loaded'):
            st.session_state.model_state.model_loaded = st.session_state.model_loaded
        if hasattr(st.session_state, 'current_model_path'):
            st.session_state.model_state.current_model_path = st.session_state.current_model_path
        if hasattr(st.session_state, 'loaded_model_beta'):
            st.session_state.model_state.loaded_model_beta = st.session_state.loaded_model_beta
        if hasattr(st.session_state, 'model_image_dims'):
            st.session_state.model_state.model_image_dims = st.session_state.model_image_dims
        if hasattr(st.session_state, 'model_num_channels'):
            st.session_state.model_state.model_num_channels = st.session_state.model_num_channels
        if hasattr(st.session_state, 'latent_values'):
            st.session_state.model_state.latent_values = st.session_state.latent_values
        if hasattr(st.session_state, 'fragment_latent_values'):
            st.session_state.model_state.fragment_latent_values = st.session_state.fragment_latent_values
        if hasattr(st.session_state, 'default_model_set'):
            st.session_state.model_state.default_model_set = st.session_state.default_model_set
        
        # UI state
        if hasattr(st.session_state, 'user_preferences'):
            st.session_state.ui_state.user_preferences = st.session_state.user_preferences
        if hasattr(st.session_state, 'is_preview_generation'):
            st.session_state.ui_state.is_preview_generation = st.session_state.is_preview_generation
        
        # Education state
        if hasattr(st.session_state, 'learning_progress'):
            st.session_state.education_state.learning_progress = st.session_state.learning_progress
        if hasattr(st.session_state, 'current_advanced_step'):
            st.session_state.education_state.current_advanced_step = st.session_state.current_advanced_step
        if hasattr(st.session_state, 'random_latent_point'):
            st.session_state.education_state.random_latent_point = st.session_state.random_latent_point
        
        # Analysis state
        if hasattr(st.session_state, 'generation_history'):
            st.session_state.analysis_state.generation_history = st.session_state.generation_history
    
    @property
    def model(self) -> ModelState:
        """Access model state."""
        return st.session_state.model_state
    
    @property
    def ui(self) -> UIState:
        """Access UI state."""
        return st.session_state.ui_state
    
    @property
    def education(self) -> EducationState:
        """Access education state."""
        return st.session_state.education_state
    
    @property
    def analysis(self) -> AnalysisState:
        """Access analysis state."""
        return st.session_state.analysis_state
    
    def update_model_state(self, **kwargs):
        """Update model state and sync to legacy keys."""
        for key, value in kwargs.items():
            if hasattr(st.session_state.model_state, key):
                setattr(st.session_state.model_state, key, value)
        self._sync_to_legacy_keys()
    
    def update_ui_state(self, **kwargs):
        """Update UI state and sync to legacy keys."""
        for key, value in kwargs.items():
            if hasattr(st.session_state.ui_state, key):
                setattr(st.session_state.ui_state, key, value)
        self._sync_to_legacy_keys()
    
    def update_education_state(self, **kwargs):
        """Update education state and sync to legacy keys."""
        for key, value in kwargs.items():
            if hasattr(st.session_state.education_state, key):
                setattr(st.session_state.education_state, key, value)
        self._sync_to_legacy_keys()
    
    def update_analysis_state(self, **kwargs):
        """Update analysis state and sync to legacy keys."""
        for key, value in kwargs.items():
            if hasattr(st.session_state.analysis_state, key):
                setattr(st.session_state.analysis_state, key, value)
        self._sync_to_legacy_keys()
    
    def reset_model_state(self):
        """Reset model state to defaults."""
        st.session_state.model_state = ModelState()
        self._sync_to_legacy_keys()
    
    def get_state_summary(self) -> Dict[str, Any]:
        """Get a summary of current state for debugging."""
        return {
            'model_loaded': self.model.model_loaded,
            'current_model_path': self.model.current_model_path,
            'latent_dim': len(self.model.latent_values),
            'learning_progress_total': sum(len(progress) for progress in self.education.learning_progress.values()),
            'generation_history_count': len(self.analysis.generation_history),
            'ui_preferences': self.ui.user_preferences
        }

def init_session_state():
    """Initialize session state for the ILVE application."""
    state_manager = SessionStateManager()
    return state_manager


def get_session_state_manager() -> SessionStateManager:
    """Get or create the session state manager."""
    if 'state_manager' not in st.session_state:
        st.session_state.state_manager = SessionStateManager()
    return st.session_state.state_manager