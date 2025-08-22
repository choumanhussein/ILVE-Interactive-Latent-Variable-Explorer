"""
Core model loading functionality for ILVE framework.
Extracts intelligent model loading capabilities from the main app.
"""

import os
import torch
import numpy as np
import streamlit as st
from datetime import datetime
from typing import Dict, Any, Optional, Tuple, Union
import sys
from pathlib import Path


def setup_model_imports():
    """Setup import paths for existing model structure."""
    current_dir = Path(__file__).parent.parent.parent
    src_path = current_dir / 'src'
    if src_path.exists():
        sys.path.insert(0, str(src_path))
    
    parent_dir = current_dir.parent
    sys.path.insert(0, str(parent_dir))
    sys.path.insert(0, str(parent_dir / 'src'))

class ModelLoadingError(Exception):
    """Custom exception for model loading errors."""
    pass

class ModelLoader:
    """
    Intelligent model loader that automatically infers model parameters.
    Extracted from the original app.py intelligent model loading functionality.
    """
    
    def __init__(self):
        setup_model_imports()
        self._import_models()
    
    def _import_models(self):
        """Import model classes with error handling."""
        try:
            from models.base_vae import VAE, VAEConfig
            from models.beta_vae import BetaVAE, ControlledBetaVAE
            self.VAE = VAE
            self.BetaVAE = BetaVAE
            self.ControlledBetaVAE = ControlledBetaVAE
            self.VAEConfig = VAEConfig
            
            
            try:
                from models.enhanced_vae import EnhancedVAE, EnhancedBetaVAE, detect_architecture_from_state_dict
                self.EnhancedVAE = EnhancedVAE
                self.EnhancedBetaVAE = EnhancedBetaVAE
                self.detect_architecture = detect_architecture_from_state_dict
                self.use_enhanced = True
                st.info("✅ Enhanced VAE architecture available")
            except ImportError:
                self.use_enhanced = False
                # st.info("ℹ️ Using standard VAE architecture")
                
        except ImportError as e:
            st.error(f"Failed to import models: {e}")
            st.error("Please ensure the models directory is accessible and contains the required files.")
            raise ModelLoadingError(f"Model import failed: {e}")
    
    def _safe_get_config_value(self, config, key, default=None):
        """Safely get a value from config, whether it's a dict or dataclass."""
        if isinstance(config, dict):
            return config.get(key, default)
        else:

            return getattr(config, key, default)
    
    @st.cache_data
    def get_available_models(_self) -> list:
        """Get list of available trained models with enhanced metadata."""
        possible_dirs = [
            "experiments/checkpoints",
            "../experiments/checkpoints", 
            "checkpoints",
            "../checkpoints",
            ".",
            ".."
        ]
        
        model_info = []
        
        for checkpoint_dir in possible_dirs:
            if os.path.exists(checkpoint_dir):
                files = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pth')]
                for f in files:
                    full_path = os.path.join(checkpoint_dir, f)
                    

                    model_type = "VAE"
                    beta_val = 1.0
                    
                    if "beta" in f.lower():
                        model_type = "β-VAE"
                        beta_val = _self._extract_beta_from_filename(f)
                    
                    
                    stat = os.stat(full_path)
                    size_mb = stat.st_size / (1024 * 1024)
                    modified = datetime.fromtimestamp(stat.st_mtime)
                    
                    model_info.append({
                        'name': f,
                        'path': full_path,
                        'type': model_type,
                        'beta': beta_val,
                        'size_mb': size_mb,
                        'modified': modified,
                        'display_name': f"{model_type} - {f}"
                    })
        

        model_info.sort(key=lambda x: x['modified'], reverse=True)
        return model_info
    
    def _extract_beta_from_filename(self, filename: str) -> float:
        """Extract beta value from filename with robust parsing."""
        try:
            filename_lower = filename.lower()
            if filename_lower.startswith('beta_vae_'):
                
                beta_part = filename[9:]  
                if beta_part.lower().endswith('.pth'):
                    beta_part = beta_part[:-4]  
                return float(beta_part)
            elif '_beta_' in filename_lower:

                beta_str = filename.split('_beta_')[-1].split('.')[0].replace('_', '.')
                return float(beta_str)
            else:
               
                import re
                beta_match = re.search(r'beta[_-]?(\d+\.?\d*)', filename_lower)
                if beta_match:
                    return float(beta_match.group(1))
        except (ValueError, IndexError):
            pass
        return 1.0 
    
    def _try_load_with_different_architectures(self, model_state_dict, input_dim, hidden_dims, latent_dim, beta, model_path):
        """Try loading with different VAE architectures until one works."""
        

        configs_to_try = [
            
            {
                'name': 'Architecture with BatchNorm (detected)',
                'use_batchnorm': True,
                'hidden_dims': hidden_dims
            },

            {
                'name': 'Standard architecture',
                'use_batchnorm': False, 
                'hidden_dims': hidden_dims
            },
            
            {
                'name': 'Alternative hidden dims [400, 200]',
                'use_batchnorm': True,
                'hidden_dims': [400, 200]
            },

            {
                'name': 'Simple 2-layer [512, 256]',
                'use_batchnorm': False,
                'hidden_dims': [512, 256]
            }
        ]
        
        for config in configs_to_try:
            try:
                st.info(f"🔄 Trying: {config['name']}")
                
                
                model = self._create_custom_vae(
                    input_dim=input_dim,
                    hidden_dims=config['hidden_dims'],
                    latent_dim=latent_dim,
                    beta=beta,
                    use_batchnorm=config['use_batchnorm'],
                    is_beta_vae="beta" in model_path.lower() or beta != 1.0
                )
                

                try:
                    model.load_state_dict(model_state_dict, strict=True)
                    st.success(f"✅ Successfully loaded with: {config['name']}")
                    return model
                except RuntimeError as e:
                    if len(str(e)) < 200:  
                        st.write(f"  ❌ Failed: {str(e)}")
                    else:
                        st.write(f"  ❌ Failed: Architecture mismatch")
                    continue
                    
            except Exception as e:
                st.write(f"  ❌ Model creation failed: {str(e)[:100]}")
                continue
        
        
        st.warning("🔄 All exact matches failed. Trying flexible loading...")
        try:
            model = self._create_custom_vae(
                input_dim=input_dim,
                hidden_dims=hidden_dims,
                latent_dim=latent_dim,
                beta=beta,
                use_batchnorm=True,  
                is_beta_vae="beta" in model_path.lower() or beta != 1.0
            )
            
            missing_keys, unexpected_keys = model.load_state_dict(model_state_dict, strict=False)
            st.warning(f"⚠️ Flexible loading: {len(missing_keys)} missing, {len(unexpected_keys)} unexpected keys")
            return model
            
        except Exception as e:
            st.error(f"Even flexible loading failed: {e}")
            return None
    
    def _create_custom_vae(self, input_dim, hidden_dims, latent_dim, beta, use_batchnorm, is_beta_vae):
        """Create a custom VAE with specified architecture."""
        
        class CustomEncoder(torch.nn.Module):
            def __init__(self, input_dim, hidden_dims, latent_dim, use_batchnorm):
                super().__init__()
                
                layers = []
                prev_dim = input_dim
                
                for hidden_dim in hidden_dims:
                    layers.append(torch.nn.Linear(prev_dim, hidden_dim))
                    if use_batchnorm:
                        layers.append(torch.nn.BatchNorm1d(hidden_dim))
                    layers.append(torch.nn.ReLU())
                    layers.append(torch.nn.Dropout(0.2))
                    prev_dim = hidden_dim
                
                self.encoder = torch.nn.Sequential(*layers)
                self.fc_mu = torch.nn.Linear(prev_dim, latent_dim)
                self.fc_logvar = torch.nn.Linear(prev_dim, latent_dim)
            
            def forward(self, x):
                h = self.encoder(x)
                return self.fc_mu(h), self.fc_logvar(h)
        
        class CustomDecoder(torch.nn.Module):
            def __init__(self, latent_dim, hidden_dims, output_dim, use_batchnorm):
                super().__init__()
                
                layers = []
                prev_dim = latent_dim
                
                for hidden_dim in reversed(hidden_dims):
                    layers.append(torch.nn.Linear(prev_dim, hidden_dim))
                    if use_batchnorm:
                        layers.append(torch.nn.BatchNorm1d(hidden_dim))
                    layers.append(torch.nn.ReLU())
                    layers.append(torch.nn.Dropout(0.2))
                    prev_dim = hidden_dim
                
                layers.append(torch.nn.Linear(prev_dim, output_dim))
                layers.append(torch.nn.Sigmoid())
                self.decoder = torch.nn.Sequential(*layers)
            
            def forward(self, z):
                return self.decoder(z)
        
        class CustomVAE(torch.nn.Module):
            def __init__(self, input_dim, hidden_dims, latent_dim, beta, use_batchnorm):
                super().__init__()
                self.input_dim = input_dim
                self.latent_dim = latent_dim
                self.beta = beta
                
                self.encoder = CustomEncoder(input_dim, hidden_dims, latent_dim, use_batchnorm)
                self.decoder = CustomDecoder(latent_dim, hidden_dims, input_dim, use_batchnorm)
            
            def reparameterize(self, mu, logvar):
                if self.training:
                    std = torch.exp(0.5 * logvar)
                    eps = torch.randn_like(std)
                    return mu + eps * std
                else:
                    return mu
            
            def forward(self, x):
                mu, logvar = self.encoder(x)
                z = self.reparameterize(mu, logvar)
                reconstruction = self.decoder(z)
                return reconstruction, mu, logvar, z
            
            def encode(self, x):
                return self.encoder(x)
            
            def decode(self, z):
                return self.decoder(z)
            
            def eval(self):
                super().eval()
                return self
        
        return CustomVAE(input_dim, hidden_dims, latent_dim, beta, use_batchnorm)
    
    @st.cache_resource
    def load_model(_self, model_path: str, device: str = 'cpu') -> Tuple[Optional[Any], float, Dict[str, Any]]:
        """
        Enhanced model loading with detailed progress and error handling.
        Returns: (model, beta, config)
        """
        progress_container = st.container()
        
        with progress_container:
            st.markdown('<div class="progress-container">', unsafe_allow_html=True)
            st.markdown("### 🔄 Loading Your AI Model...")
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            model = None
            beta = 1.0  # Default beta value
            config = {}  # Default config
            latent_dim = 2  # Default latent dimension
            input_dim = 784  # Default for MNIST 28x28
            image_dims = (28, 28)  # Default output image dimensions
            num_channels = 1  # Default output image channels

            try:
                status_text.markdown("**Step 1/4:** Checking model file integrity...")
                progress_bar.progress(25)
                
                if not os.path.exists(model_path):
                    st.markdown('<div class="status-error">❌ Model file not found! Please ensure it\'s in the "checkpoints" folder.</div>', unsafe_allow_html=True)
                    return None, None, None
                
                
                filename = os.path.basename(model_path)
                filename_beta = _self._extract_beta_from_filename(filename)
                
                status_text.markdown("**Step 2/4:** Reading model data from file...")
                progress_bar.progress(50)


                try:
                    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
                    st.info("✅ Successfully loaded checkpoint")
                except Exception as load_error:
                    st.error(f"Failed to load checkpoint: {load_error}")
                    try:
                        st.info("Attempting alternative loading method...")
                        checkpoint = torch.load(model_path, map_location=device, weights_only=True)
                        st.warning("Loaded weights only - some metadata may be missing")
                    except Exception as alt_error:
                        st.error(f"Alternative loading also failed: {alt_error}")
                        return None, None, None
                
                status_text.markdown("**Step 3/4:** Setting up the AI's 'brain' (model architecture)...")
                progress_bar.progress(75)
                
                model_state_dict = None
                if isinstance(checkpoint, dict):
                    config = checkpoint.get('config', {})
                    beta = checkpoint.get('beta', filename_beta)
                    model_state_dict = checkpoint.get('model_state_dict', checkpoint.get('state_dict', checkpoint))
                    

                    if isinstance(config, dict):
                        model_config = config.get('model', config)  
                    else:

                        model_config = config
                    
                    
                    latent_dim = _self._safe_get_config_value(model_config, 'latent_dim', 2)
                    input_dim = _self._safe_get_config_value(model_config, 'input_dim', 784)
                    hidden_dims = _self._safe_get_config_value(model_config, 'hidden_dims', [512, 256])
                    

                    if input_dim:
                        image_dims, num_channels = _self._infer_image_dimensions(input_dim)
                    
                    
                    if latent_dim == 2 and model_state_dict:
                        inferred_latent_dim = _self._infer_latent_dim_from_state_dict(model_state_dict)
                        if inferred_latent_dim != 2:
                            latent_dim = inferred_latent_dim
                            st.info(f"💡 Inferred latent_dim: **{latent_dim}** from model weights.")


                    model = _self._try_load_with_different_architectures(
                        model_state_dict, input_dim, hidden_dims, latent_dim, beta, model_path
                    )
                    
                    if model is None:
                        st.error("Failed to load model with any architecture configuration")
                        return None, None, None
                        
                else:  
                    st.warning("Checkpoint does not contain explicit config. Attempting to infer parameters.")
                    beta = filename_beta
                    

                    latent_dim_inferred = _self._infer_latent_dim_from_state_dict(checkpoint)
                    input_dim_inferred = _self._infer_input_dim_from_state_dict(checkpoint)
                    
                    st.info(f"💡 Inferred latent_dim: **{latent_dim_inferred}**, input_dim: **{input_dim_inferred}**, β: **{beta}**")
                    
                    
                    try:
                        if "beta" in model_path.lower() or beta != 1.0:
                            model = _self.BetaVAE(input_dim=input_dim_inferred, latent_dim=latent_dim_inferred, beta=beta)
                            st.info(f"Loading as BetaVAE with β={beta}")
                        else:
                            model = _self.VAE(input_dim=input_dim_inferred, latent_dim=latent_dim_inferred)
                            st.info("Loading as standard VAE")
                        

                        try:
                            model.load_state_dict(checkpoint, strict=True)
                            st.success("✅ Model weights loaded successfully (exact match)")
                        except RuntimeError:
                            st.info("🔄 Attempting flexible loading (architecture mismatch detected)")
                            missing_keys, unexpected_keys = model.load_state_dict(checkpoint, strict=False)
                            
                            if missing_keys or unexpected_keys:
                                st.warning(f"⚠️ Partial loading: {len(missing_keys)} missing, {len(unexpected_keys)} unexpected keys")
                            
                            st.success("✅ Model weights loaded with partial compatibility")
                            
                    except Exception as direct_load_e:
                        st.error(f"Failed to load raw state_dict: {direct_load_e}")
                        return None, None, None
                
                status_text.markdown("**Step 4/4:** Model is ready! Let's explore its creativity... 🎉")
                progress_bar.progress(100)
                
                model.eval() 
                

                final_config = {
                    'latent_dim': model.latent_dim,
                    'beta': beta,
                    'image_dims': image_dims,
                    'num_channels': num_channels,
                    'param_count': sum(p.numel() for p in model.parameters())
                }
                
                st.markdown('<div class="status-success">✅ Model loaded successfully!</div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
                
                return model, beta, final_config
                
            except Exception as e:
                st.markdown('<div class="status-error">❌ Failed to load model.</div>', unsafe_allow_html=True)
                st.error(f"Error details: {str(e)}")
                return None, None, None
    
    def _infer_image_dimensions(self, input_dim: int) -> Tuple[Tuple[int, int], int]:
        """Infer image dimensions from input_dim."""
        if input_dim % (28*28) == 0: 
            num_channels = input_dim // (28*28)
            return (28, 28), num_channels
        else:  
            side = int(np.sqrt(input_dim))
            if side * side == input_dim:
                return (side, side), 1
            else:
                st.warning(f"Could not reliably infer image dimensions from input_dim {input_dim}. Defaulting to 28x28x1.")
                return (28, 28), 1
    
    def _infer_latent_dim_from_state_dict(self, state_dict: Dict[str, torch.Tensor]) -> int:
        """Infer latent dimension from model state dict."""
        for k, v in state_dict.items():
            if 'fc_mu.weight' in k and len(v.shape) > 0:
                return v.shape[0]
            elif 'encoder.fc_out.weight' in k and len(v.shape) > 1:
                return v.shape[0] // 2  
        return 2  
    
    def _infer_input_dim_from_state_dict(self, state_dict: Dict[str, torch.Tensor]) -> int:
        """Infer input dimension from model state dict."""
        for k, v in state_dict.items():
            if 'decoder_lin.0.weight' in k and len(v.shape) > 1:
                return v.shape[1]
            elif 'fc_decode.weight' in k and len(v.shape) > 1:
                return v.shape[0]
        return 784  