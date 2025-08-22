"""
Configuration management for ILVE framework.
Supports YAML-based experimental configurations for research reproducibility.
"""

import yaml
import os
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field

@dataclass
class ModelConfig:
    """Configuration for a single model."""
    path: str
    beta: float = 1.0
    type: str = "VAE"
    latent_dim: Optional[int] = None
    hidden_dims: List[int] = field(default_factory=lambda: [512, 256])

@dataclass
class AnalysisConfig:
    """Configuration for analysis parameters."""
    dimensions_to_show: int = 8
    traversal_range: float = 2.5
    grid_resolution: int = 8
    num_random_samples: int = 8
    analysis_detail_steps: int = 11
    
@dataclass
class UIConfig:
    """Configuration for UI settings."""
    theme: str = "modern"
    show_advanced_by_default: bool = False
    auto_generate: bool = True
    max_latent_dims_display: int = 8

@dataclass
class ExperimentConfig:
    """Configuration for an experimental setup."""
    name: str
    description: str = ""
    models: List[ModelConfig] = field(default_factory=list)
    analysis: AnalysisConfig = field(default_factory=AnalysisConfig)
    ui: UIConfig = field(default_factory=UIConfig)

class ConfigManager:
    """Manages application and experiment configurations."""
    
    def __init__(self, config_dir: str = "config"):
        self.config_dir = Path(config_dir)
        self.experiments_dir = self.config_dir / "experiments"
        
        # Create directories if they don't exist
        self.config_dir.mkdir(exist_ok=True)
        self.experiments_dir.mkdir(exist_ok=True)
        
        # Default configuration
        self.default_config = ExperimentConfig(
            name="Default Configuration",
            description="Standard ILVE configuration for general use"
        )
    
    def load_experiment_config(self, config_name: str) -> Optional[ExperimentConfig]:
        """Load an experiment configuration from YAML file."""
        config_path = self.experiments_dir / f"{config_name}.yaml"
        
        if not config_path.exists():
            return None
        
        try:
            with open(config_path, 'r') as f:
                data = yaml.safe_load(f)
            
            return self._parse_experiment_config(data)
        except Exception as e:
            print(f"Error loading config {config_name}: {e}")
            return None
    
    def save_experiment_config(self, config: ExperimentConfig, config_name: str) -> bool:
        """Save an experiment configuration to YAML file."""
        config_path = self.experiments_dir / f"{config_name}.yaml"
        
        try:
            data = self._serialize_experiment_config(config)
            with open(config_path, 'w') as f:
                yaml.dump(data, f, default_flow_style=False, indent=2)
            return True
        except Exception as e:
            print(f"Error saving config {config_name}: {e}")
            return False
    
    def list_available_configs(self) -> List[str]:
        """List all available experiment configurations."""
        configs = []
        if self.experiments_dir.exists():
            for config_file in self.experiments_dir.glob("*.yaml"):
                configs.append(config_file.stem)
        return sorted(configs)
    
    def create_default_configs(self):
        """Create default experiment configurations."""
        # Basic VAE configuration
        basic_vae = ExperimentConfig(
            name="Basic VAE",
            description="Standard VAE with β=1.0 for baseline comparison",
            models=[
                ModelConfig(
                    path="checkpoints/vae_latent_dim_2.pth",
                    beta=1.0,
                    type="VAE",
                    latent_dim=2
                )
            ],
            analysis=AnalysisConfig(
                dimensions_to_show=2,
                traversal_range=2.0,
                grid_resolution=8
            )
        )
        
        # Beta VAE comparison configuration
        beta_comparison = ExperimentConfig(
            name="Beta VAE Comparison",
            description="Comparison of different β values for disentanglement analysis",
            models=[
                ModelConfig(path="checkpoints/beta_vae_1.0.pth", beta=1.0, type="VAE"),
                ModelConfig(path="checkpoints/beta_vae_2.0.pth", beta=2.0, type="BetaVAE"),
                ModelConfig(path="checkpoints/beta_vae_4.0.pth", beta=4.0, type="BetaVAE"),
                ModelConfig(path="checkpoints/beta_vae_8.0.pth", beta=8.0, type="BetaVAE")
            ],
            analysis=AnalysisConfig(
                dimensions_to_show=8,
                traversal_range=2.5,
                grid_resolution=10
            )
        )
        
        # High-dimensional configuration
        high_dimensional = ExperimentConfig(
            name="High Dimensional Analysis",
            description="Exploration of high-dimensional latent spaces (10D+)",
            models=[
                ModelConfig(
                    path="checkpoints/beta_vae_latent_dim_20.pth",
                    beta=4.0,
                    type="BetaVAE",
                    latent_dim=20
                )
            ],
            analysis=AnalysisConfig(
                dimensions_to_show=8,  # Show subset by default
                traversal_range=3.0,
                grid_resolution=6,
                num_random_samples=12
            ),
            ui=UIConfig(
                show_advanced_by_default=True,  # Show more controls for research
                max_latent_dims_display=20
            )
        )
        
        # Save default configurations
        self.save_experiment_config(basic_vae, "basic_vae")
        self.save_experiment_config(beta_comparison, "beta_vae_comparison")
        self.save_experiment_config(high_dimensional, "high_dimensional")
    
    def _parse_experiment_config(self, data: Dict[str, Any]) -> ExperimentConfig:
        """Parse experiment configuration from dictionary."""
        experiment_data = data.get('experiment', data)
        
        # Parse models
        models = []
        for model_data in experiment_data.get('models', []):
            models.append(ModelConfig(
                path=model_data['path'],
                beta=model_data.get('beta', 1.0),
                type=model_data.get('type', 'VAE'),
                latent_dim=model_data.get('latent_dim'),
                hidden_dims=model_data.get('hidden_dims', [512, 256])
            ))
        
        # Parse analysis config
        analysis_data = experiment_data.get('analysis', {})
        analysis = AnalysisConfig(
            dimensions_to_show=analysis_data.get('dimensions_to_show', 8),
            traversal_range=analysis_data.get('traversal_range', 2.5),
            grid_resolution=analysis_data.get('grid_resolution', 8),
            num_random_samples=analysis_data.get('num_random_samples', 8),
            analysis_detail_steps=analysis_data.get('analysis_detail_steps', 11)
        )
        
        # Parse UI config
        ui_data = experiment_data.get('ui', {})
        ui = UIConfig(
            theme=ui_data.get('theme', 'modern'),
            show_advanced_by_default=ui_data.get('show_advanced_by_default', False),
            auto_generate=ui_data.get('auto_generate', True),
            max_latent_dims_display=ui_data.get('max_latent_dims_display', 8)
        )
        
        return ExperimentConfig(
            name=experiment_data.get('name', 'Unnamed Experiment'),
            description=experiment_data.get('description', ''),
            models=models,
            analysis=analysis,
            ui=ui
        )
    
    def _serialize_experiment_config(self, config: ExperimentConfig) -> Dict[str, Any]:
        """Serialize experiment configuration to dictionary."""
        return {
            'experiment': {
                'name': config.name,
                'description': config.description,
                'models': [
                    {
                        'path': model.path,
                        'beta': model.beta,
                        'type': model.type,
                        'latent_dim': model.latent_dim,
                        'hidden_dims': model.hidden_dims
                    }
                    for model in config.models
                ],
                'analysis': {
                    'dimensions_to_show': config.analysis.dimensions_to_show,
                    'traversal_range': config.analysis.traversal_range,
                    'grid_resolution': config.analysis.grid_resolution,
                    'num_random_samples': config.analysis.num_random_samples,
                    'analysis_detail_steps': config.analysis.analysis_detail_steps
                },
                'ui': {
                    'theme': config.ui.theme,
                    'show_advanced_by_default': config.ui.show_advanced_by_default,
                    'auto_generate': config.ui.auto_generate,
                    'max_latent_dims_display': config.ui.max_latent_dims_display
                }
            }
        }


config_manager = ConfigManager()

def get_config_manager() -> ConfigManager:
    """Get the global configuration manager."""
    return config_manager

def load_config(config_name: str) -> Optional[ExperimentConfig]:
    """Load a configuration by name."""
    return config_manager.load_experiment_config(config_name)

def get_default_config() -> ExperimentConfig:
    """Get the default configuration."""
    return config_manager.default_config