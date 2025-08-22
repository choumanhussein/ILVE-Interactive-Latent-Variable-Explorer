"""
Interactive Latent Space Explorer

This script creates an interactive exploration tool for trained VAE models.
Use sliders to navigate the continuous latent space in real-time!
"""

import sys
import os
sys.path.append('../src')

import torch
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Slider
import tkinter as tk
from tkinter import ttk, filedialog
from PIL import Image, ImageTk

from src.models.base_vae import VAE
from src.models.beta_vae import BetaVAE, ControlledBetaVAE


class LatentSpaceExplorer:
    """Interactive latent space exploration tool."""
    
    def __init__(self, model_path: str, device: str = 'cpu'):
        self.device = device
        self.model = self.load_model(model_path)
        self.latent_dim = self.model.latent_dim
        
        # Current latent code
        self.current_z = torch.zeros(1, self.latent_dim, device=device)
        
        # Setup UI
        self.setup_matplotlib_interface()
    
    def load_model(self, model_path: str):
        """Load trained VAE model."""
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Determine model type and create instance
        config = checkpoint.get('config', {})
        beta = checkpoint.get('beta', 1.0)
        
        model = BetaVAE(
            input_dim=config.get('model', {}).get('input_dim', 784),
            hidden_dims=config.get('model', {}).get('hidden_dims', [512, 256]),
            latent_dim=config.get('model', {}).get('latent_dim', 2),
            beta=beta
        )
        
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        model.eval()
        
        print(f"✓ Loaded model with {model.latent_dim}D latent space, β = {beta}")
        return model
    
    def setup_matplotlib_interface(self):
        """Create matplotlib-based interactive interface."""
        # Create figure
        self.fig, (self.ax_image, self.ax_latent) = plt.subplots(1, 2, figsize=(12, 6))
        
        # Setup image display
        self.ax_image.set_title('Generated Image')
        self.ax_image.axis('off')
        
        # Setup latent space display (for 2D)
        if self.latent_dim == 2:
            self.ax_latent.set_title('Latent Space Position')
            self.ax_latent.set_xlim(-3, 3)
            self.ax_latent.set_ylim(-3, 3)
            self.ax_latent.grid(True, alpha=0.3)
            self.ax_latent.set_xlabel('Latent Dimension 1')
            self.ax_latent.set_ylabel('Latent Dimension 2')
            
            # Current position marker
            self.position_marker, = self.ax_latent.plot(0, 0, 'ro', markersize=10)
        else:
            # For higher dimensions, show bar chart of current values
            self.ax_latent.set_title('Latent Code Values')
            self.bars = self.ax_latent.bar(range(self.latent_dim), 
                                          [0] * self.latent_dim, alpha=0.7)
            self.ax_latent.set_ylim(-3, 3)
            self.ax_latent.set_xlabel('Latent Dimension')
            self.ax_latent.set_ylabel('Value')
            self.ax_latent.grid(True, alpha=0.3)
        
        # Create sliders
        self.create_sliders()
        
        # Initial generation
        self.update_display()
        
        plt.tight_layout()
        plt.show()
    
    def create_sliders(self):
        """Create sliders for each latent dimension."""
        self.sliders = []
        
        # Calculate slider positions
        bottom_margin = 0.15
        slider_height = 0.02
        slider_spacing = 0.025
        
        for i in range(min(self.latent_dim, 10)):  # Limit to 10 sliders for UI
            # Create slider axis
            slider_bottom = bottom_margin + i * slider_spacing
            ax_slider = plt.axes([0.1, slider_bottom, 0.3, slider_height])
            
            # Create slider
            slider = Slider(
                ax_slider, f'z{i}', -3.0, 3.0, valinit=0.0, 
                valfmt='%.2f'
            )
            
            # Connect callback
            slider.on_changed(lambda val, dim=i: self.update_latent_dimension(dim, val))
            self.sliders.append(slider)
    
    def update_latent_dimension(self, dim: int, value: float):
        """Update a specific latent dimension."""
        self.current_z[0, dim] = value
        self.update_display()
    
    def update_display(self):
        """Update the display with current latent code."""
        with torch.no_grad():
            # Generate image
            generated = self.model.decode(self.current_z)
            image = generated.view(28, 28).cpu().numpy()
            
            # Update image display
            self.ax_image.clear()
            self.ax_image.imshow(image, cmap='gray')
            self.ax_image.set_title('Generated Image')
            self.ax_image.axis('off')
            
            # Update latent space display
            current_values = self.current_z[0].cpu().numpy()
            
            if self.latent_dim == 2:
                # Update position marker
                self.position_marker.set_data([current_values[0]], [current_values[1]])
            else:
                # Update bar chart
                for bar, value in zip(self.bars, current_values):
                    bar.set_height(value)
        
        plt.draw()
    
    def random_sample(self):
        """Generate a random sample."""
        self.current_z = torch.randn(1, self.latent_dim, device=self.device) * 2
        
        # Update sliders
        for i, slider in enumerate(self.sliders):
            if i < self.latent_dim:
                slider.set_val(self.current_z[0, i].item())
        
        self.update_display()
    
    def interpolate_to_random(self, steps: int = 20):
        """Animate interpolation to a random point."""
        target_z = torch.randn(1, self.latent_dim, device=self.device) * 2
        
        for step in range(steps + 1):
            alpha = step / steps
            interpolated_z = (1 - alpha) * self.current_z + alpha * target_z
            
            self.current_z = interpolated_z
            
            # Update sliders
            for i, slider in enumerate(self.sliders):
                if i < self.latent_dim:
                    slider.set_val(self.current_z[0, i].item())
            
            self.update_display()
            plt.pause(0.1)


def create_simple_explorer():
    """Create a simple command-line latent space explorer."""
    print("🎮 Simple Latent Space Explorer")
    print("=" * 40)
    
    # Ask user to select model
    print("Available models:")
    checkpoint_dir = "experiments/checkpoints"
    
    if os.path.exists(checkpoint_dir):
        model_files = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pth')]
        
        if model_files:
            print("Available models:")
            for i, model_file in enumerate(model_files):
                print(f"{i+1}. {model_file}")
            
            try:
                choice = int(input(f"Select model (1-{len(model_files)}): ")) - 1
                model_path = os.path.join(checkpoint_dir, model_files[choice])
            except:
                print("Invalid choice, using first model")
                model_path = os.path.join(checkpoint_dir, model_files[0])
        else:
            print("No trained models found!")
            print("Train a model first using: python train_basic_vae.py")
            return
    else:
        print("No checkpoint directory found!")
        return
    
    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(model_path, map_location=device)
    
    config = checkpoint.get('config', {})
    beta = checkpoint.get('beta', 1.0)
    
    model = BetaVAE(
        input_dim=config.get('model', {}).get('input_dim', 784),
        latent_dim=config.get('model', {}).get('latent_dim', 2),
        beta=beta
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    print(f"✓ Loaded model: {model_files[choice]}")
    print(f"  Latent dimensions: {model.latent_dim}")
    print(f"  β value: {beta}")
    
    # Interactive exploration
    current_z = torch.zeros(1, model.latent_dim, device=device)
    
    while True:
        print(f"\nCurrent latent code: {current_z[0].cpu().numpy()}")
        
        # Generate and display image
        with torch.no_grad():
            generated = model.decode(current_z)
            image = generated.view(28, 28).cpu().numpy()
        
        # Simple visualization
        plt.figure(figsize=(4, 4))
        plt.imshow(image, cmap='gray')
        plt.title('Generated Image')
        plt.axis('off')
        plt.show()
        
        print("\nOptions:")
        print("1. Modify dimension")
        print("2. Random sample")
        print("3. Zero all dimensions")
        print("4. Quit")
        
        try:
            choice = input("Enter choice (1-4): ").strip()
            
            if choice == '1':
                dim = int(input(f"Dimension to modify (0-{model.latent_dim-1}): "))
                value = float(input("New value (-3 to 3): "))
                current_z[0, dim] = value
                
            elif choice == '2':
                current_z = torch.randn(1, model.latent_dim, device=device) * 2
                print("Generated random sample!")
                
            elif choice == '3':
                current_z = torch.zeros(1, model.latent_dim, device=device)
                print("Reset to zero!")
                
            elif choice == '4':
                break
                
        except (ValueError, IndexError):
            print("Invalid input!")
    
    print("👋 Explorer closed!")


def main():
    """Main function to launch explorer."""
    print("🚀 Latent Space Explorer")
    print("=" * 30)
    
    # Check for model files
    checkpoint_dir = "experiments/checkpoints"
    if not os.path.exists(checkpoint_dir):
        print("❌ No trained models found!")
        print("Train a model first using:")
        print("  python train_basic_vae.py --epochs 10 --no-wandb")
        return
    
    model_files = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pth')]
    if not model_files:
        print("❌ No model files found in checkpoints directory!")
        return
    
    print("Choose explorer mode:")
    print("1. Interactive (matplotlib with sliders)")
    print("2. Simple (command-line)")
    
    try:
        mode = input("Mode (1 or 2): ").strip()
        
        if mode == '1':
            # Interactive mode
            print("Select model file:")
            for i, model_file in enumerate(model_files):
                print(f"{i+1}. {model_file}")
            
            choice = int(input(f"Select (1-{len(model_files)}): ")) - 1
            model_path = os.path.join(checkpoint_dir, model_files[choice])
            
            # Launch interactive explorer
            explorer = LatentSpaceExplorer(model_path)
            
            print("\n🎮 Controls:")
            print("• Use sliders to navigate latent space")
            print("• Press 'r' for random sample")
            print("• Close window to exit")
            
        elif mode == '2':
            # Simple mode
            create_simple_explorer()
            
        else:
            print("Invalid mode!")
            
    except (ValueError, IndexError, KeyboardInterrupt):
        print("\n👋 Explorer closed!")


if __name__ == "__main__":
    main()