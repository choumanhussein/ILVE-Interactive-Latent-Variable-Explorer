"""
Visualization utilities for VAE training and evaluation.

This module provides functions for visualizing:
- Model reconstructions vs originals
- Generated samples from latent space
- Latent space distributions and organization
- Training progress and loss curves
- Interpolations in latent space
"""

import torch
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from typing import Optional, Tuple, List
import matplotlib.gridspec as gridspec


def plot_reconstructions(
    original: torch.Tensor, 
    reconstructed: torch.Tensor, 
    num_samples: int = 8,
    title: str = "Original vs Reconstructed"
) -> plt.Figure:
    """
    Plot original images alongside their reconstructions.
    
    Args:
        original: Original images [batch_size, 784]
        reconstructed: Reconstructed images [batch_size, 784]
        num_samples: Number of samples to display
        title: Plot title
    
    Returns:
        matplotlib Figure object
    """
    num_samples = min(num_samples, original.size(0))
    
    # Reshape to 28x28 for visualization
    original_2d = original[:num_samples].view(-1, 28, 28)
    reconstructed_2d = reconstructed[:num_samples].view(-1, 28, 28)
    
    fig, axes = plt.subplots(2, num_samples, figsize=(num_samples * 2, 4))
    
    for i in range(num_samples):
        # Original images
        axes[0, i].imshow(original_2d[i].detach().numpy(), cmap='gray')
        axes[0, i].set_title('Original')
        axes[0, i].axis('off')
        
        # Reconstructed images
        axes[1, i].imshow(reconstructed_2d[i].detach().numpy(), cmap='gray')
        axes[1, i].set_title('Reconstructed')
        axes[1, i].axis('off')
    
    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    
    return fig


def plot_generated_samples(
    generated: torch.Tensor,
    num_samples: int = 16,
    title: str = "Generated Samples"
) -> plt.Figure:
    """
    Plot samples generated from the latent space.
    
    Args:
        generated: Generated images [batch_size, 784]
        num_samples: Number of samples to display
        title: Plot title
    
    Returns:
        matplotlib Figure object
    """
    num_samples = min(num_samples, generated.size(0))
    grid_size = int(np.sqrt(num_samples))
    
    # Reshape to 28x28 for visualization
    generated_2d = generated[:num_samples].view(-1, 28, 28)
    
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(grid_size * 2, grid_size * 2))
    
    for i in range(num_samples):
        row, col = i // grid_size, i % grid_size
        axes[row, col].imshow(generated_2d[i].detach().numpy(), cmap='gray')
        axes[row, col].axis('off')
    
    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    
    return fig


def plot_latent_space(
    latent_codes: torch.Tensor,
    labels: torch.Tensor,
    title: str = "Latent Space Distribution",
    figsize: Tuple[int, int] = (10, 8)
) -> plt.Figure:
    """
    Plot 2D latent space distribution colored by digit labels.
    
    Args:
        latent_codes: Latent codes [batch_size, 2]
        labels: Corresponding labels [batch_size]
        title: Plot title
        figsize: Figure size
    
    Returns:
        matplotlib Figure object
    """
    if latent_codes.size(1) != 2:
        raise ValueError("This function only supports 2D latent spaces")
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Convert to numpy
    z = latent_codes.detach().numpy()
    labels_np = labels.detach().numpy()
    
    # Create scatter plot with different colors for each digit
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    
    for digit in range(10):
        mask = labels_np == digit
        if mask.any():
            ax.scatter(z[mask, 0], z[mask, 1], 
                      c=[colors[digit]], label=f'Digit {digit}', 
                      alpha=0.7, s=50)
    
    ax.set_xlabel('Latent Dimension 1')
    ax.set_ylabel('Latent Dimension 2')
    ax.set_title(title)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    return fig


def plot_latent_space_evolution(
    latent_codes_list: List[torch.Tensor],
    labels_list: List[torch.Tensor],
    epochs: List[int],
    title: str = "Latent Space Evolution"
) -> plt.Figure:
    """
    Plot evolution of latent space over training epochs.
    
    Args:
        latent_codes_list: List of latent codes for different epochs
        labels_list: List of corresponding labels
        epochs: List of epoch numbers
        title: Plot title
    
    Returns:
        matplotlib Figure object
    """
    num_epochs = len(latent_codes_list)
    cols = min(4, num_epochs)
    rows = (num_epochs + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3))
    if num_epochs == 1:
        axes = [axes]
    elif rows == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    
    for i, (latent_codes, labels, epoch) in enumerate(zip(latent_codes_list, labels_list, epochs)):
        ax = axes[i]
        
        z = latent_codes.detach().numpy()
        labels_np = labels.detach().numpy()
        
        for digit in range(10):
            mask = labels_np == digit
            if mask.any():
                ax.scatter(z[mask, 0], z[mask, 1], 
                          c=[colors[digit]], alpha=0.7, s=30)
        
        ax.set_title(f'Epoch {epoch}')
        ax.set_xlabel('Latent Dim 1')
        ax.set_ylabel('Latent Dim 2')
        ax.grid(True, alpha=0.3)
    
    # Hide unused subplots
    for i in range(num_epochs, len(axes)):
        axes[i].set_visible(False)
    
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    
    return fig


def plot_interpolation(
    model,
    start_image: torch.Tensor,
    end_image: torch.Tensor,
    num_steps: int = 10,
    title: str = "Latent Space Interpolation"
) -> plt.Figure:
    """
    Plot interpolation between two images in latent space.
    
    Args:
        model: Trained VAE model
        start_image: Starting image [1, 784]
        end_image: Ending image [1, 784]
        num_steps: Number of interpolation steps
        title: Plot title
    
    Returns:
        matplotlib Figure object
    """
    model.eval()
    
    with torch.no_grad():
        # Encode both images
        mu_start, _ = model.encode(start_image)
        mu_end, _ = model.encode(end_image)
        
        # Linear interpolation in latent space
        alphas = torch.linspace(0, 1, num_steps)
        interpolated_images = []
        
        for alpha in alphas:
            z_interp = (1 - alpha) * mu_start + alpha * mu_end
            reconstruction = model.decode(z_interp)
            interpolated_images.append(reconstruction)
        
        interpolated = torch.cat(interpolated_images, dim=0)
    
    # Plot interpolation
    fig, axes = plt.subplots(1, num_steps, figsize=(num_steps * 1.5, 2))
    
    for i in range(num_steps):
        image_2d = interpolated[i].view(28, 28)
        axes[i].imshow(image_2d.detach().numpy(), cmap='gray')
        axes[i].set_title(f'α={alphas[i]:.2f}')
        axes[i].axis('off')
    
    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    
    return fig


def plot_loss_curves(
    train_losses: List[dict],
    val_losses: List[dict],
    title: str = "Training Progress"
) -> plt.Figure:
    """
    Plot training and validation loss curves.
    
    Args:
        train_losses: List of training loss dictionaries
        val_losses: List of validation loss dictionaries
        title: Plot title
    
    Returns:
        matplotlib Figure object
    """
    epochs = range(len(train_losses))
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Total loss
    axes[0].plot(epochs, [l['total'] for l in train_losses], 'b-', label='Train', alpha=0.7)
    axes[0].plot(epochs, [l['total'] for l in val_losses], 'r-', label='Validation', alpha=0.7)
    axes[0].set_title('Total Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Reconstruction loss
    axes[1].plot(epochs, [l['reconstruction'] for l in train_losses], 'b-', label='Train', alpha=0.7)
    axes[1].plot(epochs, [l['reconstruction'] for l in val_losses], 'r-', label='Validation', alpha=0.7)
    axes[1].set_title('Reconstruction Loss')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # KL loss
    axes[2].plot(epochs, [l['kl'] for l in train_losses], 'b-', label='Train', alpha=0.7)
    axes[2].plot(epochs, [l['kl'] for l in val_losses], 'r-', label='Validation', alpha=0.7)
    axes[2].set_title('KL Divergence')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Loss')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    
    return fig


def plot_latent_space_grid(
    model,
    device: str = 'cpu',
    grid_size: int = 10,
    latent_range: Tuple[float, float] = (-3, 3),
    title: str = "Latent Space Grid Visualization"
) -> plt.Figure:
    """
    Plot a grid of generated images by systematically sampling the 2D latent space.
    
    Args:
        model: Trained VAE model
        device: Device to run inference on
        grid_size: Size of the grid (grid_size x grid_size images)
        latent_range: Range of latent values to sample
        title: Plot title
    
    Returns:
        matplotlib Figure object
    """
    model.eval()
    
    # Create grid of latent coordinates
    x = np.linspace(latent_range[0], latent_range[1], grid_size)
    y = np.linspace(latent_range[0], latent_range[1], grid_size)
    
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(grid_size, grid_size))
    
    with torch.no_grad():
        for i, yi in enumerate(y):
            for j, xi in enumerate(x):
                # Create latent point
                z = torch.tensor([[xi, yi]], dtype=torch.float32, device=device)
                
                # Generate image
                generated = model.decode(z)
                image_2d = generated.view(28, 28).cpu().numpy()
                
                # Plot
                axes[i, j].imshow(image_2d, cmap='gray')
                axes[i, j].axis('off')
    
    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    
    return fig