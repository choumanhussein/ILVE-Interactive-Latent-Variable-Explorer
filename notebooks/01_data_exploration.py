"""
Data Exploration Script - Python version
Run this to test the full pipeline interactively.
"""

import sys
import os
sys.path.append('../src')

import torch
import matplotlib.pyplot as plt
import numpy as np

from data.mnist_loader import MNISTDataModule
from models.base_vae import VAE, vae_loss

def main():
    print("=== VAE Data Exploration ===\n")
    
    # Setup
    print("PyTorch version:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    print()
    
    # Test data loading
    print("=== Testing Data Loading ===")
    
    # Initialize data module with Windows-friendly settings
    data_module = MNISTDataModule(batch_size=64, num_workers=0, pin_memory=False)
    
    # Prepare and setup data
    data_module.prepare_data()
    data_module.setup()
    
    # Get sample batches
    train_batch = data_module.get_sample_batch('train')
    images, labels = train_batch
    
    print(f"Train batch shape: {images.shape}")
    print(f"Labels shape: {labels.shape}")
    print(f"Image range: [{images.min():.3f}, {images.max():.3f}]")
    print(f"Unique labels: {torch.unique(labels)}")
    print()
    
    # Visualize some samples
    print("=== Visualizing MNIST Samples ===")
    
    # Get images for visualization (2D format)
    vis_loader = data_module.vis_dataloader(batch_size=16)
    vis_images, vis_labels = next(iter(vis_loader))
    
    fig, axes = plt.subplots(4, 4, figsize=(10, 10))
    for i in range(16):
        row, col = i // 4, i % 4
        axes[row, col].imshow(vis_images[i].squeeze(), cmap='gray')
        axes[row, col].set_title(f'Label: {vis_labels[i].item()}', fontsize=12)
        axes[row, col].axis('off')
    
    plt.suptitle('MNIST Sample Images', fontsize=16)
    plt.tight_layout()
    plt.savefig('mnist_samples.png', dpi=150, bbox_inches='tight')
    print("✓ Saved MNIST samples to 'mnist_samples.png'")
    plt.show()
    print()
    
    # Test VAE model
    print("=== Testing VAE Model ===")
    
    # Create VAE model
    model = VAE(input_dim=784, latent_dim=2, hidden_dims=[512, 256])
    
    # Test forward pass
    test_input = images[:8]  # Use 8 samples from our batch
    reconstruction, mu, logvar, z = model(test_input)
    
    print(f"Input shape: {test_input.shape}")
    print(f"Reconstruction shape: {reconstruction.shape}")
    print(f"Latent mean shape: {mu.shape}")
    print(f"Latent codes shape: {z.shape}")
    
    # Calculate loss
    total_loss, recon_loss, kl_loss = vae_loss(reconstruction, test_input, mu, logvar)
    print(f"\nLoss components:")
    print(f"Total loss: {total_loss.item():.4f}")
    print(f"Reconstruction loss: {recon_loss.item():.4f}")
    print(f"KL divergence: {kl_loss.item():.4f}")
    print()
    
    # Visualize reconstructions
    print("=== Visualizing Reconstructions (Untrained Model) ===")
    
    # Convert back to 2D for visualization
    original_2d = test_input.view(-1, 28, 28)
    reconstruction_2d = reconstruction.view(-1, 28, 28)
    
    fig, axes = plt.subplots(2, 8, figsize=(16, 4))
    
    for i in range(8):
        # Original images
        axes[0, i].imshow(original_2d[i].detach().numpy(), cmap='gray')
        axes[0, i].set_title('Original')
        axes[0, i].axis('off')
        
        # Reconstructed images
        axes[1, i].imshow(reconstruction_2d[i].detach().numpy(), cmap='gray')
        axes[1, i].set_title('Reconstructed')
        axes[1, i].axis('off')
    
    plt.suptitle('Original vs Reconstructed (Untrained VAE)', fontsize=14)
    plt.tight_layout()
    plt.savefig('vae_reconstructions_untrained.png', dpi=150, bbox_inches='tight')
    print("✓ Saved reconstructions to 'vae_reconstructions_untrained.png'")
    plt.show()
    print()
    
    # Explore latent space
    print("=== Latent Space Exploration ===")
    
    # Show latent codes for our samples
    print("Latent codes (mu) for test samples:")
    for i, (latent_code, label) in enumerate(zip(mu, vis_labels[:8])):
        print(f"Sample {i} (digit {label.item()}): z = [{latent_code[0].item():.3f}, {latent_code[1].item():.3f}]")
    
    # Visualize latent space distribution
    plt.figure(figsize=(8, 6))
    colors = plt.cm.tab10(vis_labels[:8].numpy())
    scatter = plt.scatter(mu[:, 0].detach().numpy(), mu[:, 1].detach().numpy(), 
               c=colors, s=100, alpha=0.7)
    
    # Add labels for each point
    for i, (x, y, label) in enumerate(zip(mu[:, 0].detach().numpy(), 
                                         mu[:, 1].detach().numpy(), 
                                         vis_labels[:8].numpy())):
        plt.annotate(f'{label}', (x, y), xytext=(5, 5), textcoords='offset points')
    
    plt.xlabel('Latent Dimension 1')
    plt.ylabel('Latent Dimension 2')
    plt.title('Latent Space Distribution (Untrained)')
    plt.grid(True, alpha=0.3)
    plt.savefig('latent_space_untrained.png', dpi=150, bbox_inches='tight')
    print("✓ Saved latent space plot to 'latent_space_untrained.png'")
    plt.show()
    print()
    
    # Test generation
    print("=== Testing Generation ===")
    
    # Generate some samples from the prior
    generated_samples = model.generate(num_samples=8)
    generated_2d = generated_samples.view(-1, 28, 28)
    
    fig, axes = plt.subplots(2, 4, figsize=(10, 5))
    for i in range(8):
        row, col = i // 4, i % 4
        axes[row, col].imshow(generated_2d[i].detach().numpy(), cmap='gray')
        axes[row, col].set_title(f'Generated {i+1}')
        axes[row, col].axis('off')
    
    plt.suptitle('Generated Samples (Untrained VAE)', fontsize=14)
    plt.tight_layout()
    plt.savefig('vae_generated_untrained.png', dpi=150, bbox_inches='tight')
    print("✓ Saved generated samples to 'vae_generated_untrained.png'")
    plt.show()
    print()
    
    # Test interpolation
    print("=== Testing Latent Space Interpolation ===")
    
    # Take two different samples and interpolate between them
    sample1, sample2 = test_input[0:1], test_input[1:1]
    if len(sample2) == 0:  # Safety check
        sample2 = test_input[1:2]
    
    interpolated = model.interpolate(sample1, sample2, num_steps=8)
    interpolated_2d = interpolated.view(8, 28, 28)
    
    fig, axes = plt.subplots(1, 8, figsize=(16, 2))
    for i in range(8):
        axes[i].imshow(interpolated_2d[i].detach().numpy(), cmap='gray')
        axes[i].set_title(f'Step {i+1}')
        axes[i].axis('off')
    
    plt.suptitle('Latent Space Interpolation (Untrained)', fontsize=14)
    plt.tight_layout()
    plt.savefig('vae_interpolation_untrained.png', dpi=150, bbox_inches='tight')
    print("✓ Saved interpolation to 'vae_interpolation_untrained.png'")
    plt.show()
    print()
    
    print("✓ All tests completed successfully!")
    print("\nGenerated files:")
    print("- mnist_samples.png")
    print("- vae_reconstructions_untrained.png") 
    print("- latent_space_untrained.png")
    print("- vae_generated_untrained.png")
    print("- vae_interpolation_untrained.png")
    print("\nNext steps:")
    print("1. Implement training loop")
    print("2. Add experiment tracking with Weights & Biases")
    print("3. Train the model and observe latent space organization")
    print("4. Implement β-VAE for disentanglement")

if __name__ == "__main__":
    main()