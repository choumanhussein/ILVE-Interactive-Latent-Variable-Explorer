"""
MNIST data loading and preprocessing utilities for VAE training.
"""

import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np


class FlattenTransform:
    """Custom transform to flatten images - needed for Windows multiprocessing."""
    def __call__(self, x):
        return x.view(-1)


class MNISTDataModule:
    """
    Data module for MNIST dataset with VAE-specific preprocessing.
    """
    
    def __init__(self, data_dir="./data", batch_size=128, num_workers=0, pin_memory=False):
        self.data_dir = data_dir
        self.batch_size = batch_size
        # Set num_workers=0 for Windows to avoid multiprocessing issues
        self.num_workers = 0 if num_workers is None else num_workers
        # Disable pin_memory on CPU-only setups
        self.pin_memory = pin_memory and torch.cuda.is_available()
        
        # Transform for VAE: normalize to [0, 1] and flatten
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            FlattenTransform()  # Replace lambda with proper class
        ])
        
        # For visualization (keep 2D shape)
        self.vis_transform = transforms.Compose([
            transforms.ToTensor()
        ])
        
    def prepare_data(self):
        """Download MNIST data if not already present."""
        torchvision.datasets.MNIST(
            root=self.data_dir, 
            train=True, 
            download=True
        )
        torchvision.datasets.MNIST(
            root=self.data_dir, 
            train=False, 
            download=True
        )
        
    def setup(self):
        """Create train, validation, and test datasets."""
        # Training set
        train_full = torchvision.datasets.MNIST(
            root=self.data_dir,
            train=True,
            transform=self.transform,
            download=False
        )
        
        # Split training into train/validation (50k/10k)
        train_size = 50000
        val_size = 10000
        self.train_dataset, self.val_dataset = torch.utils.data.random_split(
            train_full, [train_size, val_size]
        )
        
        # Test set
        self.test_dataset = torchvision.datasets.MNIST(
            root=self.data_dir,
            train=False,
            transform=self.transform,
            download=False
        )
        
        # Visualization dataset (keeps 2D shape)
        self.vis_dataset = torchvision.datasets.MNIST(
            root=self.data_dir,
            train=False,
            transform=self.vis_transform,
            download=False
        )
        
    def train_dataloader(self):
        """Create training dataloader."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=True
        )
        
    def val_dataloader(self):
        """Create validation dataloader."""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory
        )
        
    def test_dataloader(self):
        """Create test dataloader."""
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory
        )
        
    def vis_dataloader(self, batch_size=64):
        """Create dataloader for visualization (2D images)."""
        return DataLoader(
            self.vis_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory
        )
        
    def visualize_batch(self, batch_size=16):
        """Visualize a batch of MNIST images."""
        vis_loader = self.vis_dataloader(batch_size=batch_size)
        images, labels = next(iter(vis_loader))
        
        fig, axes = plt.subplots(4, 4, figsize=(8, 8))
        for i, (img, label) in enumerate(zip(images, labels)):
            row, col = i // 4, i % 4
            axes[row, col].imshow(img.squeeze(), cmap='gray')
            axes[row, col].set_title(f'Label: {label.item()}')
            axes[row, col].axis('off')
            
        plt.tight_layout()
        plt.show()
        
    def get_sample_batch(self, split='train'):
        """Get a sample batch for testing."""
        if split == 'train':
            loader = self.train_dataloader()
        elif split == 'val':
            loader = self.val_dataloader()
        else:
            loader = self.test_dataloader()
            
        return next(iter(loader))


def test_data_loading():
    """Test function to verify data loading works correctly."""
    print("Testing MNIST data loading...")
    
    # Initialize data module with Windows-friendly settings
    data_module = MNISTDataModule(batch_size=32, num_workers=0, pin_memory=False)
    
    # Prepare and setup data
    data_module.prepare_data()
    data_module.setup()
    
    # Test data loaders
    train_batch = data_module.get_sample_batch('train')
    val_batch = data_module.get_sample_batch('val')
    test_batch = data_module.get_sample_batch('test')
    
    print(f"Train batch shape: {train_batch[0].shape}")
    print(f"Val batch shape: {val_batch[0].shape}")
    print(f"Test batch shape: {test_batch[0].shape}")
    
    # Verify data is properly normalized and flattened
    train_images, train_labels = train_batch
    print(f"Image range: [{train_images.min():.3f}, {train_images.max():.3f}]")
    print(f"Image shape: {train_images.shape}")  # Should be [batch_size, 784]
    print(f"Labels shape: {train_labels.shape}")
    
    print("✓ Data loading test passed!")
    
    return data_module


if __name__ == "__main__":
    data_module = test_data_loading()
