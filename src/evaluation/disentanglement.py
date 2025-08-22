"""
Disentanglement Analysis and Visualization Tools

This module provides tools for analyzing and visualizing disentanglement
in continuous latent variable models, particularly β-VAE.
"""

import torch
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import matplotlib.gridspec as gridspec
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def plot_latent_traversal(
    traversals: Dict[int, torch.Tensor],
    num_dims_to_show: int = 6,
    num_samples_to_show: int = 3,
    title: str = "Latent Space Traversal"
) -> plt.Figure:
    """
    Plot latent traversal results to visualize disentanglement.
    
    Args:
        traversals: Dictionary mapping dimension to generated images
        num_dims_to_show: Number of dimensions to visualize
        num_samples_to_show: Number of samples per dimension
        title: Plot title
        
    Returns:
        matplotlib Figure object
    """
    num_dims = min(num_dims_to_show, len(traversals))
    num_steps = traversals[0].shape[2]  # [samples, steps, height*width]
    
    fig = plt.figure(figsize=(num_steps * 1.5, num_dims * num_samples_to_show * 1.5))
    gs = gridspec.GridSpec(num_dims * num_samples_to_show, num_steps, figure=fig)
    
    for dim_idx in range(num_dims):
        if dim_idx not in traversals:
            continue
            
        dim_data = traversals[dim_idx]  # [num_samples, num_steps, 784]
        
        for sample_idx in range(min(num_samples_to_show, dim_data.shape[0])):
            for step_idx in range(num_steps):
                row = dim_idx * num_samples_to_show + sample_idx
                col = step_idx
                
                ax = fig.add_subplot(gs[row, col])
                
                # Reshape and display image
                image = dim_data[sample_idx, step_idx].view(28, 28).detach().numpy()
                ax.imshow(image, cmap='gray')
                ax.axis('off')
                
                # Add labels
                if step_idx == 0:
                    ax.set_ylabel(f'Dim {dim_idx}\nSample {sample_idx}', rotation=0, 
                                ha='right', va='center')
                if row == 0:
                    ax.set_title(f'Step {step_idx}')
    
    plt.suptitle(title, fontsize=16, y=0.98)
    plt.tight_layout()
    
    return fig


def plot_disentanglement_metrics(
    metrics: Dict[str, any],
    title: str = "Disentanglement Analysis"
) -> plt.Figure:
    """
    Plot disentanglement metrics.
    
    Args:
        metrics: Dictionary containing disentanglement metrics
        title: Plot title
        
    Returns:
        matplotlib Figure object
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # Plot 1: Dimension variances
    if 'dimension_variances' in metrics:
        dim_vars = metrics['dimension_variances']
        dims = list(dim_vars.keys())
        mean_vars = [dim_vars[d]['mean_var'] for d in dims]
        std_vars = [dim_vars[d]['std_var'] for d in dims]
        
        axes[0, 0].bar(dims, mean_vars, yerr=std_vars, alpha=0.7, capsize=5)
        axes[0, 0].set_xlabel('Latent Dimension')
        axes[0, 0].set_ylabel('Mean Variance')
        axes[0, 0].set_title('Variance per Dimension')
        axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Dimension separations
    if 'dimension_separations' in metrics:
        separations = metrics['dimension_separations']
        dims = list(range(len(separations)))
        
        axes[0, 1].bar(dims, separations, alpha=0.7)
        axes[0, 1].set_xlabel('Latent Dimension')
        axes[0, 1].set_ylabel('Separation Score')
        axes[0, 1].set_title('Digit Separation per Dimension')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Highlight most separating dimension
        max_dim = np.argmax(separations)
        axes[0, 1].bar(max_dim, separations[max_dim], color='red', alpha=0.8)
    
    # Plot 3: Summary metrics
    summary_data = []
    summary_labels = []
    
    if 'avg_separation' in metrics:
        summary_data.append(metrics['avg_separation'])
        summary_labels.append('Avg Separation')
    
    if 'max_separation' in metrics:
        summary_data.append(metrics['max_separation'])
        summary_labels.append('Max Separation')
    
    if summary_data:
        axes[1, 0].bar(summary_labels, summary_data, alpha=0.7)
        axes[1, 0].set_ylabel('Score')
        axes[1, 0].set_title('Summary Metrics')
        axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Variance distribution
    if 'dimension_variances' in metrics:
        all_vars = []
        for dim in metrics['dimension_variances']:
            all_vars.append(metrics['dimension_variances'][dim]['mean_var'])
        
        axes[1, 1].hist(all_vars, bins=10, alpha=0.7, edgecolor='black')
        axes[1, 1].set_xlabel('Variance')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].set_title('Variance Distribution')
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    
    return fig


def plot_beta_comparison(
    results: Dict[float, Dict],
    metric_name: str = 'avg_separation',
    title: str = "β-VAE Comparison"
) -> plt.Figure:
    """
    Compare disentanglement across different β values.
    
    Args:
        results: Dictionary mapping β values to metric dictionaries
        metric_name: Which metric to compare
        title: Plot title
        
    Returns:
        matplotlib Figure object
    """
    beta_values = sorted(results.keys())
    metric_values = [results[beta].get(metric_name, 0) for beta in beta_values]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(beta_values, metric_values, 'bo-', linewidth=2, markersize=8)
    ax.set_xlabel('β Value')
    ax.set_ylabel(metric_name.replace('_', ' ').title())
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    
    # Highlight standard VAE (β=1)
    if 1.0 in beta_values:
        idx = beta_values.index(1.0)
        ax.plot(1.0, metric_values[idx], 'ro', markersize=12, 
               label='Standard VAE (β=1)')
        ax.legend()
    
    plt.tight_layout()
    
    return fig


def plot_latent_space_by_digit(
    latent_codes: torch.Tensor,
    labels: torch.Tensor,
    method: str = 'pca',
    title: str = "Latent Space Visualization"
) -> plt.Figure:
    """
    Visualize high-dimensional latent space using dimensionality reduction.
    
    Args:
        latent_codes: Latent codes [num_samples, latent_dim]
        labels: Corresponding labels [num_samples]
        method: 'pca' or 'tsne'
        title: Plot title
        
    Returns:
        matplotlib Figure object
    """
    # Convert to numpy
    latent_np = latent_codes.detach().cpu().numpy()
    labels_np = labels.detach().cpu().numpy()
    
    # Dimensionality reduction
    if method == 'pca':
        reducer = PCA(n_components=2)
        reduced = reducer.fit_transform(latent_np)
        subtitle = f'PCA (Variance explained: {reducer.explained_variance_ratio_.sum():.3f})'
    elif method == 'tsne':
        reducer = TSNE(n_components=2, random_state=42)
        reduced = reducer.fit_transform(latent_np)
        subtitle = 't-SNE'
    else:
        raise ValueError("Method must be 'pca' or 'tsne'")
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 8))
    
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    
    for digit in range(10):
        mask = labels_np == digit
        if mask.any():
            ax.scatter(reduced[mask, 0], reduced[mask, 1], 
                      c=[colors[digit]], label=f'Digit {digit}', 
                      alpha=0.7, s=30)
    
    ax.set_xlabel('Component 1')
    ax.set_ylabel('Component 2')
    ax.set_title(f'{title}\n{subtitle}')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    return fig


def compute_disentanglement_score(
    model,
    test_loader,
    num_samples: int = 1000,
    device: str = 'cpu'
) -> Dict[str, float]:
    """
    Compute comprehensive disentanglement scores.
    
    This is a simplified version of disentanglement metrics.
    For research, you'd want to implement more sophisticated metrics like:
    - MIG (Mutual Information Gap)
    - SAP (Separated Attribute Predictability)
    - DCI (Disentanglement, Completeness, Informativeness)
    
    Args:
        model: Trained VAE model
        test_loader: Test data loader
        num_samples: Number of samples to use
        device: Device to run on
        
    Returns:
        Dictionary of disentanglement scores
    """
    model.eval()
    
    latent_codes = []
    labels = []
    
    with torch.no_grad():
        for batch_idx, (data, batch_labels) in enumerate(test_loader):
            if len(latent_codes) * test_loader.batch_size >= num_samples:
                break
                
            data = data.to(device)
            mu, _ = model.encode(data)
            
            latent_codes.append(mu.cpu())
            labels.append(batch_labels)
    
    latent_codes = torch.cat(latent_codes, dim=0)[:num_samples]
    labels = torch.cat(labels, dim=0)[:num_samples]
    
    # Basic disentanglement metrics
    scores = {}
    
    # 1. Compute variance of each dimension
    dim_variances = torch.var(latent_codes, dim=0)
    scores['dimension_variances'] = dim_variances.tolist()
    scores['variance_entropy'] = -torch.sum(dim_variances * torch.log(dim_variances + 1e-8)).item()
    
    # 2. Compute digit separability in each dimension
    digit_separability = []
    for dim in range(latent_codes.shape[1]):
        dim_values = latent_codes[:, dim]
        
        # Compute mean distance between digit classes
        separability = 0
        count = 0
        for digit1 in range(10):
            for digit2 in range(digit1 + 1, 10):
                mask1 = labels == digit1
                mask2 = labels == digit2
                
                if mask1.sum() > 0 and mask2.sum() > 0:
                    mean1 = dim_values[mask1].mean()
                    mean2 = dim_values[mask2].mean()
                    separability += torch.abs(mean1 - mean2).item()
                    count += 1
        
        if count > 0:
            digit_separability.append(separability / count)
        else:
            digit_separability.append(0)
    
    scores['digit_separability'] = digit_separability
    scores['avg_separability'] = np.mean(digit_separability)
    scores['max_separability'] = np.max(digit_separability)
    
    # 3. Effective dimensionality (number of dimensions with significant variance)
    normalized_vars = dim_variances / dim_variances.sum()
    effective_dims = (normalized_vars > 0.01).sum().item()  # Threshold: 1% of total variance
    scores['effective_dimensions'] = effective_dims
    
    return scores


def create_disentanglement_report(
    model,
    test_loader,
    device: str = 'cpu',
    save_path: Optional[str] = None
) -> Dict[str, any]:
    """
    Create a comprehensive disentanglement analysis report.
    
    Args:
        model: Trained model
        test_loader: Test data loader
        device: Device to run on
        save_path: Path to save visualizations
        
    Returns:
        Dictionary containing all analysis results
    """
    print("Creating disentanglement analysis report...")
    
    # 1. Compute metrics
    metrics = compute_disentanglement_score(model, test_loader, device=device)
    
    # 2. Perform latent traversal (if β-VAE)
    traversals = None
    if hasattr(model, 'latent_traversal'):
        print("Performing latent traversal...")
        traversals = model.latent_traversal(num_samples=3, device=device)
    
    # 3. Collect latent representations
    print("Collecting latent representations...")
    latent_codes = []
    labels = []
    
    with torch.no_grad():
        for batch_idx, (data, batch_labels) in enumerate(test_loader):
            if batch_idx > 10:  # Limit for efficiency
                break
                
            data = data.to(device)
            mu, _ = model.encode(data)
            latent_codes.append(mu.cpu())
            labels.append(batch_labels)
    
    latent_codes = torch.cat(latent_codes, dim=0)
    labels = torch.cat(labels, dim=0)
    
    # 4. Create visualizations
    visualizations = {}
    
    # Metrics plot
    fig_metrics = plot_disentanglement_metrics(
        {'dimension_separations': metrics['digit_separability']},
        title="Disentanglement Metrics"
    )
    visualizations['metrics'] = fig_metrics
    
    # Latent traversal plot
    if traversals:
        fig_traversal = plot_latent_traversal(
            traversals, 
            title="Latent Space Traversal"
        )
        visualizations['traversal'] = fig_traversal
    
    # PCA visualization
    if latent_codes.shape[1] > 2:
        fig_pca = plot_latent_space_by_digit(
            latent_codes, labels, method='pca',
            title="Latent Space (PCA)"
        )
        visualizations['pca'] = fig_pca
    
    # Save visualizations
    if save_path:
        import os
        os.makedirs(save_path, exist_ok=True)
        
        for name, fig in visualizations.items():
            fig.savefig(os.path.join(save_path, f'{name}.png'), 
                       dpi=150, bbox_inches='tight')
            plt.close(fig)
    
    report = {
        'metrics': metrics,
        'traversals': traversals,
        'latent_codes': latent_codes,
        'labels': labels,
        'visualizations': visualizations
    }
    
    print("✓ Disentanglement analysis complete!")
    
    return report