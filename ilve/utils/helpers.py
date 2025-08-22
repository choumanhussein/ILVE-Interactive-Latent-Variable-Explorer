# =============================================================================
# utils/helpers.py
"""
General helper utilities for ILVE framework.
"""

import numpy as np
import torch
from typing import List, Tuple, Any, Union, Optional
import streamlit as st
from PIL import Image
import io

class GeneralHelpers:
    """General utility functions."""
    
    @staticmethod
    def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
        """Safely divide two numbers, returning default if denominator is zero."""
        return numerator / denominator if denominator != 0 else default
    
    @staticmethod
    def clamp(value: float, min_val: float, max_val: float) -> float:
        """Clamp a value between min and max."""
        return max(min_val, min(max_val, value))
    
    @staticmethod
    def format_number(value: Union[int, float], precision: int = 3) -> str:
        """Format a number for display."""
        if isinstance(value, int):
            return f"{value:,}"
        else:
            return f"{value:.{precision}f}"
    
    @staticmethod
    def progress_bar_with_text(progress: float, text: str) -> None:
        """Display progress bar with custom text."""
        st.progress(progress, text=text)
    
    @staticmethod
    def create_download_link(data: Any, filename: str, link_text: str) -> str:
        """Create a download link for data."""
        import base64
        
        
        if isinstance(data, str):
            data_bytes = data.encode()
        elif isinstance(data, (dict, list)):
            import json
            data_bytes = json.dumps(data, indent=2).encode()
        else:
            data_bytes = str(data).encode()
        

        b64 = base64.b64encode(data_bytes).decode()
        

        return f'<a href="data:file/txt;base64,{b64}" download="{filename}">{link_text}</a>'

class ImageHelpers:
    """Helper functions for image processing and display."""
    
    @staticmethod
    def numpy_to_pil(image: np.ndarray) -> Image.Image:
        """Convert numpy array to PIL Image."""
        if image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)
        else:
            image = image.astype(np.uint8)
        

        if len(image.shape) == 2: 
            return Image.fromarray(image, mode='L')
        elif len(image.shape) == 3:
            if image.shape[2] == 3:  
                return Image.fromarray(image, mode='RGB')
            elif image.shape[2] == 1:  
                return Image.fromarray(image.squeeze(), mode='L')
        
        
        return Image.fromarray(image)
    
    @staticmethod
    def pil_to_numpy(image: Image.Image) -> np.ndarray:
        """Convert PIL Image to numpy array."""
        return np.array(image) / 255.0
    
    @staticmethod
    def create_image_grid(images: List[np.ndarray], 
                         grid_size: Optional[Tuple[int, int]] = None) -> np.ndarray:
        """Create a grid of images."""
        if not images:
            return np.zeros((100, 100))
        
        
        if grid_size is None:
            cols = int(np.ceil(np.sqrt(len(images))))
            rows = int(np.ceil(len(images) / cols))
        else:
            rows, cols = grid_size
        
        img_h, img_w = images[0].shape[:2]
        

        grid = np.ones((rows * img_h, cols * img_w)) * 0.5  
        
        for i, img in enumerate(images):
            if i >= rows * cols:
                break
                
            row = i // cols
            col = i % cols
            
            start_row = row * img_h
            end_row = start_row + img_h
            start_col = col * img_w
            end_col = start_col + img_w
            
            if len(img.shape) == 2:  
                grid[start_row:end_row, start_col:end_col] = img
            else:  
                grid[start_row:end_row, start_col:end_col] = img[:, :, 0]
        
        return grid
    
    @staticmethod
    def save_image_as_bytes(image: np.ndarray, format: str = 'PNG') -> bytes:
        """Save image as bytes for download."""
        pil_image = ImageHelpers.numpy_to_pil(image)
        
        img_bytes = io.BytesIO()
        pil_image.save(img_bytes, format=format)
        img_bytes.seek(0)
        
        return img_bytes.getvalue()

class MathHelpers:
    """Mathematical utility functions."""
    
    @staticmethod
    def normalize_array(arr: np.ndarray, method: str = 'minmax') -> np.ndarray:
        """Normalize an array using different methods."""
        if method == 'minmax':
            min_val, max_val = arr.min(), arr.max()
            if max_val > min_val:
                return (arr - min_val) / (max_val - min_val)
            else:
                return arr
        elif method == 'zscore':
            mean_val, std_val = arr.mean(), arr.std()
            if std_val > 0:
                return (arr - mean_val) / std_val
            else:
                return arr - mean_val
        else:
            raise ValueError(f"Unknown normalization method: {method}")
    
    @staticmethod
    def calculate_correlation(x: List[float], y: List[float]) -> float:
        """Calculate correlation coefficient between two lists."""
        if len(x) != len(y) or len(x) < 2:
            return 0.0
        
        try:
            correlation = np.corrcoef(x, y)[0, 1]
            return correlation if not np.isnan(correlation) else 0.0
        except:
            return 0.0
    
    @staticmethod
    def interpolate_values(start: float, end: float, num_steps: int) -> List[float]:
        """Create interpolated values between start and end."""
        return np.linspace(start, end, num_steps).tolist()
    
    @staticmethod
    def calculate_entropy(values: List[float]) -> float:
        """Calculate entropy of a list of values."""
        if not values:
            return 0.0
        

        values_array = np.array(values)
        values_array = values_array - values_array.min()  
        
        if values_array.sum() == 0:
            return 0.0
        
        probabilities = values_array / values_array.sum()
        
        
        entropy = 0.0
        for p in probabilities:
            if p > 0:
                entropy -= p * np.log2(p)
        
        return entropy


