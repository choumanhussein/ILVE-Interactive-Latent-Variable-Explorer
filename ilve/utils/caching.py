# utils/caching.py
"""
Caching utilities for ILVE framework.
Handles model loading and analysis result caching.
"""

import streamlit as st
import hashlib
import pickle
import time
from typing import Any, Dict, Optional, Callable
from functools import wraps

class CacheManager:
    """Manages caching for expensive operations."""
    
    @staticmethod
    def generate_cache_key(*args, **kwargs) -> str:
        """Generate a cache key from arguments."""
       
        key_data = str(args) + str(sorted(kwargs.items()))
        

        return hashlib.md5(key_data.encode()).hexdigest()
    
    @staticmethod
    def cache_analysis_result(func: Callable) -> Callable:
        """Decorator to cache analysis results."""
        @wraps(func)
        def wrapper(*args, **kwargs):
            cache_key = f"analysis_{func.__name__}_{CacheManager.generate_cache_key(*args, **kwargs)}"
            
           
            if cache_key in st.session_state:
                cached_result, timestamp = st.session_state[cache_key]
                
                
                if time.time() - timestamp < 3600:
                    return cached_result
            
            
            result = func(*args, **kwargs)
            st.session_state[cache_key] = (result, time.time())
            
            return result
        
        return wrapper
    
    @staticmethod
    def clear_analysis_cache():
        """Clear all cached analysis results."""
        keys_to_remove = [key for key in st.session_state.keys() if key.startswith('analysis_')]
        for key in keys_to_remove:
            del st.session_state[key]
    
    @staticmethod
    def get_cache_stats() -> Dict[str, Any]:
        """Get statistics about cached items."""
        analysis_keys = [key for key in st.session_state.keys() if key.startswith('analysis_')]
        
        total_size = 0
        for key in analysis_keys:
            try:
                size = len(pickle.dumps(st.session_state[key]))
                total_size += size
            except:
                pass
        
        return {
            'cached_analyses': len(analysis_keys),
            'total_size_mb': total_size / (1024 * 1024),
            'cache_keys': analysis_keys
        }

def cache_analysis_result(func: Callable) -> Callable:
    """Convenience function for caching analysis results."""
    return CacheManager.cache_analysis_result(func)

