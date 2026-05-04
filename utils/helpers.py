"""
Helpers: Utility functions untuk Semantic Echo.
"""

import os
import json
import pickle
from pathlib import Path
from typing import Any, Dict, Optional


def save_model(model: Any, path: str) -> None:
    """
    Save model to disk.
    
    Args:
        model: Model object to save
        path: Path to save the model
    """
    import torch
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)


def load_model(model: Any, path: str) -> Any:
    """
    Load model from disk.
    
    Args:
        model: Model architecture to load weights into
        path: Path to the saved model
        
    Returns:
        Model with loaded weights
    """
    import torch
    model.load_state_dict(torch.load(path))
    return model


def save_cache(data: Any, path: str) -> None:
    """
    Save data to cache using pickle.
    
    Args:
        data: Data to cache
        path: Path to save the cache
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(data, f)


def load_cache(path: str) -> Optional[Any]:
    """
    Load data from cache.
    
    Args:
        path: Path to the cached data
        
    Returns:
        Cached data or None if not found
    """
    if not os.path.exists(path):
        return None
        
    with open(path, 'rb') as f:
        return pickle.load(f)


def ensure_dir(path: str) -> Path:
    """
    Ensure directory exists.
    
    Args:
        path: Directory path
        
    Returns:
        Path object for the directory
    """
    dir_path = Path(path)
    dir_path.mkdir(parents=True, exist_ok=True)
    return dir_path


def load_config(config_path: str) -> Dict:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to config.yaml
        
    Returns:
        Configuration dictionary
    """
    import yaml
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def setup_logging(log_dir: str, level: str = "INFO") -> None:
    """
    Setup logging configuration.
    
    Args:
        log_dir: Directory for log files
        level: Logging level
    """
    import logging
    
    ensure_dir(log_dir)
    
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(log_dir, 'semantic_echo.log')),
            logging.StreamHandler()
        ]
    )
