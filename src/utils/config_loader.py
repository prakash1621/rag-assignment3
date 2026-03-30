"""Configuration loader for YAML config files"""

import yaml
import os
from pathlib import Path


def load_config(config_path: str = None) -> dict:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to config.yaml file. If None, looks in project root.
    
    Returns:
        Configuration dictionary
    """
    if config_path is None:
        # Look for config.yaml in project root
        base_dir = Path(__file__).parent.parent.parent
        config_path = base_dir / "config.yaml"
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config
