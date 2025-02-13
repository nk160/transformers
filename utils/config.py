import yaml
from pathlib import Path
from typing import Dict, Any

def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def save_config(config: Dict[str, Any], save_path: str) -> None:
    """Save configuration to YAML file"""
    with open(save_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False) 