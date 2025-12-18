"""Configuration loading and validation"""
import yaml
from pathlib import Path
from typing import Dict, Any, Optional


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file
    
    Args:
        config_path: Path to YAML configuration file
        
    Returns:
        Dictionary containing configuration
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Validate required fields
    required_fields = ['dataset_name', 'input_dir', 'output_dir']
    for field in required_fields:
        if field not in config:
            raise ValueError(f"Missing required config field: {field}")
    
    return config


def save_config(config: Dict[str, Any], output_path: str):
    """Save configuration to YAML file
    
    Args:
        config: Configuration dictionary
        output_path: Path to save configuration
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)


def get_default_config(dataset_name: str = "dataset1") -> Dict[str, Any]:
    """Get default configuration template
    
    Args:
        dataset_name: Name of the dataset
        
    Returns:
        Default configuration dictionary
    """
    return {
        'dataset_name': dataset_name,
        'input_dir': f'data/{dataset_name}/images/raw',
        'output_dir': f'data/{dataset_name}/processed',
        'preprocessing': {
            'channels': {
                'dapi': 1,
                'cd3': 2,
                'aicda': 3
            },
            'quantile_normalization': {
                'enabled': True,
                'quantiles': [0.01, 0.998],
                'mask_dir': None
            }
        },
        'segmentation': {
            'prob_thresh': 0.43,
            'use_pretrained': True
        },
        'feature_extraction': {
            'cell_segmentation': {
                'enabled': True,
                'dilation_radius': 10
            },
            'extract_spatial': True
        },
        'analysis': {
            'analyses': [
                {'type': 'cell_type', 'enabled': True},
                {'type': 'tcell_interaction', 'enabled': False},
                {'type': 'boundary', 'enabled': False},
                {'type': 'correlation', 'enabled': False}
            ]
        }
    }

