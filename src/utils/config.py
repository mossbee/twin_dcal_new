import yaml
import os
from typing import Dict, Any
from pathlib import Path


class Config:
    """Configuration management utility."""
    
    def __init__(self, config_path: str = None, **kwargs):
        """
        Initialize configuration.
        
        Args:
            config_path: Path to YAML config file
            **kwargs: Additional config overrides
        """
        self.config = {}
        
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
        
        # Apply overrides
        self._update_config(self.config, kwargs)
    
    def _update_config(self, config: Dict, updates: Dict):
        """Recursively update config with new values."""
        for key, value in updates.items():
            if isinstance(value, dict) and key in config and isinstance(config[key], dict):
                self._update_config(config[key], value)
            else:
                config[key] = value
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get config value using dot notation."""
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def set(self, key: str, value: Any):
        """Set config value using dot notation."""
        keys = key.split('.')
        config = self.config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
    
    def to_dict(self) -> Dict:
        """Return config as dictionary."""
        return self.config.copy()
    
    def save(self, path: str):
        """Save config to YAML file."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False, indent=2)
    
    def update_from_args(self, args):
        """Update config from command line arguments."""
        if hasattr(args, 'config_overrides') and args.config_overrides:
            for override in args.config_overrides:
                if '=' in override:
                    key, value = override.split('=', 1)
                    # Try to parse value as appropriate type
                    try:
                        if value.lower() in ['true', 'false']:
                            value = value.lower() == 'true'
                        elif value.isdigit():
                            value = int(value)
                        elif '.' in value and value.replace('.', '').isdigit():
                            value = float(value)
                    except:
                        pass  # Keep as string
                    
                    self.set(key, value)


def load_config(config_path: str, **overrides) -> Config:
    """Load configuration from file with optional overrides."""
    return Config(config_path, **overrides)


def create_local_config():
    """Create local server configuration."""
    config = {
        'data': {
            'dataset_root': '/path/to/local/dataset',
            'batch_size': 32,
            'num_workers': 8
        },
        'training': {
            'device': 'cuda',
            'distributed': True,
            'world_size': 2
        },
        'tracking': {
            'method': 'mlflow',
            'mlflow_uri': 'http://localhost:5000'
        }
    }
    
    os.makedirs('configs', exist_ok=True)
    with open('configs/local_config.yaml', 'w') as f:
        yaml.dump(config, f, default_flow_style=False, indent=2)


def create_kaggle_config():
    """Create Kaggle configuration."""
    config = {
        'data': {
            'dataset_root': '/kaggle/input/nd-twin/ND_TWIN_Dataset_224/ND_TWIN_Dataset_224',
            'batch_size': 16,
            'num_workers': 2
        },
        'training': {
            'device': 'cuda',
            'distributed': False,
            'world_size': 1,
            'epochs': 50,  # Reduced for 12h limit
            'save_every': 1  # Save every epoch for resuming
        },
        'tracking': {
            'method': 'wandb',
            'project_name': 'twin_dcal_kaggle',
            'entity': 'hunchoquavodb-hanoi-university-of-science-and-technology'
        }
    }
    
    os.makedirs('configs', exist_ok=True)
    with open('configs/kaggle_config.yaml', 'w') as f:
        yaml.dump(config, f, default_flow_style=False, indent=2)


def setup_environment_configs():
    """Setup configuration files for different environments."""
    create_local_config()
    create_kaggle_config()
    print("Environment configuration files created:")
    print("- configs/local_config.yaml")
    print("- configs/kaggle_config.yaml")


if __name__ == "__main__":
    setup_environment_configs()
