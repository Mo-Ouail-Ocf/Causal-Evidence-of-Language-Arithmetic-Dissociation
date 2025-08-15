"""
Configuration file for Language-Arithmetic Dissociation Study
Easy switching between different models and settings
"""

# Model configurations
MODELS = {
    # Small models for quick testing
    "quick_test": {
        "name": "microsoft/DialoGPT-small",
        "device": "cpu",
        "description": "Small model (~117M params) for quick testing"
    },
    
    # Medium models for faster experiments
    "medium": {
        "name": "microsoft/DialoGPT-medium", 
        "device": "cpu",
        "description": "Medium model (~345M params) for faster experiments"
    },
    
    # Large models for full experiments
    "qwen_small": {
        "name": "Qwen/Qwen2.5-3B-Instruct",
        "device": "auto",
        "description": "Qwen2.5-3B-Instruct (~3B params) - good balance"
    },
    
    "qwen_large": {
        "name": "Qwen/Qwen2.5-7B-Instruct", 
        "device": "auto",
        "description": "Qwen2.5-7B-Instruct (~7B params) - full experiment"
    },
    
    "mistral": {
        "name": "mistralai/Mistral-7B-Instruct-v0.2",
        "device": "auto", 
        "description": "Mistral-7B-Instruct (~7B params) - alternative"
    }
}

# Experiment configurations
EXPERIMENT_CONFIGS = {
    "quick_test": {
        "top_k_neurons": 3,
        "max_layers": 1,
        "dataset_size": 3,
        "description": "Quick validation test"
    },
    
    "small_experiment": {
        "top_k_neurons": 10,
        "max_layers": 3,
        "dataset_size": 10,
        "description": "Small-scale experiment"
    },
    
    "full_experiment": {
        "top_k_neurons": 20,
        "max_layers": 5,
        "dataset_size": 20,
        "description": "Full-scale experiment"
    }
}

# Default configurations
DEFAULT_MODEL = "qwen_large"
DEFAULT_EXPERIMENT = "full_experiment"

def get_config(model_key=None, experiment_key=None):
    """
    Get configuration for model and experiment
    
    Args:
        model_key: Key from MODELS dict (default: DEFAULT_MODEL)
        experiment_key: Key from EXPERIMENT_CONFIGS dict (default: DEFAULT_EXPERIMENT)
    
    Returns:
        Dictionary with combined configuration
    """
    model_key = model_key or DEFAULT_MODEL
    experiment_key = experiment_key or DEFAULT_EXPERIMENT
    
    if model_key not in MODELS:
        raise ValueError(f"Unknown model key: {model_key}. Available: {list(MODELS.keys())}")
    
    if experiment_key not in EXPERIMENT_CONFIGS:
        raise ValueError(f"Unknown experiment key: {experiment_key}. Available: {list(EXPERIMENT_CONFIGS.keys())}")
    
    config = {
        "model": MODELS[model_key],
        "experiment": EXPERIMENT_CONFIGS[experiment_key]
    }
    
    return config

def print_available_configs():
    """Print all available configurations"""
    print("Available Models:")
    print("-" * 40)
    for key, config in MODELS.items():
        print(f"{key:15} - {config['description']}")
    
    print("\nAvailable Experiment Configurations:")
    print("-" * 40)
    for key, config in EXPERIMENT_CONFIGS.items():
        print(f"{key:15} - {config['description']}")
        print(f"{'':15}   top_k_neurons: {config['top_k_neurons']}")
        print(f"{'':15}   max_layers: {config['max_layers']}")
        print(f"{'':15}   dataset_size: {config['dataset_size']}")

if __name__ == "__main__":
    print("Configuration Options for Language-Arithmetic Dissociation Study")
    print("="*60)
    print_available_configs()
    
    print(f"\nDefault Configuration:")
    print(f"Model: {DEFAULT_MODEL}")
    print(f"Experiment: {DEFAULT_EXPERIMENT}")
    
    config = get_config()
    print(f"\nDefault Model: {config['model']['name']}")
    print(f"Default Device: {config['model']['device']}")
    print(f"Default Top-k Neurons: {config['experiment']['top_k_neurons']}") 