import yaml
import os

def load_config(config_path):
    with open(config_path, 'r', encoding='utf-8') as file:
        config = yaml.safe_load(file)
    return config

# Always resolve config path relative to project root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_PATH = os.path.join(PROJECT_ROOT, "config", "project_config.yaml")
CONFIG = load_config(CONFIG_PATH)

TRAINING_CONFIG = CONFIG.get("training")
EPOCHS = TRAINING_CONFIG.get("epochs")
BATCH_SIZE = TRAINING_CONFIG.get("batch_size")