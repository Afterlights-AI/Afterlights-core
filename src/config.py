import yaml
def load_config(config_path):
    with open(config_path, 'r', encoding='utf-8') as file:
        config = yaml.safe_load(file)
    return config

CONFIG = load_config("config/project_config.yaml")


TRAINING_CONFIG = CONFIG.get("training")
EPOCHS = TRAINING_CONFIG.get("epochs")
BATCH_SIZE = TRAINING_CONFIG.get("batch_size")



