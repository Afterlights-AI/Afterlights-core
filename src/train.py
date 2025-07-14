from training.cl_training import CLTraining
from training.anchor_cl_mining import CLSimplePairMining
from config import EPOCHS, BATCH_SIZE
import yaml
import logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def train_mode(
    model_name, 
    file_path, 
    model_output_path, 
):
    miner = CLSimplePairMining()
    trainer = CLTraining(model_name=model_name)

    pairs = miner.create_dataset(file_path)
    
    logger.info(f"preparing data on file: {file_path}")
    train_dataset = trainer.prepare_inverse_data_for_simcse(pairs)
    logger.info(f"training starts for model {model_name}, saving to {model_output_path}")
    trainer.train_simcse(train_dataset, batch_size=BATCH_SIZE, epochs=EPOCHS, warmup_steps=len(train_dataset)//10)    
    logger.info(f"training complete, saving model to {model_output_path}")
    trainer.save_model(model_output_path)

  
    
if __name__ == "__main__":
    with open("config/project_config.yaml", "r") as f:
        config = yaml.safe_load(f)

    file_path = config["dataset"]["file_path"]
    model_name = config["training"]["model_name"]
    model_output_path = config["training"]["model_output_path"]

    train_mode(
        model_name=model_name,
        file_path=file_path,
        model_output_path=model_output_path,
    )
