import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.retrieve import qdrant_retrieve_mode
import json
import sys
from tqdm import tqdm
import logging

def run_locomo_evaluation():
    """
    Run the Locomo evaluation with the specified parameters.
    """
    # Define the parameters for the evaluation
    file_path="evaluation/eval_dataset/locomo/locomo_conv-26.csv"
    model_output_path="trained_model/locomo_26"
    top_k=3
    query="When did Melanie paint a sunrise?"
    mode="contextual"  # Options: "naive_csv", "contextual"
    collection_name = file_path.split("/")[-1].split(".")[0]+ f"_{mode}_trained_hierarchical"
    qa_path = "evaluation/eval_dataset/locomo/locomo_conv-26_qa.json"
    acc = 0 
    logger = logging.getLogger(collection_name)
    logger.setLevel(logging.INFO)
    log_file = f"logs/{collection_name}.log"
    if not logger.handlers:
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        formatter = logging.Formatter('%(asctime)s - %(message)s')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    with open(qa_path, 'r', encoding='utf-8') as f:
        list_of_qa = json.load(f)
        for qa in tqdm(list_of_qa, desc="Evaluating"):
            query = qa['question']
            # answer = qa['answer']
            # print(f"Answer: {answer}")
            evidence = qa['evidence']
            # Call the qdrant_retrieve_mode function with the parameters
            output = qdrant_retrieve_mode(
                embedding_model_path=model_output_path,
                file_path=file_path,
                query=query,
                top_k=top_k,
                collection_name=collection_name,
                mode=mode
            )
            if 'idx' in output[0]:
                # If the output contains 'idx', it means we are in contextual mode
                evidence_out = [q['idx'] for q in output]
            else:
                evidence_out = output
    
            # Set up logger to log to a file named after the collection
            
            logger.info(f"Question: {query}")
            logger.info(f"Evidence: {evidence}")
            logger.info(f"Evidence out: {evidence_out}")
            logger.info("-" * 50)
            
            if not all(
                any(ev == cand or ev in cand      # exact or substring
                    for cand in evidence_out)
                for ev in evidence):
                
                continue 
            acc += 1
            print("acc increased")
            print(all(ev in evidence_out for ev in evidence))
    logger.info(f"Accuracy: {acc}/{len(list_of_qa)} = {acc/len(list_of_qa)}")
                    
if __name__ == "__main__":
    run_locomo_evaluation()
            