import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from training.anchor_cl_mining import CLSimplePairMining
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer, losses, InputExample

class CLTraining:
    def __init__(self, model_name):
        self.model = SentenceTransformer(model_name)

    def prepare_inverse_data_for_simcse(self, pairs):
        # Only need anchor and positive for MultiNegativesRankLoss
        return [InputExample(texts=[p['anchor'], p['anchor']]) for p in pairs]
    
   
    def train(self, train_examples, batch_size=8, epochs=7, warmup_steps=10):
        train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=batch_size)
        train_loss = losses.TripletLoss(model=self.model)
        print(f"Training on {len(train_examples)} triplets...")
        self.model.fit(
            train_objectives=[(train_dataloader, train_loss)],
            epochs=epochs,
            warmup_steps=warmup_steps,
            show_progress_bar=True,
            checkpoint_save_total_limit=2,
            checkpoint_path= "checkpoints/model"
        )
        
    def train_simcse(self, train_examples, batch_size=8, epochs=7, warmup_steps=5):
        train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=batch_size)
        train_loss = losses.MultipleNegativesRankingLoss(self.model)
        print(f"Training on {len(train_examples)} pairs with simcse...")
        self.model.fit(
            train_objectives=[(train_dataloader, train_loss)],
            epochs=epochs,
            warmup_steps=warmup_steps,
            show_progress_bar=True,
        )
    
    def train_multineg(self, train_examples, batch_size=8, epochs=7, warmup_steps=5):
        train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=batch_size)
        train_loss = losses.MultipleNegativesRankingLoss(self.model)
        print(f"Training on {len(train_examples)} pairs with MultiNegativesRankLoss...")
        self.model.fit(
            train_objectives=[(train_dataloader, train_loss)],
            epochs=epochs,
            warmup_steps=warmup_steps,
            show_progress_bar=True,
        )
        
    def evaluate(self, anchor, options):
        embeddings = self.model.encode([anchor] + options)
        sims = cosine_similarity([embeddings[0]], embeddings[1:])[0]
        for opt, score in zip(options, sims):
            print(f"{opt} → Similarity: {score:.2f}")
        return sims

    def save_model(self, model_output_path):
        self.model.save(model_output_path)
    
    def embed(self,options):
        embeddings = self.model.encode(options)
        return embeddings
    
    def evaluate_on_dataset(self, all_dataset, summarize_dataset, eval_output_path):
        miner = CLSimplePairMining()
        dialogues = miner.read_pairs_from_file(summarize_dataset)
        summaries = []
        
        dialogue_truth_list = []
        for didx, dialog in enumerate(dialogues):
            dialogue_summary = dialog['dialogue_summary']
            dialogue_list = dialog['dialogue_list']
            dialogue_truth_list.append(dialogue_list)
            summaries.append(dialogue_summary)

        summary_embeddings = self.embed(summaries)
        
        with open(all_dataset, 'r', encoding='utf-8') as f:
            reader = pd.read_csv(f)
            list_of_text = reader['text'].tolist()

        text_embeddings = self.embed(list_of_text)
        similarities = cosine_similarity(summary_embeddings, text_embeddings)
        
        max_indices = np.argmax(similarities, axis=1)
        
        score = 0
        for idx, best in enumerate(max_indices):
            if best in dialogue_truth_list[idx]:
                score += 1
        
        print(f"Score: {score}/{len(max_indices)}")
        with open(eval_output_path, "w") as f:
            f.write(f"Score: {score}/{len(max_indices)}\n")
            for idx, best in enumerate(max_indices):
                f.write(f"Summary: {summaries[idx]}\nBest Match: {list_of_text[best]}\n\n")
                if best in dialogue_truth_list[idx]:
                    f.write("✅ Correct\n")
                else:
                    f.write("❌Incorrect\n")

