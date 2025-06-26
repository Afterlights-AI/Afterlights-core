import pandas as pd
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
class RStyle:
    def __init__(self, model_name):
        self.model = SentenceTransformer(model_name)
    def read_file(self, all_dataset):
        with open(all_dataset, 'r', encoding='utf-8') as f:
            reader = pd.read_csv(f)
            list_of_text = reader['text'].tolist()
            talker = reader['talker'].tolist()
            time = reader['time'].tolist()

            results = []
            for i in range(len(list_of_text)):
                if talker[i].lower() == "sky":

                    _dict = {
                        'text': list_of_text[i],
                        'talker': talker[i],
                        'time': time[i],
                    }
                    results.append(_dict)

        return results
    
    def embed(self, data):
        text_data = [data['text'] for data in data]
        embeddings = self.model.encode(text_data)
        avg_embedding = np.mean(embeddings, axis=0)
        return embeddings, avg_embedding
    
    def remove_duplicates(self, data):
        seen = set()
        unique_data = []
        for item in data:
            text = item['text'].strip()
            if text not in seen:
                seen.add(text)
                unique_data.append(item)
        return unique_data
    def calculate_similarity(self, data, embeddings, avg_embedding):
        topk = 20
        similarities = cosine_similarity([avg_embedding], embeddings)
        max_indices = np.argsort(similarities, axis=1)[:, :]
        top_indices = max_indices[0][::-1]  # reverse to get highest similarity first
        bot_indices = max_indices[0]
        most_sim_dialogues = [data[i] for i in top_indices[:topk]][::-1] # biggest to smallest   
        most_sim_dialogues = [data[i] for i in top_indices] # biggest to smallest   
        most_sim_dialogues = self.remove_duplicates(most_sim_dialogues)
        least_sim_dialogues = [data[i] for i in bot_indices[:topk]]# smallest to biggest   
        similarities = similarities[0]
        most_sim_scores = [similarities[i] for i in top_indices]
        least_sim_scores = [similarities[i] for i in bot_indices]
        
        from pprint import pprint
        
        pprint(most_sim_dialogues)
        return most_sim_dialogues

if __name__ == "__main__":
    rstyle = RStyle("trained_model/model")
    data = rstyle.read_file("dataset/dataset.csv")
    embeddings, avg_embedding = rstyle.embed(data)
    similarities = rstyle.calculate_similarity(data, embeddings, avg_embedding)
    