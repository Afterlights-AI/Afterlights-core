import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

class CLRetrieve:
    """
    Class to retrieve
    """
    def __init__(self, model_name):
        self.model = SentenceTransformer(model_name)

    def read_and_embed(self, all_dataset, add_talker=True, text_embedding_only=False):
        with open(all_dataset, 'r', encoding='utf-8') as f:
            reader = pd.read_csv(f)
            list_of_text = reader['text'].tolist()
            talker = reader['talker'].tolist()
            time = reader['time'].tolist() if 'time' in reader.columns else [''] * len(list_of_text)
            if add_talker:
                talker_text = [f"{spk}: {txt}" for spk, txt in zip(talker, list_of_text)]
                text_embeddings = self.embed_all_options(talker_text)
            else:
                text_embeddings = self.embed_all_options(list_of_text)
            if text_embedding_only:
                return text_embeddings
            results = []
            for i in range(len(list_of_text)):
                
                if add_talker:
                    _dict = {
                        'text': talker_text[i],
                        'talker': talker[i],
                        'time': time[i],
                        'embedding': text_embeddings[i]
                    }
                else:
                    _dict = {
                        'text': list_of_text[i],
                        'talker': talker[i],
                        'time': time[i],
                        'embedding': text_embeddings[i]
                }
                results.append(_dict)
        return results
    
    def retrieve(self, all_dataset, query, text_embeddings, top_k=20):
        # Embed the query
        query_embedding = self.embed_all_options([query])
        with open(all_dataset, 'r', encoding='utf-8') as f:
            reader = pd.read_csv(f)
        #print(len(query_embedding), len(text_embeddings))
        similarities = cosine_similarity(query_embedding, text_embeddings)
        max_indices = np.argsort(similarities, axis=1)[:, -top_k:]
        # Print top 10 results with talker+text
        top_indices = max_indices[0][::-1]  # reverse to get highest similarity first
        
        str_output = ""
        for idx in top_indices:
            row = reader.iloc[idx]
            talker = row['talker'] if 'talker' in row else ''
            text = row['text']
            str_output += f"{talker}: {text}\n"
            
        return str_output
    
    def embed_all_options(self, options):
        embeddings = self.model.encode(options)
        return embeddings

if __name__ == "__main__":
    model_name = "trained_model/nazha_model_denoising"
    model = SentenceTransformer(model_name)
    options = ["test1"]
    embeddings = model.encode(options)
    print(embeddings)
    
    