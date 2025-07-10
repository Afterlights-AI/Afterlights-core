from retrieval.base import Indexer, Retriever
from model_management.embedding_model_controller import EmbeddingModelController
from qdrant_client.http.models import PointStruct
from database.connector import QdrantConnector
from database.qdrant_controller import QdrantController
import pandas as pd
from pydantic import BaseModel
from typing import List, Any
import os
class StructuredDialogue(BaseModel):
    text: str
    talker: str
    time: Any
    source: Any = None
    embedding: List[float]
    
class StructuredQdrantController(QdrantController):
    def __init__(self, client):
        super().__init__(client)

    def batch_struct_points(self, points: list[StructuredDialogue]):
        """Convert a list of ContextualKeyValuePair to PointStruct."""
        return [
            PointStruct(
                id=i,
                vector=point.embedding,
                payload={
                    "text": point.text, 
                    "talker": point.talker,
                    "time": point.time,
                    "source": point.source}
            )
            for i, point in enumerate(points)
        ]

class StructuredCSVIndexing(Indexer):  
    def __init__(self):
        pass
    
    def index(self, *, embedding_model_path, file_path, collection_name):
        if not file_path.endswith('.csv'):
            raise ValueError("The class must pass a CSV file.")
        
        client = QdrantConnector().connect()
        qc = StructuredQdrantController(client)
        existing_collection = qc.collection_exists(collection_name)
        
        if existing_collection:
            return
        embeddings = self.read_and_embed(embedding_model_path, file_path)
        struct_points = qc.batch_struct_points(
            points=embeddings
        )
        qc.create_collection(
            name=collection_name, 
            vector_size=len(embeddings[0].embedding)
        )
        
        qc.upsert_points(
            collection=collection_name, 
            points=struct_points)
        
            

    def read_and_embed(self, embedding_model_path, all_dataset, add_talker=True, text_embedding_only=False):
        embedding_model = EmbeddingModelController(model_name=embedding_model_path)
        with open(all_dataset, 'r', encoding='utf-8') as f:
            reader = pd.read_csv(f)
            list_of_text = reader['text'].tolist()
            talker = reader['talker'].tolist()
            time = reader['time'].tolist() if 'time' in reader.columns else [''] * len(list_of_text)
            source = reader['source'].tolist() if 'source' in reader.columns else [''] * len(list_of_text)
            
            if add_talker:
                talker_text = [f"{spk}: {txt}" for spk, txt in zip(talker, list_of_text)]
                text_embeddings = embedding_model.embed(talker_text)
            else:
                text_embeddings = embedding_model.embed(list_of_text)
            if text_embedding_only:
                return text_embeddings
            results = []
            for i in range(len(list_of_text)):
                if add_talker:
                    obj = StructuredDialogue(
                        text=talker_text[i],
                        talker=talker[i],
                        time=time[i],
                        source=source[i],
                        embedding=text_embeddings[i]
                    )
                   
                else:
                    obj = StructuredDialogue(
                        text=list_of_text[i],
                        talker=talker[i],
                        time=time[i],
                        source=source[i],
                        embedding=text_embeddings[i]
                    )
                   
                results.append(obj)
        return results
    

class StructuredCSVRetrieval(Retriever):
    def retrieve(self, collection_name, embedding_model_path, query, top_k=20):
        client = QdrantConnector().connect()
        qc = StructuredQdrantController(client)
        embedding_model = EmbeddingModelController(model_name=embedding_model_path)
        query_vector = embedding_model.embed(query)
        search_result = qc.search(
            collection_name=collection_name,
            query_vector=query_vector,
            limit=top_k
        )
        
        str_output = ""
        list_of_content = []
        for result in search_result:
            text = result.payload['text']
            talker = result.payload.get('talker', '')
            time = result.payload.get('time', '')
            tt = f"[{time}] {talker}: {text}\n"
            str_output += tt
            list_of_content.append({
                "idx": result.payload.get('source', ''),
                "text": tt
            })
        return list_of_content

if __name__ == "__main__":
    # Example usage
    model_path = "../trained_model/nazha_model"
    file_path = "../examples/cn_example_nazha_dataset.csv"
    collection_name = "naive_collection_test"
    
    indexer = StructuredCSVIndexing()
    indexer.index(
        embedding_model_path=model_path, 
        file_path=file_path, 
        collection_name=collection_name
    )    
    retriever = StructuredCSVRetrieval()
    query = "哪吒是混元珠吗？"
    
    result = retriever.retrieve(collection_name=collection_name, embedding_model_path=model_path, query=query, top_k=10)
    
    print("Retrieved Contextual Information:")
    print(result)
