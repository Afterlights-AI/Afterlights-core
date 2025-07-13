import os
from qdrant_client.http.models import PointStruct, Distance
from retrieval.model_calling import ModelContext
from database.connector import QdrantConnector
from database.qdrant_controller import QdrantController
from pydantic import BaseModel, Field
from model_management.embedding_model_controller import EmbeddingModelController
from retrieval.chunking_strategies.neighbour_sim import NeibourSimilarityChunker
from tqdm import tqdm  # Add this import at the top if not already present
from retrieval.base import Indexer
from typing import Any
import csv
class ContextualKeyValuePair(BaseModel):
    key: str = Field(..., description="The contextual summary of the information.")
    value: str = Field(..., description="The actual content for retrieval.")
    embedding: list[float] = Field(..., description="The embedding vector for the key.")
class ContextualQdrantController(QdrantController):
    def __init__(self, client):
        super().__init__(client)

    def batch_struct_points(self, points: list[ContextualKeyValuePair]):
        """Convert a list of ContextualKeyValuePair to PointStruct."""
        return [
            PointStruct(
                id=i,
                vector=point.embedding,
                payload={"key": point.key, "value": point.value}
            )
            for i, point in enumerate(points)
        ]
        

class ContextualIndexing(Indexer):
    def __init__(self):
        pass
      

    def index(self, * , embedding_model_path, file_path, collection_name, hierachical_matching=False):
        """ Logic
        We perform a key, value based retrieval.
        The key is the contextual summary of the information + the actual content
        The value is the actual content for retrieval
        
        To achieve this, we will:
        1. Read the dataset and chunk the files into smaller parts 
            - Since we are dealing with dialogues, it make sense to chunk by a section of dialogue
                _> This can be dates, chunks separated by certain delimiter, or just number of words
        2. Then call summarization on each chunk to get the key
        3. Store the key and value in a database (Qdrant)
        """
        client = QdrantConnector().connect()
        cqc = ContextualQdrantController(client)
        if cqc.collection_exists(collection_name):
            return
        # 1. Read and chunk the file
        with open(file_path, 'r', encoding='utf-8') as f:
            if file_path.endswith('.csv'):
                next(f)  # Skip header line for CSV files
            content = f.read()

        # Example: simple chunking by paragraphs (customize as needed)
        chunking_strategy = NeibourSimilarityChunker(embedding_model_name=embedding_model_path)
        chunks = chunking_strategy.chunk_text(content, max_tokens=150)

        # 2. Summarize each chunk to create keys
        model = ModelContext(model_name="gpt-4.1-nano")
        
        def succinct_contexts(chunks, model):
            keys = []

            for chunk in tqdm(chunks, desc="Generating succinct contexts"):
                model.clear_history()
                model.add_user_message(
                    messages=f"""Here is the chunk we want to situate within the whole document 
                        <chunk> 
                        {chunk}
                        </chunk> 
                        Please give a short succinct context using the language of the chunk's to situate this chunk within the overall document for the purposes of improving search retrieval of the chunk. Answer only with the succinct context and nothing else.
                    """)
                summary = model.call_model()
                keys.append(summary)
            return keys
        
        keys = succinct_contexts(chunks, model)
        
        
        # 3. Embed the keys and chunks
        if hierachical_matching:
            # restructure values based on similarity
            contextual_keys, contextual_key_index = chunking_strategy.chunk_by_similarity(chunks,  similarity_threshold=0.7)
            contextual_values = []
            for indexes in contextual_key_index:
                contextual_values.append(" ".join([chunks[i] for i in indexes]))
            keys = succinct_contexts(contextual_values, model)
            chunks = contextual_values
            
        embedder = EmbeddingModelController(model_name=embedding_model_path)
        key_embeddings = embedder.embed(keys)

        assert len(key_embeddings) == len(chunks), "Key embeddings and chunks must have the same length."
        contextual_pairs = []
        for k, v, embedding in zip(keys, chunks, key_embeddings):
                
            contextual_pairs.append(
                ContextualKeyValuePair(
                key=k,
                value=v,
                embedding=embedding.tolist()
            ))
        
        # 4. Store in Qdrant
        
        struct_points = cqc.batch_struct_points(contextual_pairs)
        cqc.create_collection(
            name=collection_name, 
            vector_size=len(key_embeddings[0])
        )
        cqc.upsert_points(
            collection=collection_name,
            points=struct_points)

class ContextualRetrieval:

    def retrieve(self, collection_name, embedding_model_path, query, top_k=20):
        """Retrieve contextual information based on a query."""

        client = QdrantConnector().connect()
        qc = ContextualQdrantController(client)
        embedding_model = EmbeddingModelController(model_name=embedding_model_path)
        query_vector = embedding_model.embed(query)
        
        search_result = qc.search(
            collection_name=collection_name,
            query_vector=query_vector,
            limit=top_k
        )
        
        str_output = []
        for result in search_result:
            # key = result.payload['key']
            value = result.payload['value']
            str_output.append(f"{value}\n")
        
        return str_output

        


if __name__ == "__main__":
    # Example usage
    model_path = "../trained_model/nazha_model"
    file_path = "../examples/cn_example_nazha_dataset.csv"
    collection_name = "contextual_collection_test"
    
    # indexer = ContextualIndexing()
    # indexer.index(
    #     embedding_model_path=model_path, 
    #     file_path=file_path, 
    #     collection_name=collection_name, 
    #     hierachical_matching=True)
    
    retriever = ContextualRetrieval()
    query = "哪吒是混元珠吗？"
    
    result = retriever.retrieve(collection_name=collection_name, embedding_model_path=model_path, query=query, top_k=5)
    
    print("Retrieved Contextual Information:")
    print(result)