import os
import argparse
print("pwd :::",os.getcwd())
from retrieval.cl_retrieve import CLRetrieve
from file_util import resolve_model_path

def argparser():
    parser = argparse.ArgumentParser(description="Contrastive Learning Training/Evaluation/Retrieval Script")
    parser.add_argument("--file_path", type=str, required=False, help="Path to the dataset CSV file")
    parser.add_argument("--model_output_path", type=str, required=True, help="Path to save or load the trained model")
    parser.add_argument("--query", type=str, default="", help="Query string for retrieval mode")
    parser.add_argument("--top_k", type=int, default=20, help="Number of top results to retrieve")
    parser.add_argument("--qdrant", action="store_true", help="Use Qdrant for retrieval")
    parser.add_argument("--mode", "-m", type=str, choices=["naive_csv", "contextual"], default="naive_csv", help="Mode of operation: naive_csv or contextual")
    args = parser.parse_args()
    
    return args

def qdrant_retrieve_mode(embedding_model_path, file_path, query, collection_name, mode, top_k=20):
    embedding_model_path = resolve_model_path(embedding_model_path)
    file_path = resolve_model_path(file_path)
    match mode:
        case "naive_csv":
            from retrieval.structured_csv_retrieve import StructuredCSVRetrieval, StructuredCSVIndexing
            indexer = StructuredCSVIndexing()
            retriever = StructuredCSVRetrieval()
            
        case "contextual":
            from retrieval.contextual_retrieve import ContextualRetrieval, ContextualIndexing
            indexer = ContextualIndexing()
            retriever = ContextualRetrieval()
            
    indexer.index(
        embedding_model_path=embedding_model_path, 
        file_path=file_path, 
        collection_name=collection_name
    )
    
    output = retriever.retrieve(
        collection_name=collection_name,
        embedding_model_path=embedding_model_path,
        query=query,
        top_k=top_k
    )
        
    return output
        
def one_time_retrieve_mode(model_output_path:str, file_path:str, query:str, top_k=20):
    model_output_path = resolve_model_path(model_output_path)
    file_path = resolve_model_path(file_path)
    retriever = CLRetrieve(model_name=model_output_path)
    text_embeddings = retriever.read_and_embed(
        file_path, 
        add_talker=True, 
        text_embedding_only=True
    )
    result = retriever.retrieve(file_path, query, text_embeddings, top_k)
    return result
       

if __name__ == "__main__":
    args = argparser()
    if args.qdrant:
        collection_name = args.file_path.split("/")[-1].split(".")[0]
        qdrant_retrieve_mode(
            embedding_model_path=args.model_output_path,
            file_path=args.file_path,
            query=args.query,
            top_k=args.top_k,
            collection_name=collection_name,
            mode=args.mode
        )
    else:
        one_time_retrieve_mode(
            model_output_path=args.model_output_path,
            file_path=args.file_path,
            query=args.query,
            top_k=args.top_k,
        )