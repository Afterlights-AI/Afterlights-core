import argparse
from retrieval.cl_retrieve import CLRetrieve
from database.connector import QdrantConnector
from database.qdrant_controller import QdrantController

def argparser():
    parser = argparse.ArgumentParser(description="Contrastive Learning Training/Evaluation/Retrieval Script")
    parser.add_argument("--file_path", type=str, required=False, help="Path to the dataset CSV file")
    parser.add_argument("--model_output_path", type=str, required=True, help="Path to save or load the trained model")
    parser.add_argument("--query", type=str, default="", help="Query string for retrieval mode")
    parser.add_argument("--top_k", type=int, default=20, help="Number of top results to retrieve")
    parser.add_argument("--qdrant", action="store_true", help="Use Qdrant for retrieval")
    args = parser.parse_args()
    
    return args

def qdrant_retrieve_mode(model_output_path, file_path, query, collection_name, top_k=20):
    client = QdrantConnector().connect()
    retriever = CLRetrieve(model_name=model_output_path)
    qc = QdrantController(client)
    query_vector = retriever.embed_all_options(query)
    existing_collection = qc.collection_exists(collection_name)
    
    if not existing_collection:
        #print(f"Collection {collection_name} does not exist. Creating a new collection.")
        embeddings = retriever.read_and_embed(file_path)
        #print(f"Embedding {len(embeddings)} items.")
        struct_points = qc.batch_struct_points(
            points=embeddings
        )
        #print(f"struct_points {len(struct_points)} items.")
        
        qc.create_collection(
            name=collection_name, 
            vector_size=len(query_vector)
        )
        
        qc.upsert_points(
            collection=collection_name, 
            points=struct_points)
        
        #print(f"Upserting {len(struct_points)} points to collection {collection_name}.")
        
    search_result = qc.search(
        collection_name=collection_name,
        query_vector=query_vector,
        limit=top_k
    )
    #print(search_result)
    str_output = ""
    for result in search_result:
        text = result.payload['text']
        tt = f"{text}\n"
        str_output += tt
    print(str_output)
    return str_output
        
def one_time_retrieve_mode(model_output_path:str, file_path:str, query:str, top_k=20):
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
            model_output_path=args.model_output_path,
            file_path=args.file_path,
            query=args.query,
            top_k=args.top_k,
            collection_name=collection_name
        )
    else:
        one_time_retrieve_mode(
            model_output_path=args.model_output_path,
            file_path=args.file_path,
            query=args.query,
            top_k=args.top_k,
        )