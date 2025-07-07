from fastapi import FastAPI, Query
from pydantic import BaseModel
from retrieve import qdrant_retrieve_mode, one_time_retrieve_mode

app = FastAPI()

class RetrieveRequest(BaseModel):
    model_output_path: str
    file_path: str
    query: str
    mode: str = "naive_csv"  # Default mode
    top_k: int = 20

@app.post("/retrieve/qdrant")
def retrieve_qdrant(req: RetrieveRequest):
    collection_name = req.file_path.split("/")[-1].split(".")[0]
    result = qdrant_retrieve_mode(
        model_output_path=req.model_output_path,
        file_path=req.file_path,
        query=req.query,
        top_k=req.top_k,
        collection_name=collection_name,
        mode=req.mode,
    )
    return {"result": result}

@app.post("/retrieve/one_time")
def retrieve_one_time(req: RetrieveRequest):
    result = one_time_retrieve_mode(
        model_output_path=req.model_output_path,
        file_path=req.file_path,
        query=req.query,
        top_k=req.top_k
    )
    return {"result": result}