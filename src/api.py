from fastapi import FastAPI, Query, BackgroundTasks
from pydantic import BaseModel
from retrieve import qdrant_retrieve_mode, one_time_retrieve_mode
from train import train_mode

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

class TrainRequest(BaseModel):
    dataset_path: str
    model_output_path: str
    collection_name: str

@app.post("/train/model")
def train_model(req: TrainRequest, background_tasks: BackgroundTasks):
    # Start training in the background
    background_tasks.add_task(
        train_mode,
        model_name=req.collection_name,  # You may want to use a different field for model_name if needed
        file_path=req.dataset_path,
        model_output_path=req.model_output_path,
    )
    return {"status": "training started"}

