import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))
import pytest
from fastapi.testclient import TestClient
from src.api import app

client = TestClient(app)

def test_train_model_endpoint():
    payload = {
        "dataset_path": "tests/mock_data/en_example_ironman_dataset.csv",
        "model_output_path": "trained_model/test_model",
        "model_name": "sentence-transformers/all-MiniLM-L6-v2"
    }
    response = client.post("/train/model", json=payload)
    assert response.status_code == 200
    assert response.json() == {"status": "training started"}

def test_retrieve_qdrant_endpoint():
    payload = {
        "model_output_path": "trained_model/iron_man_model",
        "file_path": "tests/mock_data/en_example_ironman_dataset.csv",
        "query": "What is Tony Stark's suit?",
        "mode": "naive_csv",
        "top_k": 2
    }
    response = client.post("/retrieve/qdrant", json=payload)
    print("try",response.json())
    assert response.status_code == 200
    assert "result" in response.json()

def test_retrieve_one_time_endpoint():
    payload = {
        "model_output_path": "trained_model/iron_man_model",
        "file_path": "tests/mock_data/en_example_ironman_dataset.csv",
        "query": "What is Tony Stark's suit?",
        "mode": "naive_csv",
        "top_k": 2
    }
    response = client.post("/retrieve/one_time", json=payload)
    assert response.status_code == 200
    assert "result" in response.json()
