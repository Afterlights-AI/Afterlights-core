file_path="dataset/path/to/datset"
model_output_path="trained_model/path"
top_k=30
query="What do you think about the world and life?"

# Use correct path relative to repo root
python src/retrieve.py \
    --file_path $file_path \
    --model_output_path $model_output_path \
    --query "$query" \
    --top_k $top_k \
    --qdrant