file_path="dataset/path"
model_output_path="trained_model/path"
top_k=3
query="query"
mode="contextual" # Options: "naive_csv", "contextual"
python src/retrieve.py \
    --file_path $file_path \
    --model_output_path $model_output_path \
    --query "$query" \
    --top_k $top_k \
    --mode $mode \
    --qdrant