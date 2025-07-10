file_path="evaluation/eval_dataset/locomo/locomo_conv-26.csv"
model_output_path="trained_model/all-minilm-v2"
top_k=3
query="When did Melanie paint a sunrise?"
mode="naive_csv"
# Use correct path relative to repo root
python src/retrieve.py \
    --file_path $file_path \
    --model_output_path $model_output_path \
    --query "$query" \
    --top_k $top_k \
    --mode $mode \
    --qdrant