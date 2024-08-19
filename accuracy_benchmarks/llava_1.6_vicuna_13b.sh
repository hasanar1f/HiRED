#!/bin/bash

model_id="llava-hf/llava-v1.6-vicuna-13b-hf"
model_short_name="1.6-vicuna-13b"

benchmarks=("scienceqa_img" "vqav2_val" "textvqa_val" "docvqa_val" "chartqa" "ocrbench" "mme" "pope")
token_budget_rate=0.2
alpha=0.5
limit=250

# Refactored code with the specified parameters

for task in "${benchmarks[@]}"; do
    output_path="./logs/${model_short_name}/budget_${token_budget_rate}_alpha_${alpha}/"
    echo "Running benchmark for $task with token_budget_rate: $token_budget_rate, alpha: $alpha"
    accelerate launch --num_processes=1 -m lmms_eval --model llava_hf \
        --model_args pretrained=$model_id,device_map=cuda,token_budget_rate=$token_budget_rate,alpha=$alpha \
        --tasks "$task" \
        # --limit "$limit" \
        --batch_size 1 --log_samples --output_path "$output_path"
done

echo "All benchmarks are completed!"

