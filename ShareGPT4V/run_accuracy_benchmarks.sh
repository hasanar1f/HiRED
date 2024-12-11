#!/bin/bash

# Define the list of benchmarks
benchmarks=("scienceqa_img" "vqav2_val" "textvqa_val" "docvqa_val" "chartqa" "ocrbench" "mme" "pope")

for task in "${benchmarks[@]}"; do
    echo "Running task: $task"
    python3 -m accelerate.commands.launch \
        --num_processes=1 \
        -m lmms_eval \
        --model sharegpt \
        --model_args pretrained="Lin-Chen/ShareGPT4V-7B" \
        --tasks "$task" \
        --batch_size 1 \
        --log_samples \
        --log_samples_suffix sharegpt \
        --output_path ./logs_40/
    echo "Completed task: $task"
done

echo "All tasks completed."