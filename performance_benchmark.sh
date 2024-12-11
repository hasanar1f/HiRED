#!/bin/bash

# Script to run the Python script with different batch sizes and alpha values

# Define the alpha values
token_budget_rates=(1 0.4 0.2)
batch_sizes=(1 2 4 8)
num_runs=1

# Loop over batch sizes from 1 to 16
for batch_size in "${batch_sizes[@]}"
do
    # Loop over each alpha value
    for token_budget_rate in "${token_budget_rates[@]}"
    do
        # Run the Python script with the current batch size and alpha value
        python ./run_HiRED_sys_report_multibatch.py --num_runs "$num_runs" --batch_size "$batch_size" --token_budget_rate "$token_budget_rate"
    done
done

echo "============== end of experimentes ============"