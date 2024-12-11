import torch
import time
import numpy as np
import argparse
from PIL import Image
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
from helper import get_random_image
import csv


def run_inference(batch_size, num_runs, alpha, token_budget_rate, quantization=False):
    # Initialize processor and model
    model_id = "llava-hf/llava-v1.6-vicuna-7b-hf"
    
    processor = LlavaNextProcessor.from_pretrained(model_id)
    processor.tokenizer.padding_side = "left"  # needed for batch generation

    model = LlavaNextForConditionalGeneration.from_pretrained(
        model_id, 
        torch_dtype=torch.float16, 
        low_cpu_mem_usage=True, 
        load_in_4bit=quantization,
        attn_implementation="sdpa",  # eager or sdpa or flash_attention_2
    )

    if quantization is False:  # Hot fix for: .to` is not supported for `4-bit` or `8-bit` bitsandbytes models
        model = model.to("cuda:0")

    # HiRED config
    model.config.hired_config = {
        "token_budget_rate": token_budget_rate,
        "alpha": alpha,
    }

    print(f"Running inference on model: {model_id} with HiRED config: {model.config.hired_config}")
    throughputs = []
    latencies = []
    time_to_first_tokens = []

    print(f"Running {num_runs} inferences with batch size {batch_size}...")

    for _ in range(num_runs):
        images = [get_random_image(h=1000, w=1000)] * batch_size
        prompt = "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: <image> \n What is this image about? \n ASSISTANT:"
        prompts = [prompt] * batch_size

        start_time = time.time()

        # Process inputs for the entire batch
        inputs = processor(prompts, images, return_tensors="pt", padding=True).to("cuda:0")

        # Perform batch inference
        with torch.inference_mode():
            output = model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=100,
                use_cache=True,
                return_dict_in_generate=True,
            )

        end_time = time.time()

        generated_sequences = output['sequences']
        num_tokens_generated = sum([len(seq) - len(inputs["input_ids"][0]) for seq in generated_sequences])

        total_generation_time = end_time - start_time
        throughput = num_tokens_generated / total_generation_time  # tokens per second
        throughputs.append(throughput)

        time_to_first_token = model.prefill_time - start_time
        time_to_first_tokens.append(time_to_first_token)

    # Average metrics
    avg_gen_throughput = np.mean(throughputs)
    avg_time_to_first_token = np.mean(time_to_first_tokens)
    peak_memory_used = torch.cuda.max_memory_allocated("cuda:0") / (1024 * 1024 * 1024)  # Convert to GB

    print(f"Throughput: {avg_gen_throughput} tokens/s")
    print(f"Time to first token: {avg_time_to_first_token} s")
    print(f"Peak GPU memory used: {peak_memory_used} GB")

    # Prepare data to append
    data_row = [args.batch_size, args.token_budget_rate, avg_gen_throughput, avg_time_to_first_token, peak_memory_used]
    with open(args.log_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        if file.tell() == 0:  # Check if the file is empty
            writer.writerow(['batch_size', 'budget', 'throughput', 'ttft', 'memory'])
        writer.writerow(data_row)

if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Run inference with configurable parameters.')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for inference')
    parser.add_argument('--num_runs', type=int, default=1, help='Number of runs for the inference')
    parser.add_argument('--alpha', type=float, default=0.5, help='Alpha value for HiRED config')
    parser.add_argument('--token_budget_rate', type=float, default=0.2, help='Token budget rate for HiRED config')
    parser.add_argument('--quantization', action='store_true', help='Use quantization (4-bit) if specified')
    parser.add_argument('--log_file', type=str, default='performance.csv', help='CSV file to save performance metrics')

    args = parser.parse_args()

    # Call the function with parsed arguments
    run_inference(args.batch_size, args.num_runs, args.alpha, args.token_budget_rate, args.quantization)