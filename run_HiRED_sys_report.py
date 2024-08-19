from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
import torch
from PIL import Image
import requests
import time
import numpy as np
from helper import *

# Initialize processor and model
model_id = "llava-hf/llava-v1.6-vicuna-7b-hf"

quantization = False

processor = LlavaNextProcessor.from_pretrained(model_id)
model = LlavaNextForConditionalGeneration.from_pretrained(
    model_id, 
    torch_dtype=torch.float16, 
    low_cpu_mem_usage=True, 
    load_in_4bit=quantization,
)

if quantization is False: # hot fix for: .to` is not supported for `4-bit` or `8-bit` bitsandbytes models. 
    # Please use the model as it is, since the model has already been set to the correct devices and casted to the correct `dtype`.
    model = model.to("cuda:0")

# ================ HiRED config ================
model.config.hired_config = {
    "token_budget_rate": 0.2, # 1.0 for full execution
    "alpha": 0.5,
}

print(f"Running inference on model: {model_id} for HiRED config: {model.config.hired_config}")
num_runs = 3
throughputs = []
latencies = []
time_to_first_tokens = []


print(f"Running {num_runs} inferences...")
for _ in range(num_runs):
    image = get_random_image(h=600, w=600)
    prompt = "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: <image> \n What is this image about? \n ASSISTANT:" 

    # Reset GPU memory statistics
    torch.cuda.reset_peak_memory_stats("cuda:0")

    # start counting time
    start_time = time.time()

    inputs = processor(prompt, image, return_tensors="pt").to("cuda:0")
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            do_sample=False,
            max_new_tokens=15,
            use_cache=True,
            return_dict_in_generate=True,
            )
    
    # length of newly generated tokens
    output_text = processor.decode(output['sequences'][0], skip_special_tokens=False)
    
    end_time = time.time()
    time_to_first_token = model.prefill_time - start_time
    total_generation_time = end_time - start_time
    num_tokens_generated = len(output['sequences'][0]) - len(inputs["input_ids"][0])
    throughput = num_tokens_generated / total_generation_time # tokens per second

    throughputs.append(throughput)
    time_to_first_tokens.append(time_to_first_token)


avg_gen_throughput = np.mean(throughputs)
avg_time_to_first_token = np.mean(time_to_first_tokens)
peak_memory_used = torch.cuda.max_memory_allocated("cuda:0") / (1024 * 1024 * 1024)  # Convert to GB

# Print results
print(f"Throughput: {avg_gen_throughput} tokens/s")
print(f"Time to first token: {avg_time_to_first_token} s")
print(f"Peak GPU memory used: {peak_memory_used} GB")