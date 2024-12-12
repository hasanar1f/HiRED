# ShareGPT4V [ECCV 2024] with HiRED

### Step 1: Clone the Repository

1. Clone the **ShareGPT4V** repository:

```bash
git clone https://github.com/ShareGPT4Omni/ShareGPT4V.git
cd ShareGPT4V
```

2. Replace the `./share4v/model/multimodal_encoder/clip_encoder.py` with the [clip_encoder.py](./clip_encoder.py) that contains the HiRED implementation (as well as PruMerge as a baseline).


### Step 2: Install and setup ShareGPT4V

1. Follow the installation instructions provided in the [ShareGPT4V repository](https://github.com/ShareGPT4Omni/ShareGPT4V).

2. Copy `sharegpt.py` from [sharegpt.py](ShareGPT4V/sharegpt.py) and paste it into the `lmms-eval/lmms_eval/models/` directory.

3. Modify `__init__.py`:  
   Open the file `lmms-eval/lmms_eval/models/__init__.py` and add the following key-value pair to the `AVAILABLE_MODELS` dictionary:

   ```python
   "sharegpt": "Sharegpt",
   ```

### Step 3: Run ShareGPT4V

1. Run inference using ShareGPT4V:  
   Execute the script located at `run_inference_of_ShareGPT4V.py` using the following command:

   ```bash
   python ShareGPT4V/run_inference_of_ShareGPT4V.py
    ```

2. Run the benchmark script:  
   Execute the script located at `run_accuracy_benchmarks.sh` using the following command:

   ```bash
   bash ShareGPT4V/run_accuracy_benchmarks.sh
    ```

For more details, refer to the [ShareGPT4V documentation](https://github.com/ShareGPT4Omni/ShareGPT4V).