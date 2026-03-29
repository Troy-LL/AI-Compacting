# Kaggle Testing Setup

This document contains the specific code blocks and snippets we will use for testing our LLM Compact project in a Kaggle notebook environment.

## 1. Environment Check

Run this block to verify the GPU and PyTorch version available.

```python
import torch

print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
print(f"PyTorch: {torch.__version__}")
print(f"CUDA: {torch.version.cuda}")
```

## 2. Clone Repository

Clone the specific branch of the project repository into the Kaggle working directory.

```python
import subprocess
import shutil
import os

# Ensure we aren't currently "inside" the folder we are about to delete
os.chdir("/kaggle/working")

repo_path = "/kaggle/working/hailp"
if os.path.exists(repo_path):
    shutil.rmtree(repo_path)

result = subprocess.run([
    "git", "clone",
    "--branch", "Block2",
    "https://github.com/Troy-LL/AI-Compacting.git",
    repo_path
], capture_output=True, text=True)

print(result.stdout)
print(result.stderr)
```

## 3. Setup Path and Verify Files

Add the repository to the Python path and verify that key files exist.

```python
import sys
import os

# Add to path
sys.path.insert(0, "/kaggle/working/hailp")

# Verify key files exist
key_files = [
    "models/hailp_model.py",
    "training/trainer.py",
    "training/data.py",
    "training/device.py",
    "train.py",
]

for f in key_files:
    path = f"/kaggle/working/hailp/{f}"
    status = "✓" if os.path.exists(path) else "✗ MISSING"
    print(f"{status} {f}")
```

## 4. Install Dependencies

Install the required Python packages for training.

```python
import subprocess
import sys

subprocess.run([
    sys.executable, "-m", "pip", "install", "-q",
    "datasets>=2.14.0",
    "tokenizers>=0.15.0", 
    "transformers>=4.35.0",
    "wandb>=0.16.0",
    "psutil>=5.9.0",
    "tqdm>=4.66.0",
    "accelerate>=0.24.0",
])
print("Dependencies installed")
```

## 5. Weights & Biases Login

Authenticate with your wandb account using your API key. By setting this purely via the environment, we avoid Jupyter kernel circular import issues!

```python
import os

# Set the key directly in the environment. 
# Your train.py script will automatically detect and use this when it launches later!
os.environ["WANDB_API_KEY"] = "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"
```

## 6. Kaggle Optimization Patch (Data Loading)

Kaggle provides 2 GPUs and 4 CPU cores, but the codebase defaults to 0 CPU data-workers. This is why you are seeing "pure cpu usage": the CPU is bottlenecking on tokenization, completely starving your GPU resulting in 0 VRAM usage.

*(Note: PyTorch's `DataParallel` was crashing earlier because it fundamentally does not support models that share parameters across layers, which your custom `SharedFFNPool` heavily relies on. We will train efficiently on 1 GPU instead!)*

Run this cell to dynamically patch the downloaded repository to enable multiprocessing data fetchers!

```python
import os

# 1. Patch Data Loader for CPU multi-processing (fixes the "pure CPU" bottleneck)
data_file = "/kaggle/working/hailp/training/data.py"
with open(data_file, "r") as f:
    data_code = f.read()

data_code = data_code.replace("num_workers: int = 0", "num_workers: int = 2")
with open(data_file, "w") as f:
    f.write(data_code)

print("Successfully Patched CPU multiprocessing!")
```

## 7. Configuration Override and Directory Setup

Change to the working directory, load the configuration, and override values for a quick sanity check.

```python
import os
import yaml

os.chdir("/kaggle/working/hailp")

# Override config for sanity check
with open("configs/hailp.yaml", "r") as f:
    config = yaml.safe_load(f)

# Temporarily override for quick test
config["total_steps"] = 100
config["batch_size"] = 128  # Increased to feed 2 GPUs
config["seq_len"] = 256
config["device"] = "cuda"
config["checkpoint_dir"] = "/kaggle/working/checkpoints"
config["checkpoint_every"] = 50
config["log_every"] = 10

os.makedirs(config["checkpoint_dir"], exist_ok=True)

print("Config:")
for k, v in config.items():
    print(f"  {k}: {v}")
```

## 8. Run Training (Multi-GPU)

Launch the training script using HuggingFace Accelerate so that it correctly spans across both your T4 GPUs!

```python
import sys
import os

os.chdir("/kaggle/working/hailp")

# Launch the bespoke multi-GPU script we wrote for Kaggle!
!accelerate launch --multi_gpu --num_processes 2 train_multi.py
```
