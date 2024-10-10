# 2024 FLAME AI Challenge  
[Chinese Readme](./README_zh.md)
## configuration environment
```bash
git clone https://github.com/lizhuoq/FLAME_AI.git
cd FLAME_AI
conda env create -f environment.yml
conda activate flame
```
## Interface for testing
The experiments in this repository were run on NVIDIA Tesla V100-SXM2 32GB Gpus
### Step 1: Download the weight file (requires about 160G of storage space, requires networking)
```python
from for_kaggle import download_checkpoints
download_checkpoints('checkpoints')
```
### Step 2: Inference
Inference requires at least 32 G of memory
```python
from for_kaggle import submit_api
# example
theta = np.fromfile('dataset/dataset/test/theta_K_id098830.dat', dtype='<f4').reshape(5, 113, 32) 
ustar = np.fromfile('dataset/dataset/test/ustar_ms-1_id098830.dat', dtype='<f4').reshape(5, 113, 32)
xi = np.fromfile('dataset/dataset/test/xi_id098830.dat', dtype='<f4').reshape(5, 113, 32)
submit_api(theta, ustar, xi, 40, 'pred.npy', 'cpu')
```
- The parameters of `submit_api`: `theta`, `ustar`, and `xi`, are required to have a shape of (5, 113, 32). The parameter `pred_len` can be set to any arbitrary positive integer, `save_path` is the path where the prediction file will be saved, and `device` can only be set to either `gpu` or `cpu`. The return value has a shape of (pred_len, 113, 32). (Note: This function does not require the `alpha` and `u` parameters as inputs,  and `submit_api` does not support batch operation, so only one time series can be tested at a time.)
```
def submit_api(
        theta: np.ndarray, 
        ustar: np.ndarray, 
        xi: np.ndarray, 
        pred_len: int, 
        save_path: str, 
        device: str = 'cpu', 
) -> np.ndarray:
```
- Read prediction file
```python
import numpy as np
pred = np.load(save_path) # shape: pred_len, 113, 32
```
### contact information
m15004059308@163.com