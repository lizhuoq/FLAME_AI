# 2024 FLAME AI Challenge  
[Chinese Readme](README_zh.md)
## Environment Setup
```bash
git clone https://github.com/lizhuoq/FLAME_AI.git
cd FLAME_AI
conda env create -f environment.yml
conda activate flame
```
Operating System: Linux
## Interface for Testing
### Step 1: Download the weight files (requires about 160G of storage space and internet access)
```python
from for_kaggle import download_checkpoints
download_checkpoints('checkpoints')
```
### Step 1: Retraining (no internet access required)
#### Ensure the model is autoregressive  
This repository optimizes three variables `xi`, `theta`, and `ustar` separately (Reason: Multi-target optimization increases the error for all three variables based on our tests).  
#### Cross-validation and Bagging Strategy  
This repository uses 9-fold cross-validation, where each time, data from one train ID is left out for validation, while the remaining data is used for training. Therefore, there are 9 weight files for each variable, and bagging is applied to results from these 9 weights. In total, there are 27 weight files, requiring at least 160G of storage. To reproduce my results, strictly follow this strategy.  
#### GPU and Memory  
The experiments in this repository were run on an NVIDIA Tesla V100-SXM2 32GB GPU. Ensure that your GPU has 32G of VRAM and that you have at least 32G of system memory.  
#### Data Location
Place the data under `./dataset`  
- Data storage structure:  
```
├─ dataset  
│  ├─ dataset  
│  │  ├─ test  
│  │  │  ├─ theta_K_id098830.dat    
│  │  │  ├─ ......  
│  │  │  ├─ ustar_ms-1_id098830.dat  
│  │  │  ├─ ......  
│  │  │  ├─ xi_id098830.dat  
│  │  │  ├─ ......  
│  │  ├─ test.csv  
│  │  ├─ train  
│  │  │  ├─ theta_K_id016525.dat  
│  │  │  ├─ ......  
│  │  │  ├─ ustar_ms-1_id016525.dat  
│  │  │  ├─ ......  
│  │  │  ├─ xi_id016525.dat  
│  │  │  ├─ ......  
│  │  └─ train.csv  
│  ├─ feature_scaler.joblib  
│  └─ linear_model.csv  
```  
#### Modify Absolute Paths  
To avoid unexpected issues, absolute paths are set in this repository. To reproduce the results, modify all absolute paths in the following two files to match your environment:
- `./data_provider/data_loader.py`
```python
# Lines 14-15
self.meta_path = '/data/home/scv7343/run/flame/Time-Series-Library-main/dataset/dataset/train.csv'
self.data_dir = '/data/home/scv7343/run/flame/Time-Series-Library-main/dataset/dataset/train'
# Line 31
self.scaler: dict[str, StandardScaler] = load('/data/home/scv7343/run/flame/Time-Series-Library-main/dataset/feature_scaler.joblib')
```
- `./exp/exp_flame.py`
```python
# Lines 196-198
df_meta = pd.read_csv('/data/home/scv7343/run/flame/Time-Series-Library-main/dataset/dataset/test.csv')
dir_path = '/data/home/scv7343/run/flame/Time-Series-Library-main/dataset/dataset/test'
feature_scaler: dict[str, StandardScaler] = load('/data/home/scv7343/run/flame/Time-Series-Library-main/dataset/feature_scaler.joblib')
```
#### Training Scripts
- Ensure you have 160G of storage before running the scripts, or the weights won't be saved.
- Make sure your system has at least 32G of memory, and the GPU has at least 32G of VRAM (Do not adjust hyperparameters if these conditions are not met, as it will prevent reproducibility).
```bash
bash ./scripts/for_kaggle/UNet_small_xi_train.sh
bash ./scripts/for_kaggle/UNet_small_theta_train.sh
bash ./scripts/for_kaggle/UNet_small_theta_train.sh
```  
#### Leaderboard MSE      
Best MSE: 0.00995 (obtained by bagging all cross-validation weights of two models).  
MSE using the following inference API: 0.00996  
- For convenience in reproduction, I did not use the best MSE strategy. Achieving 0.00996 is considered successful reproduction.
#### Training Logs  
To help you better reproduce the results, you can refer to my training logs:  
- [xi_train_log](logs/xi_train_log.out)
- [theta_train_log](logs/theta_train_log.out)
- [ustar_train_log](logs/ustar_train_log.out)
### Step 2: Inference  
Avoid using the inference method in the `exp` module, as it lacks bagging and post-processing steps. Instead, use the `submit_api` function below.  
Inference requires at least 32G of memory.
```python
from for_kaggle import submit_api
# example
theta = np.fromfile('dataset/dataset/test/theta_K_id098830.dat', dtype='<f4').reshape(5, 113, 32) 
ustar = np.fromfile('dataset/dataset/test/ustar_ms-1_id098830.dat', dtype='<f4').reshape(5, 113, 32)
xi = np.fromfile('dataset/dataset/test/xi_id098830.dat', dtype='<f4').reshape(5, 113, 32)
submit_api(theta, ustar, xi, 40, 'pred.npy', 'cpu')
```
- Parameters of `submit_api`: `theta`, `ustar`, and `xi` must have a shape of (5, 113, 32), `pred_len` can be any positive integer, `save_path` is the path to save the prediction file, and `device` can only be set to `gpu` or `cpu`. The return shape is (pred_len, 113, 32). (Note: This function does not require `alpha` or `u` parameters, and it does not support batch operations, so only one time series can be tested at a time).
- Setting the `device` parameter to `gpu` is recommended to speed up inference.
```python
def submit_api(
        theta: np.ndarray, 
        ustar: np.ndarray, 
        xi: np.ndarray, 
        pred_len: int, 
        save_path: str, 
        device: str = 'cpu', 
) -> np.ndarray:
```
- Reading the prediction file:
```python
import numpy as np
pred = np.load(save_path) # shape: pred_len, 113, 32
```  
- **Update: `submit_api` now supports batch operations. `theta`, `ustar`, and `xi` are required to have a shape of (B, 5, 113, 32). I recommend using batch operations to significantly improve inference speed. However, do not set the batch size too high; if the batch is too large, you should divide it into multiple chunks for inference.**  
```python 
from for_kaggle import submit_api
import numpy as np
import pandas as pd
import os
from tqdm import tqdm
from joblib import dump

test_dir = 'dataset/dataset/test'
df = pd.read_csv('dataset/dataset/test.csv')

# batch example
theta_list = []
ustar_list = []
xi_list = []
indexes = []
for i in tqdm(range(len(df))):
    theta = np.fromfile(os.path.join(test_dir, df.iloc[i]['theta_filename']), dtype='<f4').reshape(5, 113, 32)
    ustar = np.fromfile(os.path.join(test_dir, df.iloc[i]['ustar_filename']), dtype='<f4').reshape(5, 113, 32)
    xi = np.fromfile(os.path.join(test_dir, df.iloc[i]['xi_filename']), dtype='<f4').reshape(5, 113, 32)
    theta_list.append(theta)
    ustar_list.append(ustar)
    xi_list.append(xi)
    indexes.append(df.iloc[i]['id'])

submit_api(
    theta=np.stack(theta_list, axis=0), 
    ustar=np.stack(ustar_list, axis=0), 
    xi=np.stack(xi_list, axis=0), 
    pred_len=30, 
    save_path='temp/pred.npy', 
    device='cpu'
)
dump(indexes, 'temp/indexes.joblib')
```  
```python
import numpy as np
# batch read
pred = np.load(save_path) # B, pred_len, 113, 32
```
### Contact Information  
If you encounter any issues during the reproduction process, contact me at m15004059308@163.com