# 2024 FLAME AI Challenge  
## 配置环境  
```bash
git clone https://github.com/lizhuoq/FLAME_AI.git
cd FLAME_AI
conda env create -f environment.yml
conda activate flame
```
## 用于测试的接口
本仓库的实验是在NVIDIA Tesla V100-SXM2 32GB GPU上运行的
### 第一步：下载权重文件（需要大约160G的存储空间，需要联网）
```python
from for_kaggle import download_checkpoints
download_checkpoints('checkpoints')
```
### 第二步：推理
推理至少需要32G的内存
```python
from for_kaggle import submit_api
# example
theta = np.fromfile('dataset/dataset/test/theta_K_id098830.dat', dtype='<f4').reshape(5, 113, 32) 
ustar = np.fromfile('dataset/dataset/test/ustar_ms-1_id098830.dat', dtype='<f4').reshape(5, 113, 32)
xi = np.fromfile('dataset/dataset/test/xi_id098830.dat', dtype='<f4').reshape(5, 113, 32)
submit_api(theta, ustar, xi, 40, 'pred.npy', 'cpu')
```
- `submit_api`的参数，`theta`，`ustar`，`xi`要求shape为（5，113，32），`pred_len`可以人为设置为任意大的正整数，`save_path`为预测文件保存的路径，`device`只可以设置为`gpu`或者`cpu`，返回值的shaope为（pred_len，113，32）。（注意：该函数不需要输入`alpha`和`u`参数，且`submit_api`不支持batch操作，因此一次只能测试一个时间序列）
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
- 读取预测文件
```python
import numpy as np
pred = np.load(save_path) # shape: pred_len, 113, 32
```
### 联系方式
m15004059308@163.com