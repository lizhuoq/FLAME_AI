# 2024 FLAME AI Challenge  
## 配置环境  
```bash
git clone https://github.com/lizhuoq/FLAME_AI.git
cd FLAME_AI
conda env create -f environment.yml
conda activate flame
```  
操作系统：Linux
## 用于测试的接口
### 第一步：下载权重文件（需要大约160G的存储空间，需要联网）
```python
from for_kaggle import download_checkpoints
download_checkpoints('checkpoints')
```
### 第一步：重新训练  （不需要联网）
#### 确保模型是自回归的  
本仓库对`xi`，`theta`，`ustar`三个变量分别优化（原因：经过测试如果进行多目标优化，会导致三个变量的误差增加）  
#### 交叉验证和bagging策略  
本仓库采用9折交叉验证，即每次留出一个train id的数据用于验证，其余train id的数据用于训练。因此对于每个变量有9个权重文件，使用这9个权重推理得到的结果进行bagging。因此对于三个变量共有27个权重文件，因此你至少要有160G的存储空间。如果你要复现我的结果，必须严格遵守我的策略。
#### GPU和内存
本仓库的实验是在NVIDIA Tesla V100-SXM2 32GB GPU上运行的，确保你的GPU有32G的显存，确保至少有32G的内存  
#### 数据存放位置
将数据存放在`./dataset`下  
- 数据的存储结构  
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
#### 修改绝对路径  
为了避免出现意外的问题，我在本仓库设置了绝对路径，如果你要复现，需要将下面两个文件里所有绝对路径修改为你的绝对路径  
- `./data_provider/data_loader.py`
```python
# Lines 14-15
self.meta_path = '/data/home/scv7343/run/flame/Time-Series-Library-main/dataset/dataset/train.csv'
self.data_dir = '/data/home/scv7343/run/flame/Time-Series-Library-main/dataset/dataset/train'
# Lines 31
self.scaler: dict[str, StandardScaler] = load('/data/home/scv7343/run/flame/Time-Series-Library-main/dataset/feature_scaler.joblib')
```
- `./exp/exp_flame.py`
```python
# Lines 196-198
df_meta = pd.read_csv('/data/home/scv7343/run/flame/Time-Series-Library-main/dataset/dataset/test.csv')
dir_path = '/data/home/scv7343/run/flame/Time-Series-Library-main/dataset/dataset/test'
feature_scaler: dict[str, StandardScaler] = load('/data/home/scv7343/run/flame/Time-Series-Library-main/dataset/feature_scaler.joblib')
```
#### 训练脚本  
- 在运行脚本前确保你有160G的储存空间，否则会导致权重无法写入
- 确保你的内存至少32G，GPU的显存至少32G（如果不能满足，不要调节超参数，调节超参数会导致无法复现）
```bash
bash ./scripts/for_kaggle/UNet_small_xi_train.sh
bash ./scripts/for_kaggle/UNet_small_theta_train.sh
bash ./scripts/for_kaggle/UNet_small_theta_train.sh
```  
#### 排行榜MSE      
最好MSE：0.00995 对两个模型的所有交叉验证权重进行bagging  
使用下面推理api可以得到的MSE：0.00996  
- 为了方便你们复现，我没有采用最好的MSE策略，我认为达到0.00996就算复现成功 
#### 训练日志
为了帮助你更好的复现，可以参考我的训练日志
- [xi_train_log](logs/xi_train_log.out)
- [theta_train_log](logs/theta_train_log.out)
- [ustar_train_log](logs/ustar_train_log.out)
### 第二步：推理  
禁止使用exp中的inference方法，因为缺少bagging操作和后处理方法，推荐使用下面的`submit_api`函数
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
- `device`参数推荐设置为`gpu`，会加快推理速度。
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
复现过程如果有任何问题联系我  
m15004059308@163.com