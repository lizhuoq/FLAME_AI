from data_provider.data_loader import Dataset_flame
import numpy as np
import torch
from models.UNet import Model
import os
from joblib import load
from sklearn.preprocessing import StandardScaler
from typing import Optional
from huggingface_hub import snapshot_download
from tqdm import tqdm

CHECKPOINTS_ROOT = 'checkpoints'
SCALER_PATH = 'dataset/feature_scaler.joblib'
N_MODELS = 9


def download_checkpoints(local_dir: str = CHECKPOINTS_ROOT, token: Optional[str] = None, max_workers: Optional[int] = None) -> None:
    if token is None:
        # public
        snapshot_download(
            repo_id="lizhuoqun/FLAME-Checkpoints", 
            local_dir=local_dir, max_workers=max_workers 
        )
    else:
        # private
        snapshot_download(
            repo_id="lizhuoqun/FLAME-Checkpoints", 
            local_dir=local_dir, token=token, max_workers=max_workers
        )


class Configs:
    enc_in = 23
    c_out = 1
    pred_len = 20
    scale = 4
    seq_len = 5


def load_model(variable: str, 
               n_fold: int, 
               device: str) -> Model:
    if device == 'cpu':
        model = Model(Configs()).float().to(torch.device('cpu'))
    elif device == 'gpu':
        model = Model(Configs()).float().to(torch.device('cuda'))
    setting = f'flame_enc_in_23_target_{variable}_UNet_FLAME_ftM_sl5_ll48_pl20_dm512_nh8_el2_dl1_df2048_expand2_dc4_fc1_ebtimeF_dtTrue_Exp_0nf{n_fold}_we3'
    if device == 'gpu':
        model.load_state_dict(torch.load(os.path.join(CHECKPOINTS_ROOT, setting, 'checkpoint.pth')))
    elif device == 'cpu':
        model.load_state_dict(torch.load(os.path.join(CHECKPOINTS_ROOT, setting, 'checkpoint.pth'), map_location='cpu'))
    return model


def check_shape(data: np.ndarray, 
                seq_len: int) -> None:
    '''
    Input:
        data shape: seq_len, 113, 32
    '''
    assert data.ndim == 3
    assert data.shape[0] == seq_len


def transform_data(data, scaler: StandardScaler):
    return (data - scaler.mean_) / scaler.scale_


def preprocess_data(theta: np.ndarray, 
                    ustar: np.ndarray):
    scaler_dict: dict[StandardScaler] = load(SCALER_PATH)
    theta = transform_data(theta, scaler_dict['theta'])
    ustar = transform_data(ustar, scaler_dict['ustar'])
    return theta, ustar


def inference(x, variable, device):
    '''
    Input:
        x shape: 5, 23, 113, 32
    Return:
        shape: 20, 113, 32
    '''
    outputs = []
    x = torch.tensor(x).float().to(torch.device('cpu') if device == 'cpu' else torch.device('cuda')).unsqueeze(0)
    for i in tqdm(range(N_MODELS)):
        model = load_model(variable, i, device)
        model.eval()
        with torch.no_grad():
            output = model(x).detach().cpu().numpy()[0, ...]
        outputs.append(output)
    outputs = np.stack(outputs, axis=0)
    return outputs.mean(axis=0)[:, -1, :, :]


def concat_features(theta, ustar, xi, transform=False):
    '''
    Input:
        theta shape: 5, 113, 32
        ustar shape: 5, 113, 32
        xi shape: 5, 113, 32
    Return:
        x shape: 5, 23, 113, 32
    '''
    if transform:
        theta, ustar = preprocess_data(theta, ustar)
    x = np.stack([theta, ustar, xi], axis=1)
    manual_features = Dataset_flame.__feature_engineering__(theta, ustar, xi)
    x = np.concatenate([x, manual_features], axis=1)[:, :23, :, :]
    return x


def submit_api(
        theta: np.ndarray, 
        ustar: np.ndarray, 
        xi: np.ndarray, 
        pred_len: int, 
        save_path = str, 
        device: str = 'cpu', 
) -> np.ndarray:
    '''
    Input:
        theta shape: 5, 113, 32
        ustar shape: 5, 113, 32
        xi shape: 5, 113, 32
    Return:
        shape: pred_len, 113, 32
    '''
    for data in [theta, ustar, xi]:
        check_shape(data, 5)

    x = concat_features(theta, ustar, xi, transform=True)

    if pred_len <= 20:
        preds = inference(x, 'xi', device)[:pred_len]
        preds = np.clip(preds, 0, 1)
    else:
        n_ar = pred_len // 20 + 1
        preds = []
        for i in range(n_ar):
            xi = inference(x, 'xi', device)
            xi = np.clip(xi, 0, 1)
            if i != n_ar - 1:
                theta = inference(x, 'theta', device)
                ustar = inference(x, 'ustar', device)

                x = concat_features(theta[-5:, ...], ustar[-5:, ...], xi[-5:, ...])

            preds.append(xi)
        preds = np.concatenate(preds, axis=0)[:pred_len]

    np.save(save_path, preds)
    return preds


if __name__ == '__main__':
    # example
    breakpoint()
    theta = np.fromfile('dataset/dataset/test/theta_K_id098830.dat', dtype='<f4').reshape(5, 113, 32)
    ustar = np.fromfile('dataset/dataset/test/ustar_ms-1_id098830.dat', dtype='<f4').reshape(5, 113, 32)
    xi = np.fromfile('dataset/dataset/test/xi_id098830.dat', dtype='<f4').reshape(5, 113, 32)
    submit_api(theta, ustar, xi, 25, 'pred.npy', 'gpu')

    # download
    download_checkpoints(token='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') # Private
    download_checkpoints() # Public