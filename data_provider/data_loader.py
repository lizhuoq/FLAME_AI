import os
import numpy as np
import pandas as pd
import glob
import re
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
# from sktime.datasets import load_from_tsfile_to_dataframe
import warnings
from joblib import load

warnings.filterwarnings('ignore')


class Dataset_flame(Dataset):
    def __init__(self, args, flag='train'):
        self.meta_path = '/data/home/scv7343/run/flame/Time-Series-Library-main/dataset/dataset/train.csv'
        self.data_dir = '/data/home/scv7343/run/flame/Time-Series-Library-main/dataset/dataset/train'
        self.flag = flag
        self.n_fold = args.n_fold
        self.seq_len = args.seq_len
        self.pred_len = args.pred_len
        self.__read_data__()
        self.__load_scaler__()

    def __read_data__(self):
        df_meta = pd.read_csv(self.meta_path)
        val_id = df_meta.iloc[self.n_fold]['id']
        df_meta = df_meta[df_meta['id'] != val_id] if self.flag == 'train' else df_meta.iloc[[self.n_fold]]
        self.n_samples = len(df_meta) * (150 - self.seq_len - self.pred_len + 1)
        self.metadata = df_meta

    def __load_scaler__(self):
        self.scaler: dict[str, StandardScaler] = load('/data/home/scv7343/run/flame/Time-Series-Library-main/dataset/feature_scaler.joblib')

    def __getitem__(self, index):
        """
        Return:
            x shape: seq_len, n_features, 113, 32
            y shape: pred_len, n_features, 113, 32
        """
        id_index = index // (150 - self.seq_len - self.pred_len + 1)
        time_index = index % (150 - self.seq_len - self.pred_len + 1)
        theta, ustar, xi = self.__load_dynamic_features__(id_index, time_index)
        manual_feature = self.__feature_engineering__(theta[:self.seq_len], ustar[:self.seq_len], xi[:self.seq_len])
        constant_x = self.__load_constant_features__(id_index)
        constant_x0 = np.full_like(theta, constant_x[0])
        constant_x1 = np.full_like(theta, constant_x[1])
        time_series = np.stack([theta, ustar, constant_x0, constant_x1, xi], axis=1)
        x = time_series[:self.seq_len]
        x = np.concatenate([x, manual_feature], axis=1)
        y = time_series[self.seq_len:]
        return x, y

    def __load_dynamic_features__(self, id_index, time_index):
        '''
        Return:
            theta shape: self.seq_len + self.pred_len, 113, 32
            ustar shape: self.seq_len + self.pred_len, 113, 32
            xi shape: self.seq_len + self.pred_len, 113, 32
        '''
        meta = self.metadata.iloc[id_index]
        theta_filename = meta['theta_filename']
        ustar_filename = meta['ustar_filename']
        xi_filename = meta['xi_filename']
        theta = np.fromfile(os.path.join(self.data_dir, theta_filename), dtype="<f4").reshape(150,113,32)[time_index: time_index + self.seq_len + self.pred_len]
        ustar = np.fromfile(os.path.join(self.data_dir, ustar_filename), dtype="<f4").reshape(150,113,32)[time_index: time_index + self.seq_len + self.pred_len]
        xi = np.fromfile(os.path.join(self.data_dir, xi_filename), dtype="<f4").reshape(150,113,32)[time_index: time_index + self.seq_len + self.pred_len]
        theta = self.__transform_dynamic_features__(theta, 'theta')
        ustar = self.__transform_dynamic_features__(ustar, 'ustar')
        return theta, ustar, xi
    
    def __load_constant_features__(self, id_index):
        meta = self.metadata.iloc[id_index]
        scaler = self.scaler['constant']
        feature_name = scaler.feature_names_in_.tolist()
        data = meta[feature_name].values
        return (data - scaler.mean_) / scaler.scale_

    def __transform_dynamic_features__(self, data, feature_name):
        scaler = self.scaler[feature_name]
        return (data - scaler.mean_) / scaler.scale_

    def __len__(self):
        return self.n_samples
    
    @staticmethod
    def __cal_diff__(data):
        '''
        Input:
            data shape: S, H, W
        Return:
            shape: S, H, W
        '''
        data = np.diff(data, axis=0)
        return np.pad(data, ((1, 0), (0, 0), (0, 0)), 'edge')

    @staticmethod
    def __cal_mean__(data):
        '''
        Input:
            data shape: S, H, W
        Return:
            shape: S, H, W
        '''
        S, H, W = data.shape
        data = np.mean(data, axis=1, keepdims=True)
        return np.tile(data, (1, H, 1))
    
    @staticmethod
    def __cal_total_mean__(data):
        '''
        Input:
            data shape: S, H, W
        Return:
            shape: S, H, W
        '''
        S, H, W = data.shape
        data = np.mean(data, axis=(0, 1, 2), keepdims=True)
        return np.tile(data, (S, H, W))
    
    @staticmethod
    def __feature_engineering__(theta, ustar, xi):
        '''
        Return:
            shape: S, C, H, W
        '''
        theta_timediff = Dataset_flame.__cal_diff__(theta)
        ustar_timediff = Dataset_flame.__cal_diff__(ustar)
        xi_timediff = Dataset_flame.__cal_diff__(xi)
        xi_x_theta = theta * xi
        xi_x_ustar = ustar * xi
        HWdiff = [] # 6
        for data in [theta, ustar, xi]:
            HWdiff.append(Dataset_flame.__cal_diff__(data.transpose(1, 0, 2)).transpose(1, 0, 2))
            HWdiff.append(Dataset_flame.__cal_diff__(data.transpose(2, 1, 0)).transpose(2, 1, 0))
        HWTmean = [] # 18
        for data in [theta, ustar, xi, theta_timediff, ustar_timediff, xi_timediff]:
            HWTmean.append(Dataset_flame.__cal_mean__(data))
            HWTmean.append(Dataset_flame.__cal_mean__(data.transpose(1, 0, 2)).transpose(1, 0, 2))
            HWTmean.append(Dataset_flame.__cal_mean__(data.transpose(0, 2, 1)).transpose(0, 2, 1))
        total_mean = [] # 6
        for data in [theta, ustar, xi, theta_timediff, ustar_timediff, xi_timediff]:
            total_mean.append(Dataset_flame.__cal_total_mean__(data))
        return np.stack([theta_timediff, ustar_timediff, xi_timediff, xi_x_theta, xi_x_ustar] + HWdiff + HWTmean + total_mean, axis=1) # 35