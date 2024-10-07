from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, visual
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
from transformers import get_cosine_schedule_with_warmup
from sklearn.metrics import mean_squared_error
import pandas as pd
from joblib import load, dump
from sklearn.preprocessing import StandardScaler
import torch.nn.functional as F
from data_provider.data_loader import Dataset_flame

warnings.filterwarnings('ignore')


class Exp_flame(Exp_Basic):
    def __init__(self, args):
        super(Exp_flame, self).__init__(args)
        self.invalid_feature_id = [2, 3]

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                outputs = self.model(batch_x[:, :, [i for i in range(self.args.enc_in + len(self.invalid_feature_id)) if i not in self.invalid_feature_id], :, :])
                # outputs[:, :, -1, :, :] = F.sigmoid(outputs[:, :, -1, :, :])

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()

                loss = criterion(pred[:, :, -1, :, :], true[:, :, self.args.target_index, :, :])

                total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='val')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()
        scheduler = get_cosine_schedule_with_warmup(model_optim, train_steps * self.args.warmup_epochs, train_steps * self.args.train_epochs)

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                
                outputs = self.model(batch_x[:, :, [i for i in range(self.args.enc_in + len(self.invalid_feature_id)) if i not in self.invalid_feature_id], :, :])
                # outputs[:, :, -1, :, :] = F.sigmoid(outputs[:, :, -1, :, :])

                loss = criterion(outputs[:, :, -1, :, :], batch_y[:, :, self.args.target_index, :, :])
                train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                loss.backward()
                model_optim.step()
                scheduler.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            # test_loss = self.vali(test_data, test_loader, criterion)
            test_loss = vali_loss

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            # adjust_learning_rate(model_optim, epoch + 1, self.args)
            print('Updating learning rate to {}'.format(scheduler.get_last_lr()[0]))

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='val')
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        preds = []
        trues = []

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                outputs = self.model(batch_x[:, :, [i for i in range(self.args.enc_in + len(self.invalid_feature_id)) if i not in self.invalid_feature_id], :, :])
                # outputs[:, :, -1, :, :] = F.sigmoid(outputs[:, :, -1, :, :])

                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()

                pred = outputs
                true = batch_y

                preds.append(pred[:, :, -1, :, :])
                trues.append(true[:, :, self.args.target_index, :, :])

        preds = np.concatenate(preds, axis=0) # B, T, H, W
        if self.args.target_index == -1:
            preds = np.clip(preds, 0, 1)
        trues = np.concatenate(trues, axis=0)
        print('test shape:', preds.shape, trues.shape)

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
            
        mse = mean_squared_error(trues.reshape(-1), preds.reshape(-1))
        print('mse:{}'.format(mse))
        f = open("result_flame.txt", 'a')
        f.write(setting + "  \n")
        f.write('mse:{}'.format(mse))
        f.write('\n')
        f.write('\n')
        f.close()

        np.save(folder_path + 'pred.npy', preds)
        np.save(folder_path + 'true.npy', trues)

        return
    
    def inference(self, setting):
        print('loading model')
        self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        folder_path = './pred_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        df_meta = pd.read_csv('/data/home/scv7343/run/flame/Time-Series-Library-main/dataset/dataset/test.csv')
        dir_path = '/data/home/scv7343/run/flame/Time-Series-Library-main/dataset/dataset/test'
        feature_scaler: dict[str, StandardScaler] = load('/data/home/scv7343/run/flame/Time-Series-Library-main/dataset/feature_scaler.joblib')

        self.model.eval()
        with torch.no_grad():
            batch_x = []
            index_list = []
            for i in range(len(df_meta)):
                index = df_meta.iloc[i]['id']
                theta = np.fromfile(os.path.join(dir_path, df_meta.iloc[i]['theta_filename']), dtype="<f4").reshape(5,113,32)
                ustar = np.fromfile(os.path.join(dir_path, df_meta.iloc[i]['ustar_filename']), dtype="<f4").reshape(5,113,32)
                xi = np.fromfile(os.path.join(dir_path, df_meta.iloc[i]['xi_filename']), dtype="<f4").reshape(5,113,32)

                theta = (theta - feature_scaler['theta'].mean_) / feature_scaler['theta'].scale_
                ustar = (ustar - feature_scaler['ustar'].mean_) / feature_scaler['ustar'].scale_

                constant_x = df_meta.iloc[i][feature_scaler['constant'].feature_names_in_.tolist()].values
                constant_x = (constant_x - feature_scaler['constant'].mean_) / feature_scaler['constant'].scale_
                constant_x0 = np.full_like(theta, constant_x[0])
                constant_x1 = np.full_like(theta, constant_x[1])

                manual_feature = Dataset_flame.__feature_engineering__(theta, ustar, xi)

                x = np.stack([theta, ustar, constant_x0, constant_x1, xi], axis=1)
                x = np.concatenate([x, manual_feature], axis=1)
                x = torch.tensor(x).float().to(self.device).unsqueeze(0) # B S C H W

                batch_x.append(x)
                index_list.append(index)

            batch_x = torch.cat(batch_x, dim=0)

            pred = self.model(batch_x[:, :, [i for i in range(self.args.enc_in + len(self.invalid_feature_id)) if i not in self.invalid_feature_id], :, :])
            # pred[:, :, -1, :, :] = F.sigmoid(pred[:, :, -1, :, :])
            pred = pred.detach().cpu().numpy()
                
            print(pred.shape)
            np.save(os.path.join(folder_path, 'pred.npy'), pred)
            dump(index_list, os.path.join(folder_path, 'indexes.joblib'))

        # print('delecting checkpoint')
        # os.remove(os.path.join('./checkpoints/' + setting, 'checkpoint.pth'))
                

