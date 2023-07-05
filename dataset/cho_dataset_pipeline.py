import numpy as np
import pandas as pd
import torch


def arrays_to_tensor(X, Y, Z, XZ, device):
    return torch.FloatTensor(X).to(device), torch.FloatTensor(Y).to(device), torch.FloatTensor(Z).to(device), torch.FloatTensor(XZ).to(device)


class CustomDataset:
    def __init__(self, X, Y, Z):
        self.X = X
        self.Y = Y
        self.Z = Z

    def __len__(self):
        return len(self.Y)

    def __getitem__(self, index):
        x, y, z = self.X[index], self.Y[index], self.Z[index]
        return x, y, z


class FairnessChoDataset:

    def __init__(self, train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame, device=torch.device('cpu')):
        self.device = device
        self.sensitive_attrs = None
        self.Z_train = None
        self.Z_val = None
        self.Z_test = None
        self.XZ_train = None
        self.XZ_val = None
        self.XZ_test = None
        self.X_train = train.iloc[:, :-1]
        self.X_val = val.iloc[:, :-1]
        self.X_test = test.iloc[:, :-1]
        self.Y_train = train.iloc[:, -1]
        self.Y_val = val.iloc[:, -1]
        self.Y_test = test.iloc[:, -1]

    def prepare_ndarray(self, idx: int = 0):

        def _(x, y) -> (np.ndarray, np.ndarray, np.ndarray, np.ndarray):
            x = x.drop(x.columns[idx], axis=1).to_numpy(dtype=np.float64)
            y = y.to_numpy(dtype=np.float64)
            z = x.iloc[:, idx].to_numpy(dtype=np.float64)
            xz = np.concatenate([x, z.reshape(-1, 1)], axis=1)
            return x, y, z, xz

        self.X_train, self.Y_train, self.Z_train, self.XZ_train = _(self.X_train, self.Y_train)
        self.X_val, self.Y_val, self.Z_val, self.XZ_val = _(self.X_val, self.Y_val)
        self.X_test = self.X_test.to_numpy(dtype=np.float64)
        self.Y_test = self.Y_test.to_numpy(dtype=np.float64)
        self.Z_test = self.Z_test.to_numpy(dtype=np.float64)
        self.XZ_test = np.concatenate([self.X_test, self.Z_test.reshape(-1,1)], axis=1)
        self.sensitive_attrs = sorted(list(set(self.Z_train)))

    def get_dataset_in_ndarray(self):
        return (self.X_train, self.Y_train, self.Z_train, self.XZ_train), \
            (self.X_val, self.Y_val, self.Z_val, self.XZ_val), \
            (self.X_test, self.Y_test, self.Z_test, self.XZ_test)

    def get_dataset_in_tensor(self):
        x_train, y_train, z_train, xz_train = arrays_to_tensor(
            self.X_train, self.Y_train, self.Z_train, self.XZ_train, self.device)
        x_val, y_val, z_val, xz_val = arrays_to_tensor(
            self.X_val, self.Y_val, self.Z_val, self.XZ_val, self.device)
        x_test, y_test, z_test, xz_test = arrays_to_tensor(
            self.X_test, self.Y_test, self.Z_test, self.XZ_test, self.device)
        return (x_train, y_train, z_train, xz_train), \
            (x_val, y_val, z_val, xz_val), \
            (x_test, y_test, z_test, xz_test)
