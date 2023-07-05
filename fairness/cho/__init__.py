import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from dataset.cho_dataset_pipeline import CustomDataset

TAU = 0.5
# Approximation of Q-function given by López-Benítez & Casadevall (2011) based on a second-order exponential function & Q(x) = 1- Q(-x):
A = 0.4920
B = 0.2887
C = 1.1893


def q_function(x):
    return torch.exp(-A * x ** 2 - B * x - C)


def CDF_tau(y_hat, h=0.01, tau=TAU):
    m = len(y_hat)
    y_tilde = (tau - y_hat) / h
    sum_ = torch.sum(q_function(y_tilde[y_tilde > 0])) \
           + torch.sum(1 - q_function(torch.abs(y_tilde[y_tilde < 0]))) \
           + 0.5 * (len(y_tilde[y_tilde == 0]))
    return sum_ / m


def Huber_loss(x, delta):
    if x.abs() < delta:
        return (x ** 2) / 2
    return delta * (x.abs() - delta / 2)


def measures_from_y_hat(y, z, y_hat=None, threshold=0.5):
    assert isinstance(y, np.ndarray)
    assert isinstance(z, np.ndarray)
    assert y_hat is not None
    assert isinstance(y_hat, np.ndarray)

    y_tilde = None
    if y_hat is not None:
        y_tilde = (y_hat >= threshold).astype(np.float32)
    assert y_tilde.shape == y.shape and y.shape == z.shape

    # Accuracy
    acc = (y_tilde == y).astype(np.float32).mean()
    # DP
    DDP = abs(np.mean(y_tilde[z == 0]) - np.mean(y_tilde[z == 1]))
    # EO
    Y_Z0, Y_Z1 = y[z == 0], y[z == 1]
    Y1_Z0 = Y_Z0[Y_Z0 == 1]
    Y0_Z0 = Y_Z0[Y_Z0 == 0]
    Y1_Z1 = Y_Z1[Y_Z1 == 1]
    Y0_Z1 = Y_Z1[Y_Z1 == 0]

    FPR, FNR = {}, {}
    FPR[0] = np.sum(y_tilde[np.logical_and(z == 0, y == 0)]) / len(Y0_Z0)
    FPR[1] = np.sum(y_tilde[np.logical_and(z == 1, y == 0)]) / len(Y0_Z1)

    FNR[0] = np.sum(1 - y_tilde[np.logical_and(z == 0, y == 1)]) / len(Y1_Z0)
    FNR[1] = np.sum(1 - y_tilde[np.logical_and(z == 1, y == 1)]) / len(Y1_Z1)

    TPR_diff = abs((1 - FNR[0]) - (1 - FNR[1]))
    FPR_diff = abs(FPR[0] - FPR[1])
    DEO = TPR_diff + FPR_diff

    data = [acc, DDP, DEO]
    columns = ['acc', 'DDP', 'DEO']
    return pd.DataFrame([data], columns=columns)


def train_fair_classifier(dataset, net, optimizer, lr_scheduler, fairness, lambda_, h, delta, device, n_epochs=5000, batch_size=500, seed=0):
    # Retrieve train/test split pytorch tensors for index=split
    train_tensors, test_tensors = dataset.get_dataset_in_tensor()
    X_train, Y_train, Z_train, XZ_train = train_tensors
    X_test, Y_test, Z_test, XZ_test = test_tensors

    # Retrieve train/test split numpy arrays for index=split
    train_arrays, test_arrays = dataset.get_dataset_in_ndarray()
    X_train_np, Y_train_np, Z_train_np, XZ_train_np = train_arrays
    X_test_np, Y_test_np, Z_test_np, XZ_test_np = test_arrays
    sensitive_attrs = dataset.sensitive_attrs

    custom_dataset = CustomDataset(XZ_train, Y_train, Z_train)
    if batch_size == 'full':
        batch_size_ = XZ_train.shape[0]
    elif isinstance(batch_size, int):
        batch_size_ = batch_size
    data_loader = DataLoader(custom_dataset, batch_size=batch_size_, shuffle=True)

    pi = torch.tensor(np.pi).to(device)
    phi = lambda x: torch.exp(-0.5 * x ** 2) / torch.sqrt(2 * pi)  # normal distribution

    # An empty dataframe for logging experimental results
    df = pd.DataFrame()
    df_ckpt = pd.DataFrame()

    loss_function = nn.BCELoss()
    costs = []
    for epoch in range(n_epochs):
        for i, (xz_batch, y_batch, z_batch) in enumerate(data_loader):
            xz_batch, y_batch, z_batch = xz_batch.to(device), y_batch.to(device), z_batch.to(device)
            Yhat = net(xz_batch)
            Ytilde = torch.round(Yhat.detach().reshape(-1))
            cost = 0
            dtheta = 0
            m = z_batch.shape[0]

            # prediction loss
            p_loss = loss_function(Yhat.squeeze(), y_batch)
            cost += (1 - lambda_) * p_loss

            # DP_Constraint
            if fairness == 'DP':
                Pr_Ytilde1 = CDF_tau(Yhat.detach(), h, TAU)
                for z in sensitive_attrs:
                    Pr_Ytilde1_Z = CDF_tau(Yhat.detach()[z_batch == z], h, TAU)
                    m_z = z_batch[z_batch == z].shape[0]

                    Delta_z = Pr_Ytilde1_Z - Pr_Ytilde1
                    Delta_z_grad = torch.dot(phi((TAU - Yhat.detach()[z_batch == z]) / h).view(-1),
                                             Yhat[z_batch == z].view(-1)) / h / m_z
                    Delta_z_grad -= torch.dot(phi((TAU - Yhat.detach()) / h).view(-1),
                                              Yhat.view(-1)) / h / m

                    if Delta_z.abs() >= delta:
                        if Delta_z > 0:
                            Delta_z_grad *= lambda_ * delta
                            cost += Delta_z_grad
                        else:
                            Delta_z_grad *= -lambda_ * delta
                            cost += Delta_z_grad
                    else:
                        Delta_z_grad *= lambda_ * Delta_z
                        cost += Delta_z_grad

            # EO_Constraint
            elif fairness == 'EO':
                for y in [0, 1]:
                    Pr_Ytilde1_Y = CDF_tau(Yhat[y_batch == y].detach(), h, TAU)
                    m_y = y_batch[y_batch == y].shape[0]
                    for z in sensitive_attrs:
                        Pr_Ytilde1_ZY = CDF_tau(Yhat[(y_batch == y) & (z_batch == z)].detach(), h, TAU)
                        m_zy = z_batch[(y_batch == y) & (z_batch == z)].shape[0]
                        Delta_zy = Pr_Ytilde1_ZY - Pr_Ytilde1_Y
                        Delta_zy_grad = torch.dot(
                            phi((TAU - Yhat[(y_batch == y) & (z_batch == z)].detach()) / h).view(-1),
                            Yhat[(y_batch == y) & (z_batch == z)].view(-1)
                        ) / h / m_zy
                        Delta_zy_grad -= torch.dot(
                            phi((TAU - Yhat[y_batch == y].detach()) / h).view(-1),
                            Yhat[y_batch == y].view(-1)
                        ) / h / m_y

                        if Delta_zy.abs() >= delta:
                            if Delta_zy > 0:
                                Delta_zy_grad *= lambda_ * delta
                                cost += Delta_zy_grad
                            else:
                                Delta_zy_grad *= lambda_ * delta
                                cost += -lambda_ * delta * Delta_zy_grad
                        else:
                            Delta_zy_grad *= lambda_ * Delta_zy
                            cost += Delta_zy_grad

            optimizer.zero_grad()
            if (torch.isnan(cost)).any():
                continue
            cost.backward()
            optimizer.step()
            costs.append(cost.item())

            # Print the cost per 10 batches
            # if (i + 1) % 10 == 0 or (i + 1) == len(data_loader):
            #     print('Epoch [{}/{}], Batch [{}/{}], Cost: {:.4f}'.format(epoch + 1, n_epochs, i + 1, len(data_loader), cost.item()), end='\r')
        if lr_scheduler is not None:
            lr_scheduler.step()

        y_hat_train = net(XZ_train).squeeze().detach().cpu().numpy()
        df_temp = measures_from_y_hat(Y_train_np, Z_train_np, y_hat=y_hat_train, threshold=TAU)
        df_temp['epoch'] = epoch * len(data_loader) + i + 1
        df_ckpt = pd.concat([df_ckpt, df_temp], ignore_index=True)

    y_hat_test = net(XZ_test).squeeze().detach().cpu().numpy()
    df_test = measures_from_y_hat(Y_test_np, Z_test_np, y_hat=y_hat_test, threshold=TAU)

    return df_test
