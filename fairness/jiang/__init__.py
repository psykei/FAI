from math import sqrt, pi
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader

PATH = Path(__file__).parents[0]


def train_and_predict_jiang_classifier(net, fairness_dataset, device, lambda_value, epochs, batch_size, conditions):
    net = net.to(device)
    test_sol = 1e-3
    x_appro = torch.arange(test_sol, 1 - test_sol, test_sol).to(device)
    KDE_FAIR = KDE_fair(x_appro)
    penalty = KDE_FAIR.forward
    # Fair classifier training
    (
        train_datasets,
        valid_datasets,
        test_datasets,
    ) = fairness_dataset.get_dataset_in_tensor()
    _, y_train, z_train, x_train = train_datasets
    _, y_valid, z_valid, x_valid = valid_datasets
    _, y_test, z_test, x_test = test_datasets
    train_dataset = TensorDataset(x_train, y_train, z_train)
    dataloader = DataLoader(train_dataset, batch_size, shuffle=True)
    y_pred = regularized_learning(
        dataloader,
        x_valid,
        y_valid,
        z_valid,
        x_test,
        y_test,
        z_test,
        net,
        penalty,
        device,
        lambda_value,
        nn.functional.binary_cross_entropy,
        epochs,
        conditions
    )

    # Compute metrics
    # Round to the nearest integer
    y_pred = np.squeeze(np.array(y_pred))
    return y_pred


def evaluate(
    model, data_fitting_loss, fairness_penalty, x_val, y_val, z_val, device_gpu
):
    def accuracy(output, labels):
        output = output.squeeze()
        preds = (output > 0.5).type_as(labels)
        correct = preds.eq(labels).double()
        correct = correct.sum()
        acc = correct.item() / len(labels)
        return acc

    prediction = model(x_val).detach().flatten()
    loss_val = data_fitting_loss(prediction, y_val).item()
    acc_val = accuracy(prediction, y_val)
    dp_val = fairness_penalty(prediction, z_val, device_gpu).item()
    return loss_val, acc_val, dp_val


def regularized_learning(
    dataset_loader,
    x_val,
    y_val,
    z_val,
    x_test,
    y_test,
    z_test,
    model,
    fairness_penalty,
    device_gpu,
    penalty_coefficient,
    data_fitting_loss,
    num_epochs,
    conditions
):
    # mse regression objective
    # data_fitting_loss = nn.MSELoss()

    # stochastic optimizer
    optimizer = torch.optim.Adam(model.parameters())

    for epoch in range(num_epochs):
        for i, (x, y, z) in enumerate(dataset_loader):
            outputs = model(x).flatten()
            loss = data_fitting_loss(outputs, y)
            loss += penalty_coefficient * fairness_penalty(outputs, z, device_gpu)
            optimizer.zero_grad()
            if (torch.isnan(loss)).any():
                continue
            loss.backward()
            optimizer.step()
        (
            loss_val,
            acc_val,
            dp_val,
        ) = evaluate(
            model, data_fitting_loss, fairness_penalty, x_val, y_val, z_val, device_gpu
        )
        # Early stopping
        if conditions.early_stop(epoch=epoch, accuracy=acc_val, fairness_metric=dp_val):
            break
    y_test_pred = model(x_test).detach().flatten()
    return y_test_pred


class KDE_fair:
    """
    A Gaussian KDE implemented in pytorch for the gradients to flow in pytorch optimization.
    Keep in mind that KDE are not scaling well with the number of dimensions and this implementation is not really
    optimized...
    """

    def __init__(self, x_test):
        # self.train_x = x_train
        # self.train_y = y_train
        self.x_test = x_test

    def forward(self, y_train, x_train, device_gpu):
        n = x_train.size()[0]
        # print(f'n={n}')
        d = 1
        bandwidth = torch.tensor((n * (d + 2) / 4.0) ** (-1.0 / (d + 4))).to(device_gpu)

        y_hat = self.kde_regression(bandwidth, x_train, y_train)
        y_mean = torch.mean(y_train)
        pdf_values = self.pdf(bandwidth, x_train)

        DP = torch.sum(torch.abs(y_hat - y_mean) * pdf_values) / torch.sum(pdf_values)
        return DP

    def kde_regression(self, bandwidth, x_train, y_train):
        n = x_train.size()[0]
        X_repeat = self.x_test.repeat_interleave(n).reshape((-1, n))
        attention_weights = nn.functional.softmax(
            -((X_repeat - x_train) ** 2) / (bandwidth**2) / 2, dim=1
        )
        y_hat = torch.matmul(attention_weights, y_train)
        return y_hat

    def pdf(self, bandwidth, x_train):
        n = x_train.size()[0]

        data = self.x_test.repeat_interleave(n).reshape((-1, n))
        train_x = x_train.unsqueeze(0)
        two_tensor = 2
        pdf_values = (
            (
                torch.exp(
                    -(
                        (data - train_x) ** two_tensor
                        / (bandwidth**two_tensor)
                        / two_tensor
                    )
                )
            ).mean(dim=-1)
            / sqrt(2 * pi)
            / bandwidth
        )

        return pdf_values
