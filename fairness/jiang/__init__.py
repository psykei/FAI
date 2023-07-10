from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F
from utils import PyTorchConditions


PATH = Path(__file__).parents[0]


class Net(nn.Module):
    def __init__(self, input_size, neurons_per_layer=None):
        super(Net, self).__init__()
        if neurons_per_layer is None:
            neurons_per_layer = [100, 50]
        self.first = nn.Linear(input_size, neurons_per_layer[0])
        for i in range(1, len(neurons_per_layer)):
            setattr(self, f'fc{i}', nn.Linear(neurons_per_layer[i - 1], neurons_per_layer[i]))
        self.last = nn.Linear(neurons_per_layer[-1], 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = F.selu(self.first(x))
        out = F.selu(self.fc(out))
        out = self.last(out)
        return self.sigmoid(out)


def evaluate(model, data_fitting_loss, fairness_penalty, x_train, y_train, z_train, x_val, y_val, z_val, device_gpu):
    def accuracy(output, labels):
        output = output.squeeze()
        preds = (output > 0.5).type_as(labels)
        correct = preds.eq(labels).double()
        correct = correct.sum()
        acc = correct.item() / len(labels)
        return acc

    prediction = model(x_train).detach().flatten()
    loss_train = data_fitting_loss(prediction, y_train).item()
    dp_train = fairness_penalty(prediction, z_train, device_gpu).item()
    acc_train = accuracy(prediction, y_train)
    prediction = model(x_val).detach().flatten()
    loss_val = data_fitting_loss(prediction, y_val).item()
    acc_val = accuracy(prediction, y_val)
    dp_val = fairness_penalty(prediction, z_val, device_gpu).item()
    return loss_train, loss_val, acc_train, acc_val, dp_train, dp_val


def regularized_learning(dataset_loader, x_val, y_val, z_val, x_test, y_test, z_test, model, fairness_penalty,
                         device_gpu, penalty_coefficient, data_fitting_loss, num_epochs=5000):
    # mse regression objective
    # data_fitting_loss = nn.MSELoss()

    # stochastic optimizer
    optimizer = torch.optim.Adam(model.parameters())
    conditions = PyTorchConditions(fairness_metric_name="demographic_parity", model=model, max_epochs=num_epochs)

    for epoch in range(num_epochs):
        for i, (x, y, z) in enumerate(dataset_loader):
            optimizer.zero_grad()
            outputs = model(x).flatten()
            loss = data_fitting_loss(outputs, y)
            loss += penalty_coefficient * fairness_penalty(outputs, z, device_gpu)

            loss.backward()
            optimizer.step()
        loss_val, loss_test, acc_val, acc_test, dp_val, dp_test = evaluate(model, data_fitting_loss, fairness_penalty,
                                                                           x_val, y_val, z_val, x_test, y_test, z_test,
                                                                           device_gpu)
        # Early stopping
        if conditions.early_stop(epoch=epoch, accuracy=acc_val, fairness_metric=dp_val):
            break
    y_test_pred = model(x_test).detach().flatten()
    return y_test_pred
