import hashlib
import os
import random
import numpy as np
from sklearn.metrics import accuracy_score
from torch import optim, nn
from torch.utils.data import DataLoader, TensorDataset
from configuration import *
from dataset.cho_data_pipeline import FairnessChoDataset
from fairness.cho import Classifier
from fairness.jiang import PATH as JIANG_PATH, regularized_learning, KDE_fair
from fairness.metric import is_demographic_parity, is_equalized_odds, is_disparate_impact


DATA_LOSS = nn.functional.binary_cross_entropy
CONTINUOUS = True if IDX == 0 else False
# CONTINUOUS = False

for metric in CHO_METRICS:
    for JIANG_LAMBDA in JIANG_LAMBDAS:
        idf = '_'.join([str(x) for x in [SEED, K, EPOCHS, BATCH_SIZE, NEURONS_PER_LAYER, IDX, metric, JIANG_LAMBDA]])
        if not ONE_HOT:
            idf += "_" + str(ONE_HOT)
        filename = str(JIANG_PATH) + os.sep + LOG + os.sep + hashlib.md5(str(idf).encode()).hexdigest() + ".txt"
        if not os.path.exists(filename):
            dataset, train, test, kfold = initialize_experiment(filename, metric, JIANG_LAMBDA, preprocess=True, min_max=True)
            mean_accuracy, mean_demographic_parity, mean_disparate_impact, mean_equalized_odds = 0, 0, 0, 0
            for fold, (train_idx, valid_idx) in enumerate(kfold.split(train)):

                # Set a seed for random number generation
                random.seed(SEED)
                np.random.seed(SEED)
                torch.manual_seed(SEED)

                logger.info(f"Fold {fold + 1}")
                train = dataset.iloc[train_idx]
                valid = dataset.iloc[valid_idx]
                # We use the same data loader of Cho's method
                fairness_dataset = FairnessChoDataset(train, valid, test)
                # This method performs also the scaling of the data
                fairness_dataset.prepare_ndarray(IDX)
                input_dim = fairness_dataset.XZ_train.shape[1]

                # Create a classifier model
                net = Classifier(n_layers=2, n_inputs=input_dim, n_hidden_units=NEURONS_PER_LAYER)
                net = net.to(DEVICE)

                # Set an optimizer
                optimizer = optim.Adam(net.parameters())
                lr_scheduler = None

                test_sol = 1e-3
                x_appro = torch.arange(test_sol, 1 - test_sol, test_sol).to(DEVICE)
                KDE_FAIR = KDE_fair(x_appro)
                penalty = KDE_FAIR.forward

                # Fair classifier training
                train_datasets, valid_datasets, test_datasets = fairness_dataset.get_dataset_in_tensor()
                _, y_train, z_train, x_train = train_datasets
                _, y_valid, z_valid, x_valid = valid_datasets
                _, y_test, z_test, x_test = test_datasets
                train_dataset = TensorDataset(x_train, y_train, z_train)
                dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
                y_pred = regularized_learning(dataloader, x_valid, y_valid, z_valid, x_test, y_test, z_test, net, penalty, DEVICE, JIANG_LAMBDA, DATA_LOSS, EPOCHS)

                # Compute metrics
                # Round to the nearest integer
                y_pred = np.rint(y_pred)
                accuracy = accuracy_score(fairness_dataset.Y_test, y_pred)
                logger.info(f"Test accuracy: {accuracy:.4f}")
                y_pred = y_pred.detach().cpu().numpy()
                mean_accuracy += accuracy
                mean_demographic_parity += is_demographic_parity(fairness_dataset.Z_test, y_pred, continuous=CONTINUOUS)
                mean_disparate_impact += is_disparate_impact(fairness_dataset.Z_test, y_pred, continuous=CONTINUOUS)
                mean_equalized_odds += is_equalized_odds(fairness_dataset.Z_test, fairness_dataset.Y_test, y_pred, continuous=CONTINUOUS)

            mean_accuracy /= K
            mean_demographic_parity /= K
            mean_disparate_impact /= K
            mean_equalized_odds /= K
            logger.info(f"Mean accuracy: {mean_accuracy:0.4f}")
            logger.info(f"Mean demographic parity: {mean_demographic_parity:0.4f}")
            logger.info(f"Mean disparate impact: {mean_disparate_impact:0.4f}")
            logger.info(f"Mean equalized odds: {mean_equalized_odds:0.4f}")
