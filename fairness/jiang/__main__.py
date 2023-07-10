import hashlib
import os
import random
import numpy as np
from sklearn.metrics import accuracy_score
from torch import optim
from configuration import *
from dataset.cho_dataset_pipeline import FairnessChoDataset
from fairness.jiang import PATH as JIANG_PATH, Net, regularized_learning
from fairness.metric import is_demographic_parity, is_equalized_odds, is_disparate_impact

for metric in CHO_METRICS:
    for JIANG_LAMBDA in JIANG_LAMBDAS:
        idf = '_'.join([str(x) for x in [SEED, K, EPOCHS, BATCH_SIZE, NEURONS_PER_LAYER, IDX, metric, JIANG_LAMBDA]])
        if not ONE_HOT:
            idf += "_" + str(ONE_HOT)
        filename = str(JIANG_PATH) + os.sep + LOG + os.sep + hashlib.md5(str(idf).encode()).hexdigest() + ".txt"
        if not os.path.exists(filename):
            dataset, train, test, kfold = initialize_experiment(filename, metric, JIANG_LAMBDA, preprocess=False)
            mean_accuracy, mean_demographic_parity, mean_disparate_impact, mean_equalized_odds = 0, 0, 0, 0
            for fold, (train_idx, valid_idx) in enumerate(kfold.split(train)):

                # Set a seed for random number generation
                random.seed(SEED)
                np.random.seed(SEED)
                torch.manual_seed(SEED)

                logger.info(f"Fold {fold + 1}")
                train = dataset.iloc[train_idx]
                valid = dataset.iloc[valid_idx]
                fairness_dataset = FairnessChoDataset(train, valid, test)
                # This method performs also the scaling of the data
                fairness_dataset.prepare_ndarray(IDX)
                input_dim = fairness_dataset.XZ_train.shape[1]

                # Create a classifier model
                net = Net(input_size=input_dim, neurons_per_layer=NEURONS_PER_LAYER)
                net = net.to(DEVICE)

                # Set an optimizer
                optimizer = optim.Adam(net.parameters())
                lr_scheduler = None

                # Fair classifier training
                dataloader = DataLoader()
                y_pred = regularized_learning()

                # Compute metrics
                # Round to the nearest integer
                y_pred = np.rint(y_pred)
                accuracy = accuracy_score(fairness_dataset.Y_test, y_pred)
                logger.info(f"Test accuracy: {accuracy:.4f}")
                mean_accuracy += accuracy
                mean_demographic_parity += is_demographic_parity(fairness_dataset.Z_test, y_pred)
                mean_disparate_impact += is_disparate_impact(fairness_dataset.Z_test, y_pred)
                mean_equalized_odds += is_equalized_odds(fairness_dataset.Z_test, fairness_dataset.Y_test, y_pred)

            mean_accuracy /= K
            mean_demographic_parity /= K
            mean_disparate_impact /= K
            mean_equalized_odds /= K
            logger.info(f"Mean accuracy: {mean_accuracy:0.4f}")
            logger.info(f"Mean demographic parity: {mean_demographic_parity:0.4f}")
            logger.info(f"Mean disparate impact: {mean_disparate_impact:0.4f}")
            logger.info(f"Mean equalized odds: {mean_equalized_odds:0.4f}")
