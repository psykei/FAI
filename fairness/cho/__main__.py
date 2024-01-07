import hashlib
import os
import random
import numpy as np
from sklearn.metrics import accuracy_score
from torch import optim
from configuration import *
from dataset.cho_data_pipeline import FairnessChoDataset
from fairness.cho import PATH as CHO_PATH, Classifier, train_fair_classifier
from fairness.metric import demographic_parity, equalized_odds, disparate_impact

for IDX in IDXS:
    CONTINUOUS = True if IDX == 0 else False
    for metric in CHO_METRICS:
        for CHO_LAMBDA in cho_lambdas(IDX, metric):
            idf = "_".join(
                [
                    str(x)
                    for x in [
                        SEED,
                        K,
                        EPOCHS,
                        BATCH_SIZE,
                        NEURONS_PER_LAYER,
                        IDX,
                        metric,
                        CHO_LAMBDA,
                    ]
                ]
            )
            if not ONE_HOT:
                idf += "_" + str(ONE_HOT)
            filename = (
                str(CHO_PATH)
                + os.sep
                + LOG
                + os.sep
                + hashlib.md5(str(idf).encode()).hexdigest()
                + ".txt"
            )
            if not os.path.exists(filename):
                dataset, train, test, kfold = initialize_experiment(
                    filename, metric, IDX, CHO_LAMBDA, preprocess=False
                )
                (
                    mean_accuracy,
                    mean_precision,
                    mean_recall,
                    mean_f1,
                    mean_demographic_parity,
                    mean_disparate_impact,
                    mean_equalized_odds,
                ) = (0, 0, 0, 0, 0, 0, 0)
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
                    net = Classifier(
                        n_layers=len(NEURONS_PER_LAYER) + 2,
                        n_inputs=input_dim,
                        n_hidden_units=NEURONS_PER_LAYER,
                    )
                    net = net.to(DEVICE)

                    # Set an optimizer
                    optimizer = optim.Adam(net.parameters())
                    lr_scheduler = None

                    # Fair classifier training
                    y_pred = train_fair_classifier(
                        dataset=fairness_dataset,
                        net=net,
                        optimizer=optimizer,
                        lr_scheduler=lr_scheduler,
                        fairness=metric,
                        lambda_=CHO_LAMBDA,
                        h=CHO_H,
                        delta=CHO_DELTA,
                        device=DEVICE,
                        n_epochs=EPOCHS,
                        batch_size=BATCH_SIZE,
                    )

                    # Compute metrics
                    # Round to the nearest integer
                    y_pred_binary = np.rint(y_pred)
                    tp = np.sum(np.logical_and(y_pred_binary == 1, test.iloc[:, -1].to_numpy() == 1))
                    tn = np.sum(np.logical_and(y_pred_binary == 0, test.iloc[:, -1].to_numpy() == 0))
                    fp = np.sum(np.logical_and(y_pred_binary == 1, test.iloc[:, -1].to_numpy() == 0))
                    fn = np.sum(np.logical_and(y_pred_binary == 0, test.iloc[:, -1].to_numpy() == 1))
                    accuracy = accuracy_score(test.iloc[:, -1].to_numpy(), y_pred_binary)
                    precision = tp / (tp + fp)
                    recall = tp / (tp + fn)
                    f1 = 2 * (precision * recall) / (precision + recall)
                    logger.info(f"Test accuracy: {accuracy:.4f}")
                    logger.info(f"Test precision: {precision:.4f}")
                    logger.info(f"Test recall: {recall:.4f}")
                    logger.info(f"Test F1: {f1:.4f}")
                    mean_accuracy += accuracy
                    mean_precision += precision
                    mean_recall += recall
                    mean_f1 += f1
                    mean_demographic_parity += demographic_parity(
                        fairness_dataset.Z_test, y_pred, continuous=CONTINUOUS
                    )
                    mean_disparate_impact += disparate_impact(
                        fairness_dataset.Z_test, y_pred, continuous=CONTINUOUS
                    )
                    mean_equalized_odds += equalized_odds(
                        fairness_dataset.Z_test,
                        fairness_dataset.Y_test,
                        y_pred,
                        continuous=CONTINUOUS,
                    )

                mean_accuracy /= K
                mean_precision /= K
                mean_recall /= K
                mean_f1 /= K
                mean_demographic_parity /= K
                mean_disparate_impact /= K
                mean_equalized_odds /= K
                logger.info(f"Mean accuracy: {mean_accuracy:0.4f}")
                logger.info(f"Mean precision: {mean_precision:0.4f}")
                logger.info(f"Mean recall: {mean_recall:0.4f}")
                logger.info(f"Mean F1: {mean_f1:0.4f}")
                logger.info(f"Mean demographic parity: {mean_demographic_parity:0.4f}")
                logger.info(f"Mean disparate impact: {mean_disparate_impact:0.4f}")
                logger.info(f"Mean equalized odds: {mean_equalized_odds:0.4f}")
