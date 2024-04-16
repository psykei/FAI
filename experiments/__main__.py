import os
import random
import numpy as np
from dataset.adult_pytorch_data_pipeline import FairnessPyTorchDataset
from experiments import CACHE_PATH, TensorflowConditions, get_feature_data_type, PyTorchConditions, \
    PytorchNN, create_fully_connected_nn_tf, evaluate_predictions
from experiments.configuration import PATH as CONFIG_PATH, NEURONS_PER_LAYER, EPOCHS, BATCH_SIZE, K
from sklearn.model_selection import KFold
from tensorflow.python.framework.random_seed import set_seed
from experiments._logging import logger, enable_file_logging, LOG_INFO, disable_file_logging, exp_name, \
    log_experiment_setup
from tensorflow.python.compat.v2_compat import disable_v2_behavior
from tensorflow.python.framework.ops import disable_eager_execution
from tensorflow import device
from dataset.loader import load_dataset
from experiments.configuration import ADULT_PATIENCE, COMPAS_PATIENCE, from_config_file_to_dict
from fairness.cho import train_and_predict_cho_classifier
from fairness.fauci import train_and_predict_tf, create_fauci_network
from fairness.jiang import train_and_predict_jiang_classifier

if __name__ == '__main__':

    disable_v2_behavior()
    disable_eager_execution()

    # Set GPU number
    device('/physical_device:GPU:1')

    # Read the configuration files (only yaml files)
    configuration_files = [f for f in os.listdir(CONFIG_PATH) if f.endswith(".yml")]

    configurations = [from_config_file_to_dict(file_name) for file_name in configuration_files]

    for configuration in configurations:
        for setup in configuration:
            dataset = setup["dataset"]
            method = setup["method"]
            metric = setup["metric"]
            protected = setup["protected"]
            exp_seed = setup["seed"]
            lambda_info = setup["lambda"]
            max_lambda_values = lambda_info["max"]
            min_lambda_values = lambda_info["min"]
            step_lambda_values = lambda_info["step"]
            lambda_lists_of_values = [[round(value, 5) for value in np.arange(min_lambda_values[i], max_lambda_values[i], step_lambda_values[i])] for i in range(len(protected))]

            if dataset == "adult":
                patience = ADULT_PATIENCE
            elif dataset == "compas":
                patience = COMPAS_PATIENCE
            else:
                raise ValueError(f"Unknown dataset {dataset}")

            train, test = load_dataset(dataset)

            for i, feature in protected:
                lambda_values: list[float] = lambda_lists_of_values[i]
                protected_type = get_feature_data_type(dataset, feature)
                for lambda_value in lambda_values:
                    # K-fold cross validation with 5 folds
                    folds = KFold(n_splits=K, shuffle=True, random_state=exp_seed)
                    for exp_number, (train_idx, valid_idx) in enumerate(folds.split(train)):
                        file_name = exp_name(dataset, method, metric, feature, lambda_value, exp_number, exp_seed)
                        complete_file_name = CACHE_PATH / file_name
                        if os.path.exists(complete_file_name):
                            continue
                        enable_file_logging(complete_file_name, LOG_INFO)
                        log_experiment_setup(dataset, method, protected, metric, lambda_value, exp_number, exp_seed)
                        random.seed(exp_seed)
                        set_seed(exp_seed)

                        # Split the dataset into train and validation
                        train_data, valid_data = train.iloc[train_idx], train.iloc[valid_idx]

                        # Custom loss function construction
                        if method == "fauci":
                            callbacks = [TensorflowConditions(patience=patience, fairness_metric_name=metric)]
                            model = create_fully_connected_nn_tf(train.shape[1] - 1, 1, NEURONS_PER_LAYER)
                            model = create_fauci_network(model, protected, protected_type, metric, lambda_value)
                            predictions = train_and_predict_tf(model, train_data, valid_data, test, EPOCHS, BATCH_SIZE, callbacks, logger)
                            predictions = np.squeeze(predictions)
                            del model

                        elif method == "cho":
                            model = PytorchNN(train.shape[1] - 1, 1, NEURONS_PER_LAYER)
                            callbacks = PyTorchConditions(model, patience, metric, EPOCHS)
                            pt_dataset = FairnessPyTorchDataset(train_data, valid_data, test)
                            predictions = train_and_predict_cho_classifier(pt_dataset, model, metric, lambda_value, device, EPOCHS, BATCH_SIZE, callbacks)
                            predictions = predictions.detach().cpu().numpy().squeeze()
                            del model

                        elif method == "jiang":
                            model = PytorchNN(train.shape[1] - 1, 1, NEURONS_PER_LAYER)
                            callbacks = PyTorchConditions(model, patience, metric, EPOCHS)
                            pt_dataset = FairnessPyTorchDataset(train_data, valid_data, test)
                            predictions = train_and_predict_jiang_classifier(model, pt_dataset, device, lambda_value, EPOCHS, BATCH_SIZE, callbacks)

                        else:
                            raise ValueError(f"Unknown method {method}")

                        test_y = test.iloc[:, -1].reset_index(drop=True)
                        test_p = test.iloc[:, protected].reset_index(drop=True)
                        evaluate_predictions(test, predictions, test_y, logger)
                        disable_file_logging()
