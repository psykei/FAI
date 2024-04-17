from pathlib import Path
import yaml

PATH = Path(__file__).parents[0]
SEED = 0
K = 5
EPOCHS = 5000
BATCH_SIZE = 500
NEURONS_PER_LAYER = [100, 50]
ADULT_PATIENCE = 10
COMPAS_PATIENCE = 10
VERBOSE = 0


def from_config_file_to_dict(config_file: str) -> dict:
    """
    Read the configuration file and return a dictionary.

    @param config_file: configuration file
    @return: dictionary
    """
    with open(config_file, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config
