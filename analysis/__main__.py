import os

from analysis import perform_analysis
from experiments.configuration import PATH as CONFIG_PATH

if __name__ == "__main__":
    # use all the yml configuration files in the configuration folder
    configuration_files = [f for f in os.listdir(CONFIG_PATH) if f.endswith(".yml")]
    perform_analysis(configuration_files)
