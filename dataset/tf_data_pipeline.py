import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split

DEFAULT_SEED = 0
DEFAULT_TEST_SIZE = 0.2
DEFAULT_VALIDATION_SIZE = 0.2
DEFAULT_ADULT_DATASET_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"  # https://archive.ics.uci.edu/static/public/2/adult.zip


class AdultLoader:

    def __init__(self, path: str = DEFAULT_ADULT_DATASET_URL, seed: int = DEFAULT_SEED):
        self.path = path
        self.seed = seed

    def load(self):
        df = pd.read_csv(self.path, skipinitialspace=True)
        df.columns = ["Age", "WorkClass", "Fnlwgt", "Education", "EducationNumeric", "MaritalStatus",
                      "Occupation", "Relationship", "Ethnicity", "Sex", "CapitalGain", "CapitalLoss",
                      "HoursPerWeek", "NativeCountry", "income"]
        return df

    def split(self, df: pd.DataFrame, validation_size: float = DEFAULT_VALIDATION_SIZE,
              test_size: float = DEFAULT_TEST_SIZE, ) -> [pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        train_df, test_df = train_test_split(df, test_size=test_size, stratify=df['income'], random_state=self.seed)
        train_df, val_df = train_test_split(train_df, test_size=validation_size, stratify=train_df['income'],
                                            random_state=self.seed)
        return train_df, val_df, test_df


class Processor:

    def setup(self, df: pd.DataFrame) -> pd.DataFrame:
        df.drop(["Education"], axis=1, inplace=True)
        df.income = df.income.apply(
            lambda x: 0 if x.replace(" ", "") in ('<=50K', "<=50K.") else 1)
        df['Sex'] = df['Sex'].apply(lambda x: 0 if x in ['Male', ' Male', 'Male ', ' Male ', ' Male.'] else 1)
        return df