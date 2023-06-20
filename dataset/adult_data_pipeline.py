from __future__ import annotations
import pandas as pd
from sklearn.model_selection import train_test_split
from dataset import PATH as DATASET_PATH

DEFAULT_SEED = 0
DEFAULT_TEST_SIZE = 0.2
DEFAULT_VALIDATION_SIZE = 0.2
DEFAULT_ADULT_DATASET_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"  # https://archive.ics.uci.edu/static/public/2/adult.zip


def _create_cache_directory():
    if not (DATASET_PATH / "cache").exists():
        (DATASET_PATH / "cache").mkdir()


class AdultLoader:
    class AdultProcessor:

        def __init__(self, seed: int = DEFAULT_SEED):
            self.seed = seed

        def split(self, df: pd.DataFrame, validation_size: float = DEFAULT_VALIDATION_SIZE,
                  test_size: float = DEFAULT_TEST_SIZE, ) -> [pd.DataFrame, pd.DataFrame, pd.DataFrame]:
            train_df, test_df = train_test_split(df, test_size=test_size, stratify=df['income'], random_state=self.seed)
            train_df, val_df = train_test_split(train_df, test_size=validation_size, stratify=train_df['income'],
                                                random_state=self.seed)
            return train_df, val_df, test_df

        def setup(self, df: pd.DataFrame) -> pd.DataFrame:
            df.income = df.income.apply(
                lambda x: 0 if x.replace(" ", "") in ('<=50K', "<=50K.") else 1)
            for column in AdultLoader.duplicate:
                df.drop([column], axis=1, inplace=True)
            for column in AdultLoader.categorical:
                df = pd.concat([df, pd.get_dummies(df[column], prefix=column)], axis=1)
                df.drop([column], axis=1, inplace=True)
            df['Sex'] = df['Sex'].apply(lambda x: 0 if x in ['Male', ' Male', 'Male ', ' Male ', ' Male.'] else 1)
            return df

    filename = "adult.csv"
    columns = ["Age", "WorkClass", "Fnlwgt", "Education", "EducationNumeric", "MaritalStatus",
               "Occupation", "Relationship", "Ethnicity", "Sex", "CapitalGain", "CapitalLoss",
               "HoursPerWeek", "NativeCountry", "income"]
    duplicate = ["Education"]
    categorical = ["WorkClass", "MaritalStatus", "Occupation", "Relationship", "Ethnicity", "NativeCountry"]
    processor = AdultProcessor()

    def __init__(self, path: str = DEFAULT_ADULT_DATASET_URL):
        self.path = path

    def load(self):
        _create_cache_directory()
        cache_file = DATASET_PATH / "cache" / self.filename
        if cache_file.exists():
            return pd.read_csv(cache_file)
        else:
            df = pd.read_csv(self.path, skipinitialspace=True)
            df.columns = self.columns
            df.to_csv(cache_file, index=False)
            return df

    def load_preprocessed(self) -> pd.DataFrame:
        df = self.load()
        return self.processor.setup(df)

    def load_preprocessed_split(self) -> [pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        df = self.load_preprocessed()
        return self.processor.split(df)
