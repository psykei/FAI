from __future__ import annotations
import pandas as pd
from sklearn.model_selection import train_test_split
from dataset import PATH as DATASET_PATH

DEFAULT_SEED = 0
DEFAULT_TEST_SIZE = 0.2
DEFAULT_VALIDATION_SIZE = 0.2
DEFAULT_ADULT_TRAIN_SET_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"  # https://archive.ics.uci.edu/static/public/2/adult.zip
DEFAULT_ADULT_TEST_SET_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test"


def _create_cache_directory():
    if not (DATASET_PATH / "cache").exists():
        (DATASET_PATH / "cache").mkdir()


class AdultLoader:
    class AdultProcessor:
        def __init__(self, seed: int = DEFAULT_SEED):
            self.seed = seed

        def split(
            self,
            df: pd.DataFrame,
            validation_size: float = DEFAULT_VALIDATION_SIZE,
            test_size: float = DEFAULT_TEST_SIZE,
            validation: bool = True
        ) -> [pd.DataFrame, pd.DataFrame, pd.DataFrame] or [pd.DataFrame, pd.DataFrame]:
            train_df, test_df = train_test_split(
                df, test_size=test_size, stratify=df["income"], random_state=self.seed
            )
            if not validation:
                return train_df, test_df
            else:
                train_df, val_df = train_test_split(
                    train_df,
                    test_size=validation_size,
                    stratify=train_df["income"],
                    random_state=self.seed,
                )
                return train_df, val_df, test_df

        def setup(self, df: pd.DataFrame) -> pd.DataFrame:
            df.income = df.income.apply(
                lambda x: 0 if x.replace(" ", "") in ("<=50K", "<=50K.") else 1
            )
            for column in AdultLoader.duplicate:
                df.drop([column], axis=1, inplace=True)
            for column in AdultLoader.categorical:
                df = pd.concat([df, pd.get_dummies(df[column], prefix=column)], axis=1)
                df.drop([column], axis=1, inplace=True)
            df["Sex"] = df["Sex"].apply(
                lambda x: 0
                if x in ["Male", " Male", "Male ", " Male ", " Male."]
                else 1
            )
            # Boolean to float
            df = df.astype(float)
            output = df.pop("income")
            df["income"] = output
            return df

    filename = "adult.csv"
    columns = [
        "Age",
        "WorkClass",
        "Fnlwgt",
        "Education",
        "EducationNumeric",
        "MaritalStatus",
        "Occupation",
        "Relationship",
        "Ethnicity",
        "Sex",
        "CapitalGain",
        "CapitalLoss",
        "HoursPerWeek",
        "NativeCountry",
        "income",
    ]
    duplicate = ["Education"]
    categorical = [
        "WorkClass",
        "MaritalStatus",
        "Occupation",
        "Relationship",
        "Ethnicity",
        "NativeCountry",
    ]
    processor = AdultProcessor()

    def __init__(self, path: str = DEFAULT_ADULT_TRAIN_SET_URL):
        self.path = path

    def load(self, url: str = None, skiprows: int = 0) -> pd.DataFrame:
        if url is None:
            url = self.path
        _create_cache_directory()
        cache_file = DATASET_PATH / "cache" / (url.split("/")[-1] + ".csv")
        if cache_file.exists():
            return pd.read_csv(cache_file)
        else:
            df = pd.read_csv(url, skipinitialspace=True, skiprows=skiprows, header=None)
            df.columns = self.columns
            df.to_csv(cache_file, index=False)
            return df

    def load_all(self) -> [pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        df_train = self.load()
        df_test = self.load(DEFAULT_ADULT_TEST_SET_URL, skiprows=1)
        return pd.concat([df_train, df_test], axis=0)

    def load_preprocessed(self, all_datasets: bool = False) -> pd.DataFrame:
        if all_datasets:
            df = self.load_all()
        else:
            df = self.load()
        return self.processor.setup(df)

    def load_preprocessed_split(self, validation: bool = True, all_datasets: bool = False) -> [pd.DataFrame, pd.DataFrame, pd.DataFrame] or [pd.DataFrame, pd.DataFrame]:
        if all_datasets:
            df = self.load_preprocessed(all_datasets=True)
        else:
            df = self.load_preprocessed()
        return self.processor.split(df, validation=validation)
