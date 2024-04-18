from pathlib import Path

PATH = Path(__file__).parents[0]


def create_cache_directory():
    if not (PATH / "cache").exists():
        (PATH / "cache").mkdir()


def get_feature_data_type(dataset_name: str, index: int) -> str:
    if dataset_name == "adult":
        if index in [7, 8]:
            return "discrete"
        elif index == 0:
            return "continuous"
        else:
            return "unknown"
    elif dataset_name == "compas":
        return "unknown"


def get_feature_name(dataset_name: str, index: int) -> str:
    if dataset_name == "adult":
        if index == 0:
            return "age"
        elif index == 7:
            return "ethnicity"
        elif index == 8:
            return "sex"
        else:
            return "unknown"
    elif dataset_name == "compas":
        return "unknown"
