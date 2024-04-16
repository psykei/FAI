from pathlib import Path

PATH = Path(__file__).parents[0]


def create_cache_directory():
    if not (PATH / "cache").exists():
        (PATH / "cache").mkdir()
