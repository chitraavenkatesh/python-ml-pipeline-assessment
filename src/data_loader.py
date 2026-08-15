import pandas as pd


def load_data(file_path: str) -> pd.DataFrame:
    """
    Load a CSV dataset from the given file path.

    Args:
        file_path: Path to the CSV file.

    Returns:
        Loaded dataset as a pandas DataFrame.
    """
    return pd.read_csv(file_path)