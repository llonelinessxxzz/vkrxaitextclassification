from pathlib import Path
import pandas as pd

from config import INTERIM_DATA_DIR


def load_split(split_name: str) -> pd.DataFrame:
    valid_splits = {"train", "val", "test"}
    if split_name not in valid_splits:
        raise ValueError(f"split_name must be one of {valid_splits}")

    file_path = INTERIM_DATA_DIR / f"{split_name}.csv"
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    df = pd.read_csv(file_path)
    return df


def get_train_data() -> pd.DataFrame:
    return load_split("train")


def get_val_data() -> pd.DataFrame:
    return load_split("val")


def get_test_data() -> pd.DataFrame:
    return load_split("test")


def main():
    train_df = get_train_data()
    val_df = get_val_data()
    test_df = get_test_data()

    print("Train:", train_df.shape)
    print("Validation:", val_df.shape)
    print("Test:", test_df.shape)

    print("\nTrain sample:")
    print(train_df.head())


if __name__ == "__main__":
    main()