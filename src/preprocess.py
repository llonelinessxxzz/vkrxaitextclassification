import pandas as pd
from sklearn.model_selection import train_test_split

from config import (
    RAW_DATA_FILE,
    INTERIM_DATA_DIR,
    TEXT_COLUMN,
    LABEL_COLUMN,
    LABEL_MAPPING,
    RANDOM_STATE,
    TEST_SIZE,
    VAL_SIZE,
)


def load_dataset() -> pd.DataFrame:
    df = pd.read_csv(RAW_DATA_FILE, sep="\t")
    print("Columns:", df.columns.tolist())
    print("Shape:", df.shape)
    return df


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df = df[[TEXT_COLUMN, LABEL_COLUMN]].dropna()
    df[TEXT_COLUMN] = df[TEXT_COLUMN].astype(str).str.strip()
    df[LABEL_COLUMN] = df[LABEL_COLUMN].astype(str).str.strip().str.lower()

    df = df[df[TEXT_COLUMN] != ""]
    df[LABEL_COLUMN] = df[LABEL_COLUMN].replace({"neautral": "neutral"})
    df["label_id"] = df[LABEL_COLUMN].map(LABEL_MAPPING)

    df = df.dropna(subset=["label_id"])
    df["label_id"] = df["label_id"].astype(int)

    return df


def split_dataset(df: pd.DataFrame):
    train_df, test_df = train_test_split(
        df,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=df["label_id"],
    )

    val_relative_size = VAL_SIZE / (1 - TEST_SIZE)

    train_df, val_df = train_test_split(
        train_df,
        test_size=val_relative_size,
        random_state=RANDOM_STATE,
        stratify=train_df["label_id"],
    )

    return train_df, val_df, test_df


def save_splits(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    INTERIM_DATA_DIR.mkdir(parents=True, exist_ok=True)

    train_df.to_csv(INTERIM_DATA_DIR / "train.csv", index=False)
    val_df.to_csv(INTERIM_DATA_DIR / "val.csv", index=False)
    test_df.to_csv(INTERIM_DATA_DIR / "test.csv", index=False)


def main():
    df = load_dataset()
    df = clean_dataset(df)

    train_df, val_df, test_df = split_dataset(df)
    save_splits(train_df, val_df, test_df)

    print("Dataset preprocessing completed.")
    print(f"Train size: {len(train_df)}")
    print(f"Validation size: {len(val_df)}")
    print(f"Test size: {len(test_df)}")


if __name__ == "__main__":
    main()