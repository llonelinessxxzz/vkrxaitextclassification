from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
TABLES_DIR = REPORTS_DIR / "tables"
GENERATED_DIR = REPORTS_DIR / "generated"

CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"

RAW_DATA_FILE = RAW_DATA_DIR / "data.csv"

TEXT_COLUMN = "review"
LABEL_COLUMN = "sentiment"

LABEL_MAPPING = {
    "negative": 0,
    "neutral": 1,
    "positive": 2,
    "neautral": 1
}

RANDOM_STATE = 42
TEST_SIZE = 0.2
VAL_SIZE = 0.1