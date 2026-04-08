import pandas as pd
import matplotlib.pyplot as plt

from config import FIGURES_DIR, GENERATED_DIR


ERROR_SUMMARY_PATH = GENERATED_DIR / "error_summary_lstm.csv"
MISCLASSIFIED_PATH = GENERATED_DIR / "misclassified_examples_lstm.csv"
TAXONOMY_SUMMARY_PATH = GENERATED_DIR / "taxonomy_summary_lstm.csv"
TAXONOMY_PATH = GENERATED_DIR / "taxonomy_lstm.csv"

ERROR_PAIRS_FIG_PATH = FIGURES_DIR / "error_pairs_lstm.png"
TAXONOMY_FIG_PATH = FIGURES_DIR / "taxonomy_distribution_lstm.png"
ERROR_LENGTH_HIST_FIG_PATH = FIGURES_DIR / "error_length_hist_lstm.png"
ERROR_LENGTH_BOX_FIG_PATH = FIGURES_DIR / "error_length_boxplot_lstm.png"


def plot_error_pairs(error_summary_df: pd.DataFrame) -> None:
    if error_summary_df.empty:
        print("error_summary_lstm.csv is empty. Skipping error pairs plot.")
        return

    labels = [
        f"{row['true_label']} → {row['pred_label']}"
        for _, row in error_summary_df.iterrows()
    ]
    counts = error_summary_df["count"].tolist()

    plt.figure(figsize=(10, 6))
    plt.bar(labels, counts)
    plt.xlabel("Error pair")
    plt.ylabel("Count")
    plt.title("Distribution of Misclassification Pairs - LSTM")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(ERROR_PAIRS_FIG_PATH, dpi=300, bbox_inches="tight")
    plt.close()


def plot_taxonomy_distribution(taxonomy_summary_df: pd.DataFrame) -> None:
    if taxonomy_summary_df.empty:
        print("taxonomy_summary_lstm.csv is empty. Skipping taxonomy plot.")
        return

    labels = taxonomy_summary_df["error_type"].tolist()
    counts = taxonomy_summary_df["count"].tolist()

    plt.figure(figsize=(10, 6))
    plt.bar(labels, counts)
    plt.xlabel("Error type")
    plt.ylabel("Count")
    plt.title("Error Taxonomy Distribution - LSTM")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(TAXONOMY_FIG_PATH, dpi=300, bbox_inches="tight")
    plt.close()


def plot_error_length_histogram(errors_df: pd.DataFrame) -> None:
    if errors_df.empty:
        print("misclassified_examples_lstm.csv is empty. Skipping histogram.")
        return

    plt.figure(figsize=(9, 5))
    plt.hist(errors_df["text_length_tokens"], bins=30)
    plt.xlabel("Text length (tokens)")
    plt.ylabel("Count")
    plt.title("Distribution of Misclassified Text Lengths - LSTM")
    plt.tight_layout()
    plt.savefig(ERROR_LENGTH_HIST_FIG_PATH, dpi=300, bbox_inches="tight")
    plt.close()


def plot_error_length_boxplot(taxonomy_df: pd.DataFrame) -> None:
    if taxonomy_df.empty:
        print("taxonomy_lstm.csv is empty. Skipping boxplot.")
        return

    grouped = (
        taxonomy_df.groupby("error_type")["text_length_tokens"]
        .apply(list)
        .to_dict()
    )

    if not grouped:
        print("No grouped taxonomy data found. Skipping boxplot.")
        return

    labels = list(grouped.keys())
    values = list(grouped.values())

    plt.figure(figsize=(11, 6))
    plt.boxplot(values, tick_labels=labels, vert=True)
    plt.xlabel("Error type")
    plt.ylabel("Text length (tokens)")
    plt.title("Text Length by Error Type - LSTM")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(ERROR_LENGTH_BOX_FIG_PATH, dpi=300, bbox_inches="tight")
    plt.close()


def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    error_summary_df = pd.read_csv(ERROR_SUMMARY_PATH)
    errors_df = pd.read_csv(MISCLASSIFIED_PATH)
    taxonomy_summary_df = pd.read_csv(TAXONOMY_SUMMARY_PATH)
    taxonomy_df = pd.read_csv(TAXONOMY_PATH)

    plot_error_pairs(error_summary_df)
    plot_taxonomy_distribution(taxonomy_summary_df)
    plot_error_length_histogram(errors_df)
    plot_error_length_boxplot(taxonomy_df)

    print("Visualization completed.")
    print("\nSaved figures:")
    print(ERROR_PAIRS_FIG_PATH)
    print(TAXONOMY_FIG_PATH)
    print(ERROR_LENGTH_HIST_FIG_PATH)
    print(ERROR_LENGTH_BOX_FIG_PATH)


if __name__ == "__main__":
    main()