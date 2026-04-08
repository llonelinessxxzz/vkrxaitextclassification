import pandas as pd
import matplotlib.pyplot as plt

from config import FIGURES_DIR, GENERATED_DIR, TABLES_DIR


METRICS_LSTM_PATH = TABLES_DIR / "metrics_lstm.csv"
METRICS_RUBERT_PATH = TABLES_DIR / "metrics_rubert.csv"

REPORT_LSTM_PATH = TABLES_DIR / "classification_report_lstm.csv"
REPORT_RUBERT_PATH = TABLES_DIR / "classification_report_rubert.csv"

MISCLASSIFIED_LSTM_PATH = GENERATED_DIR / "misclassified_examples_lstm.csv"
MISCLASSIFIED_RUBERT_PATH = GENERATED_DIR / "misclassified_examples_rubert.csv"

ERROR_SUMMARY_LSTM_PATH = GENERATED_DIR / "error_summary_lstm.csv"
ERROR_SUMMARY_RUBERT_PATH = GENERATED_DIR / "error_summary_rubert.csv"

MODEL_COMPARISON_PATH = TABLES_DIR / "model_comparison.csv"
ERROR_COUNT_COMPARISON_PATH = TABLES_DIR / "error_count_comparison.csv"
ERROR_PAIRS_COMPARISON_PATH = TABLES_DIR / "error_pairs_comparison.csv"
CLASS_F1_COMPARISON_PATH = TABLES_DIR / "class_f1_comparison.csv"

METRICS_FIG_PATH = FIGURES_DIR / "model_metrics_comparison.png"
ERROR_COUNT_FIG_PATH = FIGURES_DIR / "model_error_count_comparison.png"
CLASS_F1_FIG_PATH = FIGURES_DIR / "class_f1_comparison.png"
ERROR_PAIRS_FIG_PATH = FIGURES_DIR / "error_pairs_comparison.png"


def load_metrics(path):
    return pd.read_csv(path)


def load_report(path):
    df = pd.read_csv(path, index_col=0)
    return df


def build_model_comparison():
    metrics_lstm = load_metrics(METRICS_LSTM_PATH).rename(columns={"value": "lstm"})
    metrics_rubert = load_metrics(METRICS_RUBERT_PATH).rename(columns={"value": "rubert"})

    comparison_df = metrics_lstm.merge(metrics_rubert, on="metric", how="inner")
    comparison_df["difference_rubert_minus_lstm"] = comparison_df["rubert"] - comparison_df["lstm"]
    comparison_df.to_csv(MODEL_COMPARISON_PATH, index=False)

    return comparison_df


def build_error_count_comparison():
    lstm_errors = pd.read_csv(MISCLASSIFIED_LSTM_PATH)
    rubert_errors = pd.read_csv(MISCLASSIFIED_RUBERT_PATH)

    error_count_df = pd.DataFrame(
        {
            "model": ["LSTM", "RuBERT"],
            "misclassified_count": [len(lstm_errors), len(rubert_errors)],
        }
    )
    error_count_df["correct_predictions"] = 18000 - error_count_df["misclassified_count"]
    error_count_df.to_csv(ERROR_COUNT_COMPARISON_PATH, index=False)

    return error_count_df


def build_error_pairs_comparison():
    lstm_pairs = pd.read_csv(ERROR_SUMMARY_LSTM_PATH).rename(columns={"count": "lstm_count"})
    rubert_pairs = pd.read_csv(ERROR_SUMMARY_RUBERT_PATH).rename(columns={"count": "rubert_count"})

    pairs_df = lstm_pairs.merge(
        rubert_pairs,
        on=["true_label", "pred_label"],
        how="outer",
    ).fillna(0)

    pairs_df["lstm_count"] = pairs_df["lstm_count"].astype(int)
    pairs_df["rubert_count"] = pairs_df["rubert_count"].astype(int)
    pairs_df["difference_rubert_minus_lstm"] = pairs_df["rubert_count"] - pairs_df["lstm_count"]
    pairs_df["pair"] = pairs_df["true_label"] + " → " + pairs_df["pred_label"]

    pairs_df = pairs_df.sort_values(
        by=["lstm_count", "rubert_count"],
        ascending=False,
    ).reset_index(drop=True)

    pairs_df.to_csv(ERROR_PAIRS_COMPARISON_PATH, index=False)
    return pairs_df


def build_class_f1_comparison():
    report_lstm = load_report(REPORT_LSTM_PATH)
    report_rubert = load_report(REPORT_RUBERT_PATH)

    rows = ["negative", "neutral", "positive", "macro avg", "weighted avg"]

    class_f1_df = pd.DataFrame(
        {
            "class_or_avg": rows,
            "lstm_f1": [report_lstm.loc[row, "f1-score"] for row in rows],
            "rubert_f1": [report_rubert.loc[row, "f1-score"] for row in rows],
        }
    )
    class_f1_df["difference_rubert_minus_lstm"] = class_f1_df["rubert_f1"] - class_f1_df["lstm_f1"]
    class_f1_df.to_csv(CLASS_F1_COMPARISON_PATH, index=False)

    return class_f1_df


def plot_metrics_comparison(comparison_df):
    plot_df = comparison_df[
        comparison_df["metric"].isin(
            ["accuracy", "precision_macro", "recall_macro", "f1_macro"]
        )
    ].copy()

    x = range(len(plot_df))
    width = 0.35

    plt.figure(figsize=(10, 6))
    plt.bar([i - width / 2 for i in x], plot_df["lstm"], width=width, label="LSTM")
    plt.bar([i + width / 2 for i in x], plot_df["rubert"], width=width, label="RuBERT")

    plt.xticks(list(x), plot_df["metric"], rotation=20, ha="right")
    plt.ylabel("Score")
    plt.title("Main Metrics Comparison: LSTM vs RuBERT")
    plt.legend()
    plt.tight_layout()
    plt.savefig(METRICS_FIG_PATH, dpi=300, bbox_inches="tight")
    plt.close()


def plot_error_count_comparison(error_count_df):
    plt.figure(figsize=(7, 5))
    plt.bar(error_count_df["model"], error_count_df["misclassified_count"])
    plt.ylabel("Number of misclassified examples")
    plt.title("Misclassified Examples: LSTM vs RuBERT")

    for idx, value in enumerate(error_count_df["misclassified_count"]):
        plt.text(idx, value, str(value), ha="center", va="bottom")

    plt.tight_layout()
    plt.savefig(ERROR_COUNT_FIG_PATH, dpi=300, bbox_inches="tight")
    plt.close()


def plot_class_f1_comparison(class_f1_df):
    x = range(len(class_f1_df))
    width = 0.35

    plt.figure(figsize=(10, 6))
    plt.bar([i - width / 2 for i in x], class_f1_df["lstm_f1"], width=width, label="LSTM")
    plt.bar([i + width / 2 for i in x], class_f1_df["rubert_f1"], width=width, label="RuBERT")

    plt.xticks(list(x), class_f1_df["class_or_avg"], rotation=20, ha="right")
    plt.ylabel("F1-score")
    plt.title("F1-score by Class: LSTM vs RuBERT")
    plt.legend()
    plt.tight_layout()
    plt.savefig(CLASS_F1_FIG_PATH, dpi=300, bbox_inches="tight")
    plt.close()


def plot_error_pairs_comparison(pairs_df, top_n=6):
    plot_df = pairs_df.head(top_n).copy()

    x = range(len(plot_df))
    width = 0.35

    plt.figure(figsize=(11, 6))
    plt.bar([i - width / 2 for i in x], plot_df["lstm_count"], width=width, label="LSTM")
    plt.bar([i + width / 2 for i in x], plot_df["rubert_count"], width=width, label="RuBERT")

    plt.xticks(list(x), plot_df["pair"], rotation=25, ha="right")
    plt.ylabel("Count")
    plt.title("Top Misclassification Pairs: LSTM vs RuBERT")
    plt.legend()
    plt.tight_layout()
    plt.savefig(ERROR_PAIRS_FIG_PATH, dpi=300, bbox_inches="tight")
    plt.close()


def main():
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    comparison_df = build_model_comparison()
    error_count_df = build_error_count_comparison()
    pairs_df = build_error_pairs_comparison()
    class_f1_df = build_class_f1_comparison()

    plot_metrics_comparison(comparison_df)
    plot_error_count_comparison(error_count_df)
    plot_class_f1_comparison(class_f1_df)
    plot_error_pairs_comparison(pairs_df, top_n=6)

    print("Model comparison completed.\n")

    print("Main metrics comparison:")
    print(comparison_df)

    print("\nError count comparison:")
    print(error_count_df)

    print("\nClass F1 comparison:")
    print(class_f1_df)

    print("\nTop error pairs comparison:")
    print(pairs_df.head(10)[["pair", "lstm_count", "rubert_count", "difference_rubert_minus_lstm"]])

    print("\nSaved files:")
    print(MODEL_COMPARISON_PATH)
    print(ERROR_COUNT_COMPARISON_PATH)
    print(ERROR_PAIRS_COMPARISON_PATH)
    print(CLASS_F1_COMPARISON_PATH)
    print(METRICS_FIG_PATH)
    print(ERROR_COUNT_FIG_PATH)
    print(CLASS_F1_FIG_PATH)
    print(ERROR_PAIRS_FIG_PATH)


if __name__ == "__main__":
    main()