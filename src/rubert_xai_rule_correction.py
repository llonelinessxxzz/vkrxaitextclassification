import re
from itertools import product

import matplotlib.pyplot as plt
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from torch.utils.data import DataLoader, Dataset

from config import CHECKPOINTS_DIR, FIGURES_DIR, GENERATED_DIR, RANDOM_STATE, TABLES_DIR
from dataset import get_test_data, get_val_data
from rubert_model import get_rubert_model, get_rubert_tokenizer


MODEL_DIR = CHECKPOINTS_DIR / "rubert_best"
OUT_DIR = GENERATED_DIR / "rubert_xai_rule_correction"

MAX_LEN = 192
BATCH_SIZE = 16
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLASS_NAMES = ["negative", "neutral", "positive"]
ID_TO_LABEL = {0: "negative", 1: "neutral", 2: "positive"}
LABEL_TO_ID = {label: idx for idx, label in ID_TO_LABEL.items()}

NEGATIVE_RULE_PATTERNS = [
    r"\bне\s+совет",
    r"\bне\s+рекоменд",
    r"\bне\s+понрав",
    r"\bне\s+подош",
    r"\bне\s+стоит",
    r"\bне\s+устро",
]
NEUTRAL_RULE_PATTERNS = [
    r"\bне\s+лучш",
    r"\bне\s+идеал",
    r"\bне\s+очень",
]
POSITIVE_RULE_PATTERNS = [
    r"\bне\s+плох",
    r"\bнеплох",
    r"\bне\s+ужас",
    r"\bне\s+разочаров",
]


torch.manual_seed(RANDOM_STATE)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_STATE)


class RuBERTDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer):
        self.texts = df["review"].astype(str).tolist()
        self.labels = df["label_id"].astype(int).tolist()
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx: int):
        encoding = self.tokenizer(
            self.texts[idx],
            add_special_tokens=True,
            max_length=MAX_LEN,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["review"] = df["review"].fillna("").astype(str)
    df["label_id"] = df["label_id"].astype(int)
    df["sentiment"] = df["label_id"].map(ID_TO_LABEL)
    return df


def predict_dataframe(model, tokenizer, df: pd.DataFrame) -> pd.DataFrame:
    model.eval()
    loader = DataLoader(RuBERTDataset(df, tokenizer), batch_size=BATCH_SIZE, shuffle=False)
    preds = []
    confidences = []
    probs_all = []

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            probs = torch.softmax(outputs.logits, dim=1).cpu()
            preds.extend(torch.argmax(probs, dim=1).tolist())
            confidences.extend(torch.max(probs, dim=1).values.tolist())
            probs_all.extend(probs.tolist())

    pred_df = df.copy().reset_index(drop=True)
    pred_df["original_pred_label_id"] = preds
    pred_df["original_pred_label"] = pred_df["original_pred_label_id"].map(ID_TO_LABEL)
    pred_df["original_confidence"] = confidences
    for idx, label in ID_TO_LABEL.items():
        pred_df[f"proba_{label}"] = [row[idx] for row in probs_all]
    return pred_df


def contains_pattern(text: str, patterns: list[str]) -> bool:
    return any(re.search(pattern, str(text).lower()) for pattern in patterns)


def add_rule_flags(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["rule_negative_match"] = df["review"].apply(lambda text: contains_pattern(text, NEGATIVE_RULE_PATTERNS))
    df["rule_neutral_match"] = df["review"].apply(lambda text: contains_pattern(text, NEUTRAL_RULE_PATTERNS))
    df["rule_positive_match"] = df["review"].apply(lambda text: contains_pattern(text, POSITIVE_RULE_PATTERNS))
    df["any_rule_match"] = df[
        ["rule_negative_match", "rule_neutral_match", "rule_positive_match"]
    ].any(axis=1)
    return df


def apply_rules(df: pd.DataFrame, threshold: float, positive_target: int, neutral_target: int) -> pd.Series:
    corrected = df["original_pred_label_id"].copy()
    low_confidence = df["original_confidence"] <= threshold

    negative_mask = df["rule_negative_match"] & low_confidence
    neutral_mask = df["rule_neutral_match"] & low_confidence
    positive_mask = df["rule_positive_match"] & low_confidence

    corrected.loc[negative_mask] = LABEL_TO_ID["negative"]
    corrected.loc[neutral_mask] = neutral_target
    corrected.loc[positive_mask] = positive_target
    return corrected


def score_predictions(y_true, y_pred) -> dict:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "precision_weighted": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "recall_weighted": recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
    }


def tune_rules_on_validation(val_pred_df: pd.DataFrame):
    rows = []
    thresholds = [0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]
    positive_targets = [LABEL_TO_ID["neutral"], LABEL_TO_ID["positive"]]
    neutral_targets = [LABEL_TO_ID["negative"], LABEL_TO_ID["neutral"]]

    for threshold, positive_target, neutral_target in product(thresholds, positive_targets, neutral_targets):
        corrected = apply_rules(val_pred_df, threshold, positive_target, neutral_target)
        metrics = score_predictions(val_pred_df["label_id"], corrected)
        rows.append(
            {
                "threshold": threshold,
                "positive_rule_target": ID_TO_LABEL[positive_target],
                "neutral_rule_target": ID_TO_LABEL[neutral_target],
                "changed_count": int((corrected != val_pred_df["original_pred_label_id"]).sum()),
                **metrics,
            }
        )

    tuning_df = pd.DataFrame(rows).sort_values(["f1_macro", "accuracy"], ascending=False)
    best = tuning_df.iloc[0]
    return best, tuning_df


def add_corrected_columns(df: pd.DataFrame, best_rule) -> pd.DataFrame:
    df = df.copy()
    positive_target = LABEL_TO_ID[best_rule["positive_rule_target"]]
    neutral_target = LABEL_TO_ID[best_rule["neutral_rule_target"]]
    corrected = apply_rules(df, best_rule["threshold"], positive_target, neutral_target)

    df["corrected_pred_label_id"] = corrected
    df["corrected_pred_label"] = df["corrected_pred_label_id"].map(ID_TO_LABEL)
    df["changed_prediction"] = df["corrected_pred_label_id"] != df["original_pred_label_id"]
    df["original_correct"] = df["label_id"] == df["original_pred_label_id"]
    df["corrected_correct"] = df["label_id"] == df["corrected_pred_label_id"]
    return df


def save_metrics(val_df: pd.DataFrame, test_df: pd.DataFrame):
    rows = []
    for split_name, df in [("val", val_df), ("test", test_df)]:
        rows.append(
            {
                "model": "rubert_original",
                "split": split_name,
                "subset": "all",
                **score_predictions(df["label_id"], df["original_pred_label_id"]),
            }
        )
        rows.append(
            {
                "model": "rubert_rule_corrected",
                "split": split_name,
                "subset": "all",
                **score_predictions(df["label_id"], df["corrected_pred_label_id"]),
            }
        )

        subset = df[df["any_rule_match"]]
        if not subset.empty:
            rows.append(
                {
                    "model": "rubert_original",
                    "split": split_name,
                    "subset": "rule_matched",
                    **score_predictions(subset["label_id"], subset["original_pred_label_id"]),
                }
            )
            rows.append(
                {
                    "model": "rubert_rule_corrected",
                    "split": split_name,
                    "subset": "rule_matched",
                    **score_predictions(subset["label_id"], subset["corrected_pred_label_id"]),
                }
            )

    metrics_df = pd.DataFrame(rows)
    metrics_path = TABLES_DIR / "metrics_rubert_xai_rule_correction.csv"
    metrics_df.to_csv(metrics_path, index=False)
    return metrics_df, metrics_path


def save_rule_summary(test_df: pd.DataFrame):
    rows = []
    for subset_name, mask in {
        "all_test": pd.Series(True, index=test_df.index),
        "rule_matched": test_df["any_rule_match"],
        "negative_rule": test_df["rule_negative_match"],
        "neutral_rule": test_df["rule_neutral_match"],
        "positive_rule": test_df["rule_positive_match"],
    }.items():
        part = test_df[mask]
        if part.empty:
            continue
        rows.append(
            {
                "subset": subset_name,
                "count": len(part),
                "changed_prediction_count": int(part["changed_prediction"].sum()),
                "fixed_count": int(((~part["original_correct"]) & part["corrected_correct"]).sum()),
                "new_error_count": int((part["original_correct"] & (~part["corrected_correct"])).sum()),
                "original_correct": int(part["original_correct"].sum()),
                "corrected_correct": int(part["corrected_correct"].sum()),
            }
        )

    summary_df = pd.DataFrame(rows)
    summary_path = TABLES_DIR / "xai_rule_correction_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    ax = summary_df.set_index("subset")[["fixed_count", "new_error_count"]].plot(kind="bar", figsize=(10, 5))
    ax.set_xlabel("Subset")
    ax.set_ylabel("Count")
    ax.set_title("Rule-Based Correction from XAI Error Analysis")
    ax.tick_params(axis="x", rotation=25)
    plt.tight_layout()
    fig_path = FIGURES_DIR / "xai_rule_correction_fixed_vs_new_errors.png"
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close()

    return summary_path, fig_path


def save_classification_report(test_df: pd.DataFrame):
    report_df = pd.DataFrame(
        classification_report(
            test_df["label_id"],
            test_df["corrected_pred_label_id"],
            target_names=CLASS_NAMES,
            output_dict=True,
            zero_division=0,
        )
    ).transpose()
    report_path = TABLES_DIR / "classification_report_rubert_xai_rule_correction.csv"
    report_df.to_csv(report_path)
    return report_path


def main():
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Using device: {DEVICE}")

    tokenizer = get_rubert_tokenizer(MODEL_DIR)
    model = get_rubert_model(MODEL_DIR).to(DEVICE)

    val_df = clean_dataframe(get_val_data())
    test_df = clean_dataframe(get_test_data())

    print("Predicting validation and test with original RuBERT...")
    val_pred_df = add_rule_flags(predict_dataframe(model, tokenizer, val_df))
    test_pred_df = add_rule_flags(predict_dataframe(model, tokenizer, test_df))

    best_rule, tuning_df = tune_rules_on_validation(val_pred_df)
    tuning_path = OUT_DIR / "rule_tuning_on_validation.csv"
    tuning_df.to_csv(tuning_path, index=False)

    val_corrected_df = add_corrected_columns(val_pred_df, best_rule)
    test_corrected_df = add_corrected_columns(test_pred_df, best_rule)

    val_corrected_df.to_csv(OUT_DIR / "predictions_val_rubert_xai_rule_correction.csv", index=False)
    test_corrected_df.to_csv(OUT_DIR / "predictions_test_rubert_xai_rule_correction.csv", index=False)

    changed_df = test_corrected_df[test_corrected_df["changed_prediction"]].copy()
    fixed_df = test_corrected_df[(~test_corrected_df["original_correct"]) & test_corrected_df["corrected_correct"]].copy()
    new_errors_df = test_corrected_df[
        test_corrected_df["original_correct"] & (~test_corrected_df["corrected_correct"])
    ].copy()

    changed_path = OUT_DIR / "changed_predictions_rubert_xai_rule_correction.csv"
    fixed_path = OUT_DIR / "fixed_examples_rubert_xai_rule_correction.csv"
    new_errors_path = OUT_DIR / "new_errors_rubert_xai_rule_correction.csv"
    changed_df.to_csv(changed_path, index=False)
    fixed_df.to_csv(fixed_path, index=False)
    new_errors_df.to_csv(new_errors_path, index=False)

    metrics_df, metrics_path = save_metrics(val_corrected_df, test_corrected_df)
    report_path = save_classification_report(test_corrected_df)
    summary_path, fig_path = save_rule_summary(test_corrected_df)

    best_rule_path = TABLES_DIR / "best_rule_config_rubert_xai_rule_correction.csv"
    pd.DataFrame([best_rule]).to_csv(best_rule_path, index=False)

    print("\nBest validation rule:")
    print(best_rule)
    print("\nMetrics:")
    print(metrics_df)

    print("\nSaved files:")
    print(tuning_path)
    print(best_rule_path)
    print(metrics_path)
    print(report_path)
    print(summary_path)
    print(fig_path)
    print(changed_path)
    print(fixed_path)
    print(new_errors_path)


if __name__ == "__main__":
    main()
