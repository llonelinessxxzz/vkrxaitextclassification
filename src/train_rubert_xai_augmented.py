import random
import re
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import get_linear_schedule_with_warmup

from config import CHECKPOINTS_DIR, FIGURES_DIR, GENERATED_DIR, RANDOM_STATE, TABLES_DIR
from dataset import get_test_data, get_train_data, get_val_data
from rubert_model import get_rubert_model, get_rubert_tokenizer


SOURCE_MODEL_DIR = CHECKPOINTS_DIR / "rubert_best"
AUGMENTED_MODEL_DIR = CHECKPOINTS_DIR / "rubert_xai_augmented"

GENERATED_OUT_DIR = GENERATED_DIR / "rubert_xai_augmented"

MAX_LEN = 192
BATCH_SIZE = 16
NUM_EPOCHS = 1
LEARNING_RATE = 1e-5
WARMUP_RATIO = 0.1
MAX_VAL_ERROR_EXAMPLES = 900
SYNTHETIC_REPEAT = 18

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLASS_NAMES = ["negative", "neutral", "positive"]
ID_TO_LABEL = {0: "negative", 1: "neutral", 2: "positive"}
LABEL_TO_ID = {label: idx for idx, label in ID_TO_LABEL.items()}

NEGATION_RE = re.compile(r"\bне\s+\w+", flags=re.IGNORECASE)

POSITIVE_WORDS = [
    "плохой",
    "плохая",
    "плохое",
    "плохие",
    "ужасный",
    "ужасная",
    "ужасное",
    "разочаровал",
    "разочаровала",
]
NEGATIVE_WORDS = [
    "лучший",
    "лучшая",
    "лучшее",
    "идеальный",
    "идеальная",
    "идеально",
    "советую",
    "рекомендую",
    "понравился",
    "понравилась",
    "подошел",
    "подошла",
]


def set_seed(seed: int = RANDOM_STATE):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["review"] = df["review"].fillna("").astype(str)
    df["label_id"] = df["label_id"].astype(int)
    df["sentiment"] = df["label_id"].map(ID_TO_LABEL)
    return df


def has_negation_context(text: str) -> bool:
    return bool(NEGATION_RE.search(str(text).lower()))


def has_negated_positive_word(text: str) -> bool:
    text = str(text).lower()
    return any(f"не {word}" in text for word in NEGATIVE_WORDS)


def has_negated_negative_word(text: str) -> bool:
    text = str(text).lower()
    return any(f"не {word}" in text for word in POSITIVE_WORDS) or "неплох" in text


class RuBERTDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer, max_len: int = MAX_LEN):
        self.texts = df["review"].astype(str).tolist()
        self.labels = df["label_id"].astype(int).tolist()
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx: int):
        encoding = self.tokenizer(
            self.texts[idx],
            add_special_tokens=True,
            max_length=self.max_len,
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


def make_loader(df: pd.DataFrame, tokenizer, shuffle: bool = False) -> DataLoader:
    dataset = RuBERTDataset(df, tokenizer)
    return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=shuffle)


def predict_dataframe(model, tokenizer, df: pd.DataFrame) -> pd.DataFrame:
    model.eval()
    loader = make_loader(df, tokenizer, shuffle=False)

    preds = []
    confidences = []
    probs_all = []

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            probs = torch.softmax(outputs.logits, dim=1).cpu().numpy()
            batch_preds = probs.argmax(axis=1)

            preds.extend(batch_preds.tolist())
            confidences.extend(probs.max(axis=1).tolist())
            probs_all.extend(probs.tolist())

    out_df = df.copy().reset_index(drop=True)
    out_df["pred_label_id"] = preds
    out_df["true_label"] = out_df["label_id"].map(ID_TO_LABEL)
    out_df["pred_label"] = out_df["pred_label_id"].map(ID_TO_LABEL)
    out_df["confidence"] = confidences
    for idx, label in ID_TO_LABEL.items():
        out_df[f"proba_{label}"] = [row[idx] for row in probs_all]
    return out_df


def create_synthetic_negation_examples() -> pd.DataFrame:
    rows = [
        ("Товар не плохой, качество нормальное, покупкой довольна.", "positive"),
        ("Материал не плохой, за свою цену очень даже хороший.", "positive"),
        ("Заказ не разочаровал, все пришло быстро и аккуратно.", "positive"),
        ("Вещь не ужасная, выглядит хорошо и носить можно.", "positive"),
        ("Не идеально, но нормально, пользоваться можно.", "neutral"),
        ("Не лучший товар, но за эти деньги вполне нормально.", "neutral"),
        ("Не плохой и не отличный, обычное качество.", "neutral"),
        ("Не идеально, есть мелкие недочеты, но в целом нормально.", "neutral"),
        ("Не советую покупать, качество плохое.", "negative"),
        ("Не рекомендую товар, ожидания не оправдались.", "negative"),
        ("Не понравился материал, выглядит дешево.", "negative"),
        ("Не подошел размер и качество не устроило.", "negative"),
        ("Не лучший вариант, своих денег не стоит.", "negative"),
        ("Не идеально и носить неудобно, покупкой недовольна.", "negative"),
    ]

    augmented_rows = []
    for repeat_id in range(SYNTHETIC_REPEAT):
        for text, label in rows:
            augmented_rows.append(
                {
                    "review": text,
                    "sentiment": label,
                    "label_id": LABEL_TO_ID[label],
                    "augmentation_source": "synthetic_negation_template",
                    "augmentation_repeat": repeat_id + 1,
                }
            )

    return pd.DataFrame(augmented_rows)


def build_error_focused_training_data(train_df, val_predictions_df):
    val_errors = val_predictions_df[
        val_predictions_df["label_id"] != val_predictions_df["pred_label_id"]
    ].copy()
    val_errors["has_negation_context"] = val_errors["review"].apply(has_negation_context)
    val_errors["has_negated_positive_word"] = val_errors["review"].apply(has_negated_positive_word)
    val_errors["has_negated_negative_word"] = val_errors["review"].apply(has_negated_negative_word)

    negation_errors = val_errors[val_errors["has_negation_context"]].copy()
    other_high_conf_errors = val_errors.sort_values("confidence", ascending=False).head(300)

    selected_errors = pd.concat([negation_errors, other_high_conf_errors], ignore_index=True)
    selected_errors = selected_errors.drop_duplicates(subset=["review"]).head(MAX_VAL_ERROR_EXAMPLES)
    selected_errors["augmentation_source"] = "validation_error"
    selected_errors["augmentation_repeat"] = 1

    synthetic_df = create_synthetic_negation_examples()

    selected_cols = ["review", "sentiment", "label_id", "augmentation_source", "augmentation_repeat"]
    augmented_extra = pd.concat(
        [selected_errors[selected_cols], synthetic_df[selected_cols]],
        ignore_index=True,
    )

    train_augmented = pd.concat(
        [
            train_df.assign(augmentation_source="original_train", augmentation_repeat=0)[selected_cols],
            augmented_extra,
        ],
        ignore_index=True,
    )

    return train_augmented, selected_errors, synthetic_df, val_errors


def train_epoch(model, loader, optimizer, scheduler):
    model.train()
    total_loss = 0.0
    all_preds = []
    all_labels = []

    for batch in loader:
        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        labels = batch["labels"].to(DEVICE)

        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        all_preds.extend(outputs.logits.argmax(dim=1).detach().cpu().tolist())
        all_labels.extend(labels.detach().cpu().tolist())

    return total_loss / len(loader), accuracy_score(all_labels, all_preds)


def evaluate_loss_acc(model, loader):
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            total_loss += outputs.loss.item()
            all_preds.extend(outputs.logits.argmax(dim=1).cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    return total_loss / len(loader), accuracy_score(all_labels, all_preds)


def metric_row(y_true, y_pred, model_name: str, subset: str) -> dict:
    return {
        "model": model_name,
        "subset": subset,
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "precision_weighted": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "recall_weighted": recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
    }


def save_metrics_and_reports(base_test_pred, aug_test_pred):
    y_true = base_test_pred["label_id"].tolist()
    y_base = base_test_pred["pred_label_id"].tolist()
    y_aug = aug_test_pred["pred_label_id"].tolist()

    metrics_df = pd.DataFrame(
        [
            metric_row(y_true, y_base, "rubert_original", "test"),
            metric_row(y_true, y_aug, "rubert_xai_augmented", "test"),
        ]
    )

    negation_mask = aug_test_pred["review"].apply(has_negation_context)
    if negation_mask.any():
        metrics_df = pd.concat(
            [
                metrics_df,
                pd.DataFrame(
                    [
                        metric_row(
                            base_test_pred.loc[negation_mask, "label_id"],
                            base_test_pred.loc[negation_mask, "pred_label_id"],
                            "rubert_original",
                            "test_negation_subset",
                        ),
                        metric_row(
                            aug_test_pred.loc[negation_mask, "label_id"],
                            aug_test_pred.loc[negation_mask, "pred_label_id"],
                            "rubert_xai_augmented",
                            "test_negation_subset",
                        ),
                    ]
                ),
            ],
            ignore_index=True,
        )

    metrics_path = TABLES_DIR / "metrics_rubert_xai_augmented.csv"
    metrics_df.to_csv(metrics_path, index=False)

    report_df = pd.DataFrame(
        classification_report(
            y_true,
            y_aug,
            target_names=CLASS_NAMES,
            output_dict=True,
            zero_division=0,
        )
    ).transpose()
    report_path = TABLES_DIR / "classification_report_rubert_xai_augmented.csv"
    report_df.to_csv(report_path)

    return metrics_df, metrics_path, report_path


def save_confusion_matrix(pred_df):
    cm = confusion_matrix(pred_df["label_id"], pred_df["pred_label_id"], labels=[0, 1, 2])

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm)
    ax.set_xticks([0, 1, 2])
    ax.set_yticks([0, 1, 2])
    ax.set_xticklabels(CLASS_NAMES)
    ax.set_yticklabels(CLASS_NAMES)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title("Confusion Matrix - RuBERT XAI-Augmented")

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")

    fig.colorbar(im)
    plt.tight_layout()
    fig_path = FIGURES_DIR / "confusion_matrix_rubert_xai_augmented.png"
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close()
    return fig_path


def save_improvement_analysis(base_test_pred, aug_test_pred):
    comparison_df = base_test_pred[
        ["review", "sentiment", "label_id", "true_label", "pred_label_id", "pred_label", "confidence"]
    ].copy()
    comparison_df = comparison_df.rename(
        columns={
            "pred_label_id": "original_pred_label_id",
            "pred_label": "original_pred_label",
            "confidence": "original_confidence",
        }
    )
    comparison_df["augmented_pred_label_id"] = aug_test_pred["pred_label_id"]
    comparison_df["augmented_pred_label"] = aug_test_pred["pred_label"]
    comparison_df["augmented_confidence"] = aug_test_pred["confidence"]
    comparison_df["original_correct"] = comparison_df["label_id"] == comparison_df["original_pred_label_id"]
    comparison_df["augmented_correct"] = comparison_df["label_id"] == comparison_df["augmented_pred_label_id"]
    comparison_df["changed_prediction"] = (
        comparison_df["original_pred_label_id"] != comparison_df["augmented_pred_label_id"]
    )
    comparison_df["has_negation_context"] = comparison_df["review"].apply(has_negation_context)
    comparison_df["has_negated_positive_word"] = comparison_df["review"].apply(has_negated_positive_word)
    comparison_df["has_negated_negative_word"] = comparison_df["review"].apply(has_negated_negative_word)

    comparison_path = GENERATED_OUT_DIR / "prediction_comparison_rubert_xai_augmented.csv"
    comparison_df.to_csv(comparison_path, index=False)

    fixed_df = comparison_df[(~comparison_df["original_correct"]) & (comparison_df["augmented_correct"])].copy()
    broken_df = comparison_df[(comparison_df["original_correct"]) & (~comparison_df["augmented_correct"])].copy()

    fixed_path = GENERATED_OUT_DIR / "fixed_examples_rubert_xai_augmented.csv"
    broken_path = GENERATED_OUT_DIR / "new_errors_rubert_xai_augmented.csv"
    fixed_df.to_csv(fixed_path, index=False)
    broken_df.to_csv(broken_path, index=False)

    summary_rows = []
    for subset_name, mask in {
        "all_test": pd.Series(True, index=comparison_df.index),
        "negation_context": comparison_df["has_negation_context"],
        "negated_positive_word": comparison_df["has_negated_positive_word"],
        "negated_negative_word": comparison_df["has_negated_negative_word"],
    }.items():
        part = comparison_df[mask]
        if part.empty:
            continue
        summary_rows.append(
            {
                "subset": subset_name,
                "count": len(part),
                "original_correct": int(part["original_correct"].sum()),
                "augmented_correct": int(part["augmented_correct"].sum()),
                "fixed_count": int(((~part["original_correct"]) & part["augmented_correct"]).sum()),
                "new_error_count": int((part["original_correct"] & (~part["augmented_correct"])).sum()),
                "changed_prediction_count": int(part["changed_prediction"].sum()),
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    summary_path = TABLES_DIR / "xai_error_loop_rubert_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    ax = summary_df.set_index("subset")[["fixed_count", "new_error_count"]].plot(kind="bar", figsize=(10, 5))
    ax.set_xlabel("Subset")
    ax.set_ylabel("Count")
    ax.set_title("XAI Error Loop: Fixed vs New Errors")
    ax.tick_params(axis="x", rotation=25)
    plt.tight_layout()
    fig_path = FIGURES_DIR / "xai_error_loop_rubert_fixed_vs_new_errors.png"
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close()

    return summary_path, comparison_path, fixed_path, broken_path, fig_path


def save_training_artifacts(history, train_augmented, selected_errors, synthetic_df, val_errors):
    history_df = pd.DataFrame(history)
    history_path = TABLES_DIR / "history_rubert_xai_augmented.csv"
    history_df.to_csv(history_path, index=False)

    train_augmented_path = GENERATED_OUT_DIR / "train_augmented_rubert_xai_augmented.csv"
    selected_errors_path = GENERATED_OUT_DIR / "selected_validation_errors_for_augmentation.csv"
    synthetic_path = GENERATED_OUT_DIR / "synthetic_negation_examples.csv"
    val_errors_path = GENERATED_OUT_DIR / "validation_errors_with_negation_flags.csv"

    train_augmented.to_csv(train_augmented_path, index=False)
    selected_errors.to_csv(selected_errors_path, index=False)
    synthetic_df.to_csv(synthetic_path, index=False)
    val_errors.to_csv(val_errors_path, index=False)

    source_counts = train_augmented["augmentation_source"].value_counts().rename_axis("source").reset_index(name="count")
    source_counts_path = TABLES_DIR / "augmentation_sources_rubert_xai_augmented.csv"
    source_counts.to_csv(source_counts_path, index=False)

    ax = source_counts.plot(kind="bar", x="source", y="count", legend=False, figsize=(9, 5))
    ax.set_xlabel("Augmentation source")
    ax.set_ylabel("Count")
    ax.set_title("RuBERT XAI-Augmented Training Data")
    ax.tick_params(axis="x", rotation=25)
    plt.tight_layout()
    fig_path = FIGURES_DIR / "augmentation_sources_rubert_xai_augmented.png"
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close()

    return {
        "history": history_path,
        "train_augmented": train_augmented_path,
        "selected_errors": selected_errors_path,
        "synthetic": synthetic_path,
        "val_errors": val_errors_path,
        "source_counts": source_counts_path,
        "source_counts_fig": fig_path,
    }


def main():
    set_seed(RANDOM_STATE)

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    GENERATED_OUT_DIR.mkdir(parents=True, exist_ok=True)
    AUGMENTED_MODEL_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Using device: {DEVICE}")
    print(f"Source model: {SOURCE_MODEL_DIR}")
    print(f"Augmented model will be saved to: {AUGMENTED_MODEL_DIR}")

    tokenizer = get_rubert_tokenizer(SOURCE_MODEL_DIR)
    source_model = get_rubert_model(SOURCE_MODEL_DIR).to(DEVICE)

    train_df = clean_dataframe(get_train_data())
    val_df = clean_dataframe(get_val_data())
    test_df = clean_dataframe(get_test_data())

    print("Predicting validation with original RuBERT to collect errors...")
    val_predictions = predict_dataframe(source_model, tokenizer, val_df)
    train_augmented, selected_errors, synthetic_df, val_errors = build_error_focused_training_data(
        train_df,
        val_predictions,
    )

    artifact_paths = save_training_artifacts(
        history=[],
        train_augmented=train_augmented,
        selected_errors=selected_errors,
        synthetic_df=synthetic_df,
        val_errors=val_errors,
    )
    print(f"Selected validation errors: {len(selected_errors)}")
    print(f"Synthetic negation examples: {len(synthetic_df)}")
    print(f"Augmented train size: {len(train_augmented)}")

    model = get_rubert_model(SOURCE_MODEL_DIR).to(DEVICE)
    train_loader = make_loader(train_augmented, tokenizer, shuffle=True)
    val_loader = make_loader(val_df, tokenizer, shuffle=False)

    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    total_steps = len(train_loader) * NUM_EPOCHS
    warmup_steps = int(total_steps * WARMUP_RATIO)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    history = []
    for epoch in range(NUM_EPOCHS):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, scheduler)
        val_loss, val_acc = evaluate_loss_acc(model, val_loader)
        row = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "max_len": MAX_LEN,
            "learning_rate": LEARNING_RATE,
            "augmented_train_size": len(train_augmented),
        }
        history.append(row)
        print(
            f"Epoch {epoch + 1}/{NUM_EPOCHS} | "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}"
        )

    model.save_pretrained(AUGMENTED_MODEL_DIR)
    tokenizer.save_pretrained(AUGMENTED_MODEL_DIR)

    artifact_paths = save_training_artifacts(
        history=history,
        train_augmented=train_augmented,
        selected_errors=selected_errors,
        synthetic_df=synthetic_df,
        val_errors=val_errors,
    )

    print("Evaluating original and augmented RuBERT on test...")
    base_test_pred = predict_dataframe(source_model, tokenizer, test_df)
    aug_test_pred = predict_dataframe(model, tokenizer, test_df)

    base_test_pred.to_csv(GENERATED_OUT_DIR / "predictions_rubert_original_test.csv", index=False)
    aug_test_pred.to_csv(GENERATED_OUT_DIR / "predictions_rubert_xai_augmented_test.csv", index=False)

    metrics_df, metrics_path, report_path = save_metrics_and_reports(base_test_pred, aug_test_pred)
    cm_path = save_confusion_matrix(aug_test_pred)
    summary_path, comparison_path, fixed_path, broken_path, loop_fig_path = save_improvement_analysis(
        base_test_pred,
        aug_test_pred,
    )

    print("\nMetrics:")
    print(metrics_df)

    print("\nSaved files:")
    print(AUGMENTED_MODEL_DIR)
    print(metrics_path)
    print(report_path)
    print(cm_path)
    print(summary_path)
    print(comparison_path)
    print(fixed_path)
    print(broken_path)
    print(loop_fig_path)
    for path in artifact_paths.values():
        print(path)


if __name__ == "__main__":
    main()
