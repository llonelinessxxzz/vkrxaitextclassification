import matplotlib.pyplot as plt
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
from torch.utils.data import DataLoader, Dataset

from config import CHECKPOINTS_DIR, FIGURES_DIR, TABLES_DIR, RANDOM_STATE
from dataset import get_test_data
from rubert_model import get_rubert_model, get_rubert_tokenizer


MODEL_DIR = CHECKPOINTS_DIR / "rubert_best"

METRICS_PATH = TABLES_DIR / "metrics_rubert.csv"
REPORT_PATH = TABLES_DIR / "classification_report_rubert.csv"
CM_FIG_PATH = FIGURES_DIR / "confusion_matrix_rubert.png"

MAX_LEN = 128
BATCH_SIZE = 16
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLASS_NAMES = ["negative", "neutral", "positive"]


torch.manual_seed(RANDOM_STATE)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_STATE)


class RuBERTDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer, max_len: int = MAX_LEN):
        self.texts = df["review"].astype(str).tolist()
        self.labels = df["label_id"].astype(int).tolist()
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx: int):
        text = self.texts[idx]
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
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
            "labels": torch.tensor(label, dtype=torch.long),
        }


def create_test_dataloader(tokenizer):
    test_df = get_test_data()
    test_dataset = RuBERTDataset(test_df, tokenizer)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    return test_loader


def load_model_and_tokenizer():
    tokenizer = get_rubert_tokenizer(MODEL_DIR)
    model = get_rubert_model(MODEL_DIR).to(DEVICE)
    model.eval()
    return model, tokenizer


def predict(model, dataloader):
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

            logits = outputs.logits
            preds = torch.argmax(logits, dim=1)

            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    return all_labels, all_preds


def save_metrics(y_true, y_pred):
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    metrics_df = pd.DataFrame(
        {
            "metric": [
                "accuracy",
                "precision_macro",
                "recall_macro",
                "f1_macro",
                "precision_weighted",
                "recall_weighted",
                "f1_weighted",
            ],
            "value": [
                accuracy_score(y_true, y_pred),
                precision_score(y_true, y_pred, average="macro", zero_division=0),
                recall_score(y_true, y_pred, average="macro", zero_division=0),
                f1_score(y_true, y_pred, average="macro", zero_division=0),
                precision_score(y_true, y_pred, average="weighted", zero_division=0),
                recall_score(y_true, y_pred, average="weighted", zero_division=0),
                f1_score(y_true, y_pred, average="weighted", zero_division=0),
            ],
        }
    )
    metrics_df.to_csv(METRICS_PATH, index=False)

    report_dict = classification_report(
        y_true,
        y_pred,
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0,
    )
    report_df = pd.DataFrame(report_dict).transpose()
    report_df.to_csv(REPORT_PATH)

    return metrics_df, report_df


def save_confusion_matrix(y_true, y_pred):
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm)

    ax.set_xticks([0, 1, 2])
    ax.set_yticks([0, 1, 2])
    ax.set_xticklabels(CLASS_NAMES)
    ax.set_yticklabels(CLASS_NAMES)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title("Confusion Matrix - RuBERT")

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")

    fig.colorbar(im)
    plt.tight_layout()
    plt.savefig(CM_FIG_PATH, dpi=300, bbox_inches="tight")
    plt.close()


def main():
    print(f"Using device: {DEVICE}")

    model, tokenizer = load_model_and_tokenizer()
    test_loader = create_test_dataloader(tokenizer)

    y_true, y_pred = predict(model, test_loader)

    metrics_df, report_df = save_metrics(y_true, y_pred)
    save_confusion_matrix(y_true, y_pred)

    print("\nMain metrics:")
    print(metrics_df)

    print("\nClassification report:")
    print(report_df)

    print("\nSaved files:")
    print(METRICS_PATH)
    print(REPORT_PATH)
    print(CM_FIG_PATH)


if __name__ == "__main__":
    main()