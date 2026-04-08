import re
from collections import Counter

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
from dataset import get_test_data, get_train_data
from lstm_model import LSTMClassifier


torch.manual_seed(RANDOM_STATE)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_STATE)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MAX_VOCAB_SIZE = 30000
MAX_LEN = 150
BATCH_SIZE = 32
EMBEDDING_DIM = 128
HIDDEN_DIM = 128
OUTPUT_DIM = 3
NUM_LAYERS = 1
DROPOUT = 0.3
BIDIRECTIONAL = True

PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"

BEST_MODEL_PATH = CHECKPOINTS_DIR / "lstm_best.pt"

ID_TO_LABEL = {
    0: "negative",
    1: "neutral",
    2: "positive",
}


def simple_tokenize(text: str) -> list[str]:
    text = text.lower().strip()
    text = re.sub(r"[^\w\sа-яё]", " ", text, flags=re.IGNORECASE)
    return text.split()


def build_vocab(texts, max_vocab_size: int = MAX_VOCAB_SIZE) -> dict[str, int]:
    counter = Counter()

    for text in texts:
        counter.update(simple_tokenize(str(text)))

    most_common = counter.most_common(max_vocab_size - 2)

    vocab = {
        PAD_TOKEN: 0,
        UNK_TOKEN: 1,
    }

    for idx, (token, _) in enumerate(most_common, start=2):
        vocab[token] = idx

    return vocab


def encode_text(text: str, vocab: dict[str, int], max_len: int = MAX_LEN) -> list[int]:
    tokens = simple_tokenize(str(text))
    token_ids = [vocab.get(token, vocab[UNK_TOKEN]) for token in tokens]

    if len(token_ids) > max_len:
        token_ids = token_ids[:max_len]
    else:
        token_ids += [vocab[PAD_TOKEN]] * (max_len - len(token_ids))

    return token_ids


class ReviewsDataset(Dataset):
    def __init__(self, df: pd.DataFrame, vocab: dict[str, int]):
        self.texts = df["review"].tolist()
        self.labels = df["label_id"].tolist()
        self.vocab = vocab

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int):
        text_ids = encode_text(self.texts[idx], self.vocab)
        label = self.labels[idx]

        return (
            torch.tensor(text_ids, dtype=torch.long),
            torch.tensor(label, dtype=torch.long),
        )


def create_test_dataloader_and_vocab():
    train_df = get_train_data()
    test_df = get_test_data()

    vocab = build_vocab(train_df["review"].tolist())
    test_dataset = ReviewsDataset(test_df, vocab)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    return test_loader, vocab


def load_model(vocab: dict[str, int]) -> LSTMClassifier:
    model = LSTMClassifier(
        vocab_size=len(vocab),
        embedding_dim=EMBEDDING_DIM,
        hidden_dim=HIDDEN_DIM,
        output_dim=OUTPUT_DIM,
        pad_idx=vocab[PAD_TOKEN],
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
        bidirectional=BIDIRECTIONAL,
    ).to(DEVICE)

    model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=DEVICE))
    model.eval()
    return model


def predict(model, dataloader):
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for texts, labels in dataloader:
            texts = texts.to(DEVICE)

            outputs = model(texts)
            preds = torch.argmax(outputs, dim=1).cpu().tolist()

            all_preds.extend(preds)
            all_labels.extend(labels.tolist())

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

    metrics_path = TABLES_DIR / "metrics_lstm.csv"
    metrics_df.to_csv(metrics_path, index=False)

    report_dict = classification_report(
        y_true,
        y_pred,
        target_names=["negative", "neutral", "positive"],
        output_dict=True,
        zero_division=0,
    )
    report_df = pd.DataFrame(report_dict).transpose()

    report_path = TABLES_DIR / "classification_report_lstm.csv"
    report_df.to_csv(report_path)

    return metrics_df, report_df, metrics_path, report_path


def save_confusion_matrix(y_true, y_pred):
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm)

    ax.set_xticks([0, 1, 2])
    ax.set_yticks([0, 1, 2])
    ax.set_xticklabels(["negative", "neutral", "positive"])
    ax.set_yticklabels(["negative", "neutral", "positive"])
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title("Confusion Matrix - LSTM")

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")

    fig.colorbar(im)
    plt.tight_layout()

    fig_path = FIGURES_DIR / "confusion_matrix_lstm.png"
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close()

    return fig_path


def main():
    print(f"Using device: {DEVICE}")

    test_loader, vocab = create_test_dataloader_and_vocab()
    model = load_model(vocab)

    y_true, y_pred = predict(model, test_loader)

    metrics_df, report_df, metrics_path, report_path = save_metrics(y_true, y_pred)
    fig_path = save_confusion_matrix(y_true, y_pred)

    print("\nMain metrics:")
    print(metrics_df)

    print("\nClassification report:")
    print(report_df)

    print("\nSaved files:")
    print(metrics_path)
    print(report_path)
    print(fig_path)


if __name__ == "__main__":
    main()