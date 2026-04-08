import re
from collections import Counter

import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader, Dataset

from config import CHECKPOINTS_DIR, FIGURES_DIR, TABLES_DIR, RANDOM_STATE
from dataset import get_test_data, get_train_data, get_val_data
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
NUM_EPOCHS = 10
LEARNING_RATE = 5e-4
NUM_LAYERS = 1
DROPOUT = 0.3
BIDIRECTIONAL = True

PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"

BEST_MODEL_PATH = CHECKPOINTS_DIR / "lstm_best.pt"
HISTORY_PATH = TABLES_DIR / "history_lstm.csv"
LOSS_FIG_PATH = FIGURES_DIR / "lstm_loss.png"
ACC_FIG_PATH = FIGURES_DIR / "lstm_accuracy.png"


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


def create_dataloaders():
    train_df = get_train_data()
    val_df = get_val_data()
    test_df = get_test_data()

    vocab = build_vocab(train_df["review"].tolist())

    train_dataset = ReviewsDataset(train_df, vocab)
    val_dataset = ReviewsDataset(val_df, vocab)
    test_dataset = ReviewsDataset(test_df, vocab)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    return train_loader, val_loader, test_loader, vocab


def calculate_accuracy(predictions: torch.Tensor, labels: torch.Tensor) -> float:
    preds = torch.argmax(predictions, dim=1)
    return (preds == labels).float().mean().item()


def train_epoch(model, dataloader, criterion, optimizer):
    model.train()

    epoch_loss = 0.0
    epoch_acc = 0.0

    for texts, labels in dataloader:
        texts = texts.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(texts)
        loss = criterion(outputs, labels)
        acc = calculate_accuracy(outputs, labels)

        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        epoch_acc += acc

    return epoch_loss / len(dataloader), epoch_acc / len(dataloader)


def evaluate(model, dataloader, criterion):
    model.eval()

    epoch_loss = 0.0
    epoch_acc = 0.0

    with torch.no_grad():
        for texts, labels in dataloader:
            texts = texts.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(texts)
            loss = criterion(outputs, labels)
            acc = calculate_accuracy(outputs, labels)

            epoch_loss += loss.item()
            epoch_acc += acc

    return epoch_loss / len(dataloader), epoch_acc / len(dataloader)


def predict_labels(model, dataloader):
    model.eval()

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


def save_history(history: list[dict]) -> pd.DataFrame:
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    history_df = pd.DataFrame(history)
    history_df.to_csv(HISTORY_PATH, index=False)
    return history_df


def plot_history(history_df: pd.DataFrame) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 5))
    plt.plot(history_df["epoch"], history_df["train_loss"], label="train_loss")
    plt.plot(history_df["epoch"], history_df["val_loss"], label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("LSTM Training Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(LOSS_FIG_PATH, dpi=300, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(history_df["epoch"], history_df["train_acc"], label="train_acc")
    plt.plot(history_df["epoch"], history_df["val_acc"], label="val_acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("LSTM Training Accuracy")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(ACC_FIG_PATH, dpi=300, bbox_inches="tight")
    plt.close()


def main():
    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Using device: {DEVICE}")

    train_loader, val_loader, test_loader, vocab = create_dataloaders()

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

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_val_acc = 0.0
    best_epoch = 0
    history = []

    for epoch in range(NUM_EPOCHS):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc = evaluate(model, val_loader, criterion)

        history_row = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
        }
        history.append(history_row)

        print(
            f"Epoch {epoch + 1}/{NUM_EPOCHS} | "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch + 1
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            print(f"Best model saved to: {BEST_MODEL_PATH}")

    history_df = save_history(history)
    plot_history(history_df)

    model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=DEVICE))

    test_loss, test_acc = evaluate(model, test_loader, criterion)
    y_true, y_pred = predict_labels(model, test_loader)
    test_acc_sklearn = accuracy_score(y_true, y_pred)

    print("\nBest validation result:")
    print(f"Best epoch: {best_epoch}")
    print(f"Best val acc: {best_val_acc:.4f}")

    print("\nFinal test results:")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Acc: {test_acc:.4f}")
    print(f"Test Acc (sklearn): {test_acc_sklearn:.4f}")

    print("\nSaved files:")
    print(BEST_MODEL_PATH)
    print(HISTORY_PATH)
    print(LOSS_FIG_PATH)
    print(ACC_FIG_PATH)


if __name__ == "__main__":
    main()