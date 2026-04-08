import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import get_linear_schedule_with_warmup

from config import CHECKPOINTS_DIR, FIGURES_DIR, TABLES_DIR, RANDOM_STATE
from dataset import get_test_data, get_train_data, get_val_data
from rubert_model import get_rubert_model, get_rubert_tokenizer


MODEL_DIR = CHECKPOINTS_DIR / "rubert_best"
HISTORY_PATH = TABLES_DIR / "history_rubert.csv"
LOSS_FIG_PATH = FIGURES_DIR / "rubert_loss.png"
ACC_FIG_PATH = FIGURES_DIR / "rubert_accuracy.png"

MODEL_NAME = "DeepPavlov/rubert-base-cased"
MAX_LEN = 128
BATCH_SIZE = 16
NUM_EPOCHS = 3
LEARNING_RATE = 2e-5
WARMUP_RATIO = 0.1

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int = RANDOM_STATE):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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

        item = {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(label, dtype=torch.long),
        }
        return item


def create_dataloaders(tokenizer):
    train_df = get_train_data()
    val_df = get_val_data()
    test_df = get_test_data()

    train_dataset = RuBERTDataset(train_df, tokenizer)
    val_dataset = RuBERTDataset(val_df, tokenizer)
    test_dataset = RuBERTDataset(test_df, tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    return train_loader, val_loader, test_loader


def train_epoch(model, dataloader, optimizer, scheduler):
    model.train()

    total_loss = 0.0
    all_preds = []
    all_labels = []

    for batch in dataloader:
        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        labels = batch["labels"].to(DEVICE)

        optimizer.zero_grad()

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )

        loss = outputs.loss
        logits = outputs.logits

        loss.backward()
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()

        preds = torch.argmax(logits, dim=1).detach().cpu().numpy()
        true_labels = labels.detach().cpu().numpy()

        all_preds.extend(preds)
        all_labels.extend(true_labels)

    avg_loss = total_loss / len(dataloader)
    acc = accuracy_score(all_labels, all_preds)

    return avg_loss, acc


def evaluate(model, dataloader):
    model.eval()

    total_loss = 0.0
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
                labels=labels,
            )

            loss = outputs.loss
            logits = outputs.logits

            total_loss += loss.item()

            preds = torch.argmax(logits, dim=1).detach().cpu().numpy()
            true_labels = labels.detach().cpu().numpy()

            all_preds.extend(preds)
            all_labels.extend(true_labels)

    avg_loss = total_loss / len(dataloader)
    acc = accuracy_score(all_labels, all_preds)

    return avg_loss, acc


def save_history(history: list[dict]) -> pd.DataFrame:
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    history_df = pd.DataFrame(history)
    history_df.to_csv(HISTORY_PATH, index=False)
    return history_df


def plot_history(history_df: pd.DataFrame):
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 5))
    plt.plot(history_df["epoch"], history_df["train_loss"], label="train_loss")
    plt.plot(history_df["epoch"], history_df["val_loss"], label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("RuBERT Training Loss")
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
    plt.title("RuBERT Training Accuracy")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(ACC_FIG_PATH, dpi=300, bbox_inches="tight")
    plt.close()


def main():
    set_seed(RANDOM_STATE)

    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Using device: {DEVICE}")

    tokenizer = get_rubert_tokenizer(MODEL_NAME)
    model = get_rubert_model(MODEL_NAME).to(DEVICE)

    train_loader, val_loader, test_loader = create_dataloaders(tokenizer)

    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)

    total_steps = len(train_loader) * NUM_EPOCHS
    warmup_steps = int(total_steps * WARMUP_RATIO)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    best_val_acc = 0.0
    best_epoch = 0
    history = []

    for epoch in range(NUM_EPOCHS):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, scheduler)
        val_loss, val_acc = evaluate(model, val_loader)

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

            model.save_pretrained(MODEL_DIR)
            tokenizer.save_pretrained(MODEL_DIR)

            print(f"Best model saved to: {MODEL_DIR}")

    history_df = save_history(history)
    plot_history(history_df)

    best_model = get_rubert_model(MODEL_DIR).to(DEVICE)
    test_loss, test_acc = evaluate(best_model, test_loader)

    print("\nBest validation result:")
    print(f"Best epoch: {best_epoch}")
    print(f"Best val acc: {best_val_acc:.4f}")
    print("\nFinal test results:")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Acc: {test_acc:.4f}")

    print("\nSaved files:")
    print(MODEL_DIR)
    print(HISTORY_PATH)
    print(LOSS_FIG_PATH)
    print(ACC_FIG_PATH)


if __name__ == "__main__":
    main()