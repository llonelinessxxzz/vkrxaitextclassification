import re
from collections import Counter

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from config import CHECKPOINTS_DIR, GENERATED_DIR, RANDOM_STATE
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
MISCLASSIFIED_PATH = GENERATED_DIR / "misclassified_examples_lstm.csv"
ERROR_SUMMARY_PATH = GENERATED_DIR / "error_summary_lstm.csv"

ID_TO_LABEL = {
    0: "negative",
    1: "neutral",
    2: "positive",
}


def simple_tokenize(text: str) -> list[str]:
    text = str(text).lower().strip()
    text = re.sub(r"[^\w\sа-яё]", " ", text, flags=re.IGNORECASE)
    return text.split()


def build_vocab(texts, max_vocab_size: int = MAX_VOCAB_SIZE) -> dict[str, int]:
    counter = Counter()

    for text in texts:
        counter.update(simple_tokenize(text))

    most_common = counter.most_common(max_vocab_size - 2)

    vocab = {
        PAD_TOKEN: 0,
        UNK_TOKEN: 1,
    }

    for idx, (token, _) in enumerate(most_common, start=2):
        vocab[token] = idx

    return vocab


def encode_text(text: str, vocab: dict[str, int], max_len: int = MAX_LEN) -> list[int]:
    tokens = simple_tokenize(text)
    token_ids = [vocab.get(token, vocab[UNK_TOKEN]) for token in tokens]

    if len(token_ids) > max_len:
        token_ids = token_ids[:max_len]
    else:
        token_ids += [vocab[PAD_TOKEN]] * (max_len - len(token_ids))

    return token_ids


class ReviewsDataset(Dataset):
    def __init__(self, df: pd.DataFrame, vocab: dict[str, int]):
        self.df = df.reset_index(drop=True)
        self.texts = self.df["review"].tolist()
        self.labels = self.df["label_id"].tolist()
        self.vocab = vocab

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int):
        text_ids = encode_text(self.texts[idx], self.vocab)
        label = self.labels[idx]

        return {
            "text_ids": torch.tensor(text_ids, dtype=torch.long),
            "label": torch.tensor(label, dtype=torch.long),
            "text": self.texts[idx],
        }


def collate_fn(batch):
    text_ids = torch.stack([item["text_ids"] for item in batch])
    labels = torch.stack([item["label"] for item in batch])
    texts = [item["text"] for item in batch]

    return text_ids, labels, texts


def create_test_dataloader_and_vocab():
    train_df = get_train_data()
    test_df = get_test_data()

    vocab = build_vocab(train_df["review"].tolist())
    test_dataset = ReviewsDataset(test_df, vocab)

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
    )

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


def collect_misclassified_examples(model, dataloader) -> pd.DataFrame:
    rows = []

    with torch.no_grad():
        for text_ids, labels, texts in dataloader:
            text_ids = text_ids.to(DEVICE)

            outputs = model(text_ids)
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1).cpu()

            for i in range(len(texts)):
                true_id = int(labels[i].item())
                pred_id = int(preds[i].item())
                confidence = float(probs[i, pred_id].cpu().item())
                text = texts[i]

                if true_id != pred_id:
                    rows.append(
                        {
                            "text": text,
                            "true_label_id": true_id,
                            "true_label": ID_TO_LABEL[true_id],
                            "pred_label_id": pred_id,
                            "pred_label": ID_TO_LABEL[pred_id],
                            "confidence": round(confidence, 6),
                            "text_length_chars": len(str(text)),
                            "text_length_tokens": len(simple_tokenize(text)),
                        }
                    )

    return pd.DataFrame(rows)


def save_error_summary(errors_df: pd.DataFrame) -> pd.DataFrame:
    if errors_df.empty:
        summary_df = pd.DataFrame(
            columns=["true_label", "pred_label", "count"]
        )
    else:
        summary_df = (
            errors_df.groupby(["true_label", "pred_label"])
            .size()
            .reset_index(name="count")
            .sort_values("count", ascending=False)
        )

    summary_df.to_csv(ERROR_SUMMARY_PATH, index=False)
    return summary_df


def main():
    GENERATED_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Using device: {DEVICE}")

    test_loader, vocab = create_test_dataloader_and_vocab()
    model = load_model(vocab)

    errors_df = collect_misclassified_examples(model, test_loader)
    errors_df.to_csv(MISCLASSIFIED_PATH, index=False)

    summary_df = save_error_summary(errors_df)

    print("\nError analysis completed.")
    print(f"Total misclassified examples: {len(errors_df)}")
    print(f"Misclassified examples saved to: {MISCLASSIFIED_PATH}")
    print(f"Error summary saved to: {ERROR_SUMMARY_PATH}")

    print("\nTop error pairs:")
    print(summary_df.head(10))

    if not errors_df.empty:
        print("\nSample misclassified examples:")
        print(
            errors_df[
                ["text", "true_label", "pred_label", "confidence", "text_length_tokens"]
            ].head(5)
        )


if __name__ == "__main__":
    main()