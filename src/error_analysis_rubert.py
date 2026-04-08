import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from config import CHECKPOINTS_DIR, GENERATED_DIR, RANDOM_STATE
from dataset import get_test_data
from rubert_model import get_rubert_model, get_rubert_tokenizer


MODEL_DIR = CHECKPOINTS_DIR / "rubert_best"

MISCLASSIFIED_PATH = GENERATED_DIR / "misclassified_examples_rubert.csv"
ERROR_SUMMARY_PATH = GENERATED_DIR / "error_summary_rubert.csv"

MAX_LEN = 128
BATCH_SIZE = 16
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLASS_NAMES = ["negative", "neutral", "positive"]
ID_TO_LABEL = {0: "negative", 1: "neutral", 2: "positive"}


torch.manual_seed(RANDOM_STATE)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_STATE)


def simple_token_count(text: str) -> int:
    return len(str(text).split())


class RuBERTDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer, max_len: int = MAX_LEN):
        self.df = df.reset_index(drop=True)
        self.texts = self.df["review"].astype(str).tolist()
        self.labels = self.df["label_id"].astype(int).tolist()
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
            "text": text,
        }


def collate_fn(batch):
    return {
        "input_ids": torch.stack([item["input_ids"] for item in batch]),
        "attention_mask": torch.stack([item["attention_mask"] for item in batch]),
        "labels": torch.stack([item["labels"] for item in batch]),
        "texts": [item["text"] for item in batch],
    }


def create_test_dataloader(tokenizer):
    test_df = get_test_data()
    test_dataset = RuBERTDataset(test_df, tokenizer)

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
    )
    return test_loader


def load_model_and_tokenizer():
    tokenizer = get_rubert_tokenizer(MODEL_DIR)
    model = get_rubert_model(MODEL_DIR).to(DEVICE)
    model.eval()
    return model, tokenizer


def collect_misclassified_examples(model, dataloader) -> pd.DataFrame:
    rows = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["labels"]
            texts = batch["texts"]

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

            probs = torch.softmax(outputs.logits, dim=1).cpu()
            preds = torch.argmax(probs, dim=1)

            for i in range(len(texts)):
                true_id = int(labels[i].item())
                pred_id = int(preds[i].item())
                confidence = float(probs[i, pred_id].item())
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
                            "text_length_tokens": simple_token_count(text),
                        }
                    )

    return pd.DataFrame(rows)


def save_error_summary(errors_df: pd.DataFrame) -> pd.DataFrame:
    if errors_df.empty:
        summary_df = pd.DataFrame(columns=["true_label", "pred_label", "count"])
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

    model, tokenizer = load_model_and_tokenizer()
    test_loader = create_test_dataloader(tokenizer)

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