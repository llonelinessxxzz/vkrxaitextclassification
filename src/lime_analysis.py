import re
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from lime.lime_text import LimeTextExplainer

from config import CHECKPOINTS_DIR, FIGURES_DIR, GENERATED_DIR, RANDOM_STATE
from dataset import get_train_data
from lstm_model import LSTMClassifier


torch.manual_seed(RANDOM_STATE)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_STATE)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MAX_VOCAB_SIZE = 30000
MAX_LEN = 150
BATCH_SIZE = 64
EMBEDDING_DIM = 128
HIDDEN_DIM = 128
OUTPUT_DIM = 3
NUM_LAYERS = 1
DROPOUT = 0.3
BIDIRECTIONAL = True

PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"

BEST_MODEL_PATH = CHECKPOINTS_DIR / "lstm_best.pt"
TAXONOMY_PATH = GENERATED_DIR / "taxonomy_lstm.csv"
MISCLASSIFIED_PATH = GENERATED_DIR / "misclassified_examples_lstm.csv"

LIME_DIR = GENERATED_DIR / "lime_lstm"
LIME_HTML_DIR = LIME_DIR / "html"
LIME_FIG_DIR = FIGURES_DIR / "lime_lstm"

SELECTED_EXAMPLES_PATH = LIME_DIR / "lime_selected_examples_lstm.csv"
LIME_SUMMARY_PATH = LIME_DIR / "lime_explanations_lstm.csv"

CLASS_NAMES = ["negative", "neutral", "positive"]
LABEL_TO_ID = {label: idx for idx, label in enumerate(CLASS_NAMES)}
ID_TO_LABEL = {idx: label for idx, label in enumerate(CLASS_NAMES)}


def simple_tokenize(text: str) -> list[str]:
    text = str(text).lower().strip()
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
    tokens = simple_tokenize(text)
    token_ids = [vocab.get(token, vocab[UNK_TOKEN]) for token in tokens]

    if len(token_ids) > max_len:
        token_ids = token_ids[:max_len]
    else:
        token_ids += [vocab[PAD_TOKEN]] * (max_len - len(token_ids))

    return token_ids


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


def predict_proba_factory(model, vocab):
    def predict_proba(texts: list[str]) -> np.ndarray:
        encoded = [encode_text(text, vocab) for text in texts]
        probs_all = []

        with torch.no_grad():
            for i in range(0, len(encoded), BATCH_SIZE):
                batch = encoded[i:i + BATCH_SIZE]
                batch_tensor = torch.tensor(batch, dtype=torch.long).to(DEVICE)

                outputs = model(batch_tensor)
                probs = torch.softmax(outputs, dim=1).cpu().numpy()
                probs_all.append(probs)

        return np.vstack(probs_all)

    return predict_proba


def load_error_dataframe() -> pd.DataFrame:
    if TAXONOMY_PATH.exists():
        df = pd.read_csv(TAXONOMY_PATH)
    else:
        df = pd.read_csv(MISCLASSIFIED_PATH)

    return df


def select_examples(df: pd.DataFrame, max_examples: int = 12, per_type: int = 2) -> pd.DataFrame:
    df = df.copy()

    if "confidence" in df.columns:
        df = df.sort_values("confidence", ascending=False).reset_index(drop=True)

    selected_parts = []

    if "error_type" in df.columns:
        error_type_order = df["error_type"].value_counts().index.tolist()

        for error_type in error_type_order:
            part = df[df["error_type"] == error_type].head(per_type)
            if not part.empty:
                selected_parts.append(part)

        if selected_parts:
            selected_df = pd.concat(selected_parts, ignore_index=True)
            selected_df = selected_df.drop_duplicates(subset=["text"])
        else:
            selected_df = df.head(max_examples).copy()
    else:
        selected_df = df.head(max_examples).copy()

    if len(selected_df) < max_examples:
        remaining = df[~df["text"].isin(selected_df["text"])].head(max_examples - len(selected_df))
        selected_df = pd.concat([selected_df, remaining], ignore_index=True)

    selected_df = selected_df.head(max_examples).reset_index(drop=True)
    return selected_df


def explain_examples(model, vocab, examples_df: pd.DataFrame) -> pd.DataFrame:
    predict_proba = predict_proba_factory(model, vocab)

    explainer = LimeTextExplainer(
        class_names=CLASS_NAMES,
        random_state=RANDOM_STATE,
    )

    rows = []

    for idx, row in examples_df.iterrows():
        text = str(row["text"])
        true_label = str(row["true_label"])
        pred_label = str(row["pred_label"])
        pred_label_id = LABEL_TO_ID[pred_label]

        explanation = explainer.explain_instance(
            text_instance=text,
            classifier_fn=predict_proba,
            labels=[pred_label_id],
            num_features=10,
            num_samples=1000,
        )

        html_path = LIME_HTML_DIR / f"lime_example_{idx + 1}.html"
        explanation.save_to_file(str(html_path))

        fig = explanation.as_pyplot_figure(label=pred_label_id)
        fig_path = LIME_FIG_DIR / f"lime_example_{idx + 1}.png"
        fig.tight_layout()
        fig.savefig(fig_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

        feature_list = explanation.as_list(label=pred_label_id)

        for rank, (feature, weight) in enumerate(feature_list, start=1):
            rows.append(
                {
                    "example_id": idx + 1,
                    "text": text,
                    "true_label": true_label,
                    "pred_label": pred_label,
                    "confidence": row["confidence"] if "confidence" in row else None,
                    "error_type": row["error_type"] if "error_type" in row else None,
                    "feature_rank": rank,
                    "feature": feature,
                    "weight": weight,
                    "html_path": str(html_path),
                    "figure_path": str(fig_path),
                }
            )

        print(
            f"Processed example {idx + 1}/{len(examples_df)} | "
            f"true={true_label} | pred={pred_label}"
        )

    return pd.DataFrame(rows)


def main():
    LIME_DIR.mkdir(parents=True, exist_ok=True)
    LIME_HTML_DIR.mkdir(parents=True, exist_ok=True)
    LIME_FIG_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Using device: {DEVICE}")

    train_df = get_train_data()
    vocab = build_vocab(train_df["review"].tolist())
    model = load_model(vocab)

    errors_df = load_error_dataframe()
    selected_df = select_examples(errors_df, max_examples=12, per_type=2)
    selected_df.to_csv(SELECTED_EXAMPLES_PATH, index=False)

    lime_summary_df = explain_examples(model, vocab, selected_df)
    lime_summary_df.to_csv(LIME_SUMMARY_PATH, index=False)

    print("\nLIME analysis completed.")
    print(f"Selected examples saved to: {SELECTED_EXAMPLES_PATH}")
    print(f"LIME summary saved to: {LIME_SUMMARY_PATH}")
    print(f"HTML explanations directory: {LIME_HTML_DIR}")
    print(f"Figure explanations directory: {LIME_FIG_DIR}")


if __name__ == "__main__":
    main()