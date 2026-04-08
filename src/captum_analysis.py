import re

import matplotlib.pyplot as plt
import pandas as pd
import torch
from captum.attr import LayerIntegratedGradients

from config import CHECKPOINTS_DIR, FIGURES_DIR, GENERATED_DIR, RANDOM_STATE
from rubert_model import get_rubert_model, get_rubert_tokenizer


MODEL_DIR = CHECKPOINTS_DIR / "rubert_best"
MISCLASSIFIED_PATH = GENERATED_DIR / "misclassified_examples_rubert.csv"

CAPTUM_DIR = GENERATED_DIR / "captum_rubert"
CAPTUM_FIG_DIR = FIGURES_DIR / "captum_rubert"

SELECTED_EXAMPLES_PATH = CAPTUM_DIR / "captum_selected_examples_rubert.csv"
TOKEN_ATTRIBUTIONS_PATH = CAPTUM_DIR / "captum_token_attributions_rubert.csv"
TOP_FEATURES_PATH = CAPTUM_DIR / "captum_top_features_rubert.csv"

MAX_LEN = 128
MAX_EXAMPLES = 8

CLASS_NAMES = ["negative", "neutral", "positive"]
LABEL_TO_ID = {"negative": 0, "neutral": 1, "positive": 2}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


torch.manual_seed(RANDOM_STATE)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_STATE)


def load_model_and_tokenizer():
    model = get_rubert_model(MODEL_DIR).to(DEVICE)
    model.eval()
    tokenizer = get_rubert_tokenizer(MODEL_DIR)
    return model, tokenizer


def load_errors() -> pd.DataFrame:
    return pd.read_csv(MISCLASSIFIED_PATH)


def select_examples(df: pd.DataFrame, max_examples: int = MAX_EXAMPLES) -> pd.DataFrame:
    df = df.copy().sort_values("confidence", ascending=False).reset_index(drop=True)

    selected_parts = []
    pair_order = (
        df.groupby(["true_label", "pred_label"])
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
    )

    for _, row in pair_order.iterrows():
        part = df[
            (df["true_label"] == row["true_label"]) &
            (df["pred_label"] == row["pred_label"])
        ].head(2)
        if not part.empty:
            selected_parts.append(part)

    if selected_parts:
        selected_df = pd.concat(selected_parts, ignore_index=True)
        selected_df = selected_df.drop_duplicates(subset=["text"])
    else:
        selected_df = df.head(max_examples).copy()

    if len(selected_df) < max_examples:
        remaining = df[~df["text"].isin(selected_df["text"])].head(max_examples - len(selected_df))
        selected_df = pd.concat([selected_df, remaining], ignore_index=True)

    selected_df = selected_df.head(max_examples).reset_index(drop=True)
    return selected_df


def clean_token(token: str) -> str:
    token = str(token).strip()
    token = token.replace("##", "")
    token = re.sub(r"\s+", " ", token)
    return token


def forward_func(input_ids, attention_mask):
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
    )
    return outputs.logits


def build_baseline(input_ids, tokenizer):
    baseline_id = tokenizer.pad_token_id
    if baseline_id is None:
        baseline_id = 0
    return torch.full_like(input_ids, fill_value=baseline_id)


def normalize_attributions(attr_tensor: torch.Tensor) -> torch.Tensor:
    norm = torch.norm(attr_tensor)
    if norm.item() == 0:
        return attr_tensor
    return attr_tensor / norm


def save_token_bar_plot(example_df: pd.DataFrame, example_id: int, pred_label: str):
    plot_df = (
        example_df.sort_values("abs_attr_value", ascending=False)
        .head(12)
        .sort_values("attr_value", ascending=True)
    )

    plt.figure(figsize=(10, 6))
    plt.barh(plot_df["token"], plot_df["attr_value"])
    plt.xlabel("Integrated Gradients attribution")
    plt.ylabel("Token")
    plt.title(f"Top Captum Tokens - Example {example_id} - Predicted: {pred_label}")
    plt.tight_layout()
    fig_path = CAPTUM_FIG_DIR / f"captum_example_{example_id}.png"
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close()


def aggregate_top_features(token_df: pd.DataFrame) -> pd.DataFrame:
    agg_df = (
        token_df.groupby(["pred_label", "token"], as_index=False)
        .agg(
            mean_attr_value=("attr_value", "mean"),
            abs_mean_attr_value=("abs_attr_value", "mean"),
            count=("token", "count"),
        )
        .sort_values(["pred_label", "abs_mean_attr_value", "count"], ascending=[True, False, False])
    )

    top_df = agg_df.groupby("pred_label", group_keys=False).head(15).reset_index(drop=True)
    top_df.to_csv(TOP_FEATURES_PATH, index=False)
    return top_df


def main():
    global model

    CAPTUM_DIR.mkdir(parents=True, exist_ok=True)
    CAPTUM_FIG_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Using device: {DEVICE}")

    model, tokenizer = load_model_and_tokenizer()

    errors_df = load_errors()
    selected_df = select_examples(errors_df, max_examples=MAX_EXAMPLES)
    selected_df.to_csv(SELECTED_EXAMPLES_PATH, index=False)

    lig = LayerIntegratedGradients(forward_func, model.bert.embeddings.word_embeddings)

    rows = []

    for i in range(len(selected_df)):
        text = str(selected_df.loc[i, "text"])
        true_label = selected_df.loc[i, "true_label"]
        pred_label = selected_df.loc[i, "pred_label"]
        pred_id = LABEL_TO_ID[pred_label]

        encoding = tokenizer(
            text,
            add_special_tokens=True,
            max_length=MAX_LEN,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )

        input_ids = encoding["input_ids"].to(DEVICE)
        attention_mask = encoding["attention_mask"].to(DEVICE)
        baseline_ids = build_baseline(input_ids, tokenizer).to(DEVICE)

        attributions, delta = lig.attribute(
            inputs=input_ids,
            baselines=baseline_ids,
            additional_forward_args=(attention_mask,),
            target=pred_id,
            return_convergence_delta=True,
        )

        token_attr = attributions.sum(dim=-1).squeeze(0).detach().cpu()
        token_attr = normalize_attributions(token_attr)

        input_ids_cpu = input_ids.squeeze(0).detach().cpu().tolist()
        tokens = tokenizer.convert_ids_to_tokens(input_ids_cpu)

        example_rows = []

        for token, value in zip(tokens, token_attr.tolist()):
            token = clean_token(token)

            if token in {"", "[PAD]", "[CLS]", "[SEP]"}:
                continue

            row = {
                "example_id": i + 1,
                "text": text,
                "true_label": true_label,
                "pred_label": pred_label,
                "confidence": selected_df.loc[i, "confidence"],
                "token": token,
                "attr_value": float(value),
                "abs_attr_value": float(abs(value)),
                "delta": float(delta.detach().cpu().item()) if hasattr(delta, "detach") else float(delta),
            }
            rows.append(row)
            example_rows.append(row)

        example_df = pd.DataFrame(example_rows)
        if not example_df.empty:
            save_token_bar_plot(example_df, i + 1, pred_label)

        print(
            f"Processed example {i + 1}/{len(selected_df)} | "
            f"true={true_label} | pred={pred_label}"
        )

    token_df = pd.DataFrame(rows)
    token_df.to_csv(TOKEN_ATTRIBUTIONS_PATH, index=False)

    top_df = aggregate_top_features(token_df)

    print("\nCaptum analysis completed.")
    print(f"Selected examples saved to: {SELECTED_EXAMPLES_PATH}")
    print(f"Token attributions saved to: {TOKEN_ATTRIBUTIONS_PATH}")
    print(f"Top features saved to: {TOP_FEATURES_PATH}")
    print(f"Figures directory: {CAPTUM_FIG_DIR}")

    print("\nTop Captum features:")
    print(top_df.head(20))


if __name__ == "__main__":
    main()