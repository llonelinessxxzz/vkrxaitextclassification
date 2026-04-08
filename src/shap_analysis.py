from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import torch
from transformers import pipeline

from config import CHECKPOINTS_DIR, FIGURES_DIR, GENERATED_DIR, RANDOM_STATE
from rubert_model import get_rubert_model, get_rubert_tokenizer


MODEL_DIR = CHECKPOINTS_DIR / "rubert_best"
MISCLASSIFIED_PATH = GENERATED_DIR / "misclassified_examples_rubert.csv"

SHAP_DIR = GENERATED_DIR / "shap_rubert"
SHAP_FIG_DIR = FIGURES_DIR / "shap_rubert"

SELECTED_EXAMPLES_PATH = SHAP_DIR / "shap_selected_examples_rubert.csv"
TOKEN_ATTRIBUTIONS_PATH = SHAP_DIR / "shap_token_attributions_rubert.csv"
TOP_FEATURES_PATH = SHAP_DIR / "shap_top_features_rubert.csv"

MAX_LEN = 128
MAX_EXAMPLES = 8

CLASS_NAMES = ["negative", "neutral", "positive"]
LABEL_TO_ID = {"negative": 0, "neutral": 1, "positive": 2}


torch.manual_seed(RANDOM_STATE)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_STATE)


def load_model_and_tokenizer():
    model = get_rubert_model(MODEL_DIR)
    tokenizer = get_rubert_tokenizer(MODEL_DIR)
    return model, tokenizer


def build_classifier(model, tokenizer):
    device = 0 if torch.cuda.is_available() else -1

    clf = pipeline(
        task="text-classification",
        model=model,
        tokenizer=tokenizer,
        device=device,
        truncation=True,
        max_length=MAX_LEN,
        top_k=None,
    )

    # SHAP рекомендует оборачивать pipeline вручную
    wrapped = shap.models.TransformersPipeline(clf, rescale_to_logits=True)
    return wrapped, tokenizer


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
    token = str(token).replace("\n", " ").strip()
    return token


def safe_output_names(single_exp):
    output_names = getattr(single_exp, "output_names", None)
    if output_names is None:
        return CLASS_NAMES
    return list(output_names)


def resolve_label_index(pred_label: str, output_names: list[str]) -> int:
    pred_label = str(pred_label).lower()

    for idx, name in enumerate(output_names):
        if str(name).lower() == pred_label:
            return idx

    if pred_label in LABEL_TO_ID:
        label_id = LABEL_TO_ID[pred_label]
        for idx, name in enumerate(output_names):
            if str(name).upper() == f"LABEL_{label_id}":
                return idx

    return LABEL_TO_ID.get(pred_label, 0)


def extract_single_example_tokens_and_values(single_exp, pred_label: str):
    """
    Надёжно извлекает токены и SHAP-значения для одного примера,
    даже если общая структура Explanation нерегулярная.
    """
    output_names = safe_output_names(single_exp)
    pred_idx = resolve_label_index(pred_label, output_names)

    tokens = list(single_exp.data)
    values = np.array(single_exp.values, dtype=object)

    # Ожидаемый случай: [tokens, classes]
    if values.ndim == 2:
        if pred_idx >= values.shape[1]:
            pred_idx = 0
        token_values = np.asarray(values[:, pred_idx], dtype=float)

    # Иногда SHAP уже возвращает одномерный вектор под один выход
    elif values.ndim == 1:
        if len(values) > 0 and isinstance(values[0], (list, tuple, np.ndarray)):
            stacked = np.array([np.array(v, dtype=float) for v in values], dtype=object)

            try:
                stacked2 = np.vstack(stacked)
                if stacked2.ndim == 2:
                    if pred_idx >= stacked2.shape[1]:
                        pred_idx = 0
                    token_values = stacked2[:, pred_idx]
                else:
                    token_values = np.asarray(stacked2, dtype=float).reshape(-1)
            except Exception:
                token_values = np.array(
                    [float(np.array(v).reshape(-1)[0]) for v in values],
                    dtype=float,
                )
        else:
            token_values = np.asarray(values, dtype=float)

    else:
        raise ValueError(f"Unsupported SHAP value shape for one example: {values.shape}")

    tokens = [clean_token(tok) for tok in tokens]

    # На всякий случай выравниваем длины
    min_len = min(len(tokens), len(token_values))
    tokens = tokens[:min_len]
    token_values = token_values[:min_len]

    return tokens, token_values


def save_token_bar_plot(example_df: pd.DataFrame, example_id: int, pred_label: str):
    plot_df = (
        example_df.sort_values("abs_shap_value", ascending=False)
        .head(12)
        .sort_values("shap_value", ascending=True)
    )

    plt.figure(figsize=(10, 6))
    plt.barh(plot_df["token"], plot_df["shap_value"])
    plt.xlabel("SHAP value")
    plt.ylabel("Token")
    plt.title(f"Top SHAP Tokens - Example {example_id} - Predicted: {pred_label}")
    plt.tight_layout()
    fig_path = SHAP_FIG_DIR / f"shap_example_{example_id}.png"
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close()


def aggregate_top_features(token_df: pd.DataFrame) -> pd.DataFrame:
    agg_df = (
        token_df.groupby(["pred_label", "token"], as_index=False)
        .agg(
            mean_shap_value=("shap_value", "mean"),
            abs_mean_shap_value=("abs_shap_value", "mean"),
            count=("token", "count"),
        )
        .sort_values(["pred_label", "abs_mean_shap_value", "count"], ascending=[True, False, False])
    )

    top_df = agg_df.groupby("pred_label", group_keys=False).head(15).reset_index(drop=True)
    top_df.to_csv(TOP_FEATURES_PATH, index=False)
    return top_df


def main():
    SHAP_DIR.mkdir(parents=True, exist_ok=True)
    SHAP_FIG_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading model and tokenizer...")
    model, tokenizer = load_model_and_tokenizer()
    classifier, masker = build_classifier(model, tokenizer)

    print("Loading misclassified examples...")
    errors_df = load_errors()
    selected_df = select_examples(errors_df, max_examples=MAX_EXAMPLES)
    selected_df.to_csv(SELECTED_EXAMPLES_PATH, index=False)

    texts = selected_df["text"].astype(str).tolist()

    print("Building SHAP explainer...")
    explainer = shap.Explainer(classifier, masker)

    print("Computing SHAP values...")
    shap_values = explainer(texts)

    rows = []

    for i in range(len(selected_df)):
        single_exp = shap_values[i]
        pred_label = selected_df.loc[i, "pred_label"]

        tokens, token_values = extract_single_example_tokens_and_values(single_exp, pred_label)

        example_rows = []

        for token, shap_value in zip(tokens, token_values):
            if token == "":
                continue

            row = {
                "example_id": i + 1,
                "text": selected_df.loc[i, "text"],
                "true_label": selected_df.loc[i, "true_label"],
                "pred_label": pred_label,
                "confidence": selected_df.loc[i, "confidence"],
                "token": token,
                "shap_value": float(shap_value),
                "abs_shap_value": float(abs(shap_value)),
            }
            rows.append(row)
            example_rows.append(row)

        example_df = pd.DataFrame(example_rows)
        if not example_df.empty:
            save_token_bar_plot(example_df, i + 1, pred_label)

        print(
            f"Processed example {i + 1}/{len(selected_df)} | "
            f"true={selected_df.loc[i, 'true_label']} | pred={pred_label}"
        )

    token_df = pd.DataFrame(rows)
    token_df.to_csv(TOKEN_ATTRIBUTIONS_PATH, index=False)

    top_df = aggregate_top_features(token_df)

    print("\nSHAP analysis completed.")
    print(f"Selected examples saved to: {SELECTED_EXAMPLES_PATH}")
    print(f"Token attributions saved to: {TOKEN_ATTRIBUTIONS_PATH}")
    print(f"Top features saved to: {TOP_FEATURES_PATH}")
    print(f"Figures directory: {SHAP_FIG_DIR}")

    print("\nTop SHAP features:")
    print(top_df.head(20))


if __name__ == "__main__":
    main()