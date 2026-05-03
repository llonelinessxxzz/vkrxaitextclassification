from itertools import combinations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lime.lime_text import LimeTextExplainer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.pipeline import Pipeline
from sklearn.utils import resample

from config import FIGURES_DIR, GENERATED_DIR, RANDOM_STATE, TABLES_DIR
from dataset import get_test_data, get_train_data, get_val_data


CLASS_NAMES = ["negative", "neutral", "positive"]
ID_TO_LABEL = {idx: label for idx, label in enumerate(CLASS_NAMES)}
LABEL_TO_ID = {label: idx for idx, label in ID_TO_LABEL.items()}

BASELINE_NAME = "tfidf_logreg"
BALANCING_METHODS = ["none", "class_weight", "oversampling"]

BASELINE_GENERATED_DIR = GENERATED_DIR / BASELINE_NAME
LIME_DIR = BASELINE_GENERATED_DIR / "lime_stability"
LIME_HTML_DIR = LIME_DIR / "html"
LIME_FIG_DIR = FIGURES_DIR / "lime_tfidf_logreg"

MAX_FEATURES = 50000
LIME_EXAMPLES = 12
LIME_RUNS = 5
LIME_NUM_FEATURES = 10
LIME_NUM_SAMPLES = 1000


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["review"] = df["review"].fillna("").astype(str)
    df["label_id"] = df["label_id"].astype(int)
    return df


def class_distribution(df: pd.DataFrame, split_name: str) -> pd.DataFrame:
    counts = (
        df["label_id"]
        .value_counts()
        .reindex(range(len(CLASS_NAMES)), fill_value=0)
        .rename_axis("label_id")
        .reset_index(name="count")
    )
    counts["label"] = counts["label_id"].map(ID_TO_LABEL)
    counts["split"] = split_name
    counts["share"] = counts["count"] / counts["count"].sum()
    return counts[["split", "label_id", "label", "count", "share"]]


def save_class_imbalance_stats(train_df, val_df, test_df):
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    stats_df = pd.concat(
        [
            class_distribution(train_df, "train"),
            class_distribution(val_df, "val"),
            class_distribution(test_df, "test"),
        ],
        ignore_index=True,
    )

    max_count = stats_df[stats_df["split"] == "train"]["count"].max()
    min_count = stats_df[stats_df["split"] == "train"]["count"].min()
    imbalance_ratio = max_count / min_count if min_count else np.nan
    stats_df["train_imbalance_ratio_max_to_min"] = imbalance_ratio

    stats_path = TABLES_DIR / "class_distribution_tfidf_logreg.csv"
    stats_df.to_csv(stats_path, index=False)

    pivot = stats_df.pivot(index="label", columns="split", values="count").loc[CLASS_NAMES]
    ax = pivot[["train", "val", "test"]].plot(kind="bar", figsize=(9, 5))
    ax.set_xlabel("Class")
    ax.set_ylabel("Count")
    ax.set_title("Class Distribution by Split")
    ax.tick_params(axis="x", rotation=0)
    plt.tight_layout()
    fig_path = FIGURES_DIR / "class_distribution_tfidf_logreg.png"
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close()

    return stats_df, stats_path, fig_path


def build_model(class_weight=None) -> Pipeline:
    return Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    max_features=MAX_FEATURES,
                    ngram_range=(1, 2),
                    min_df=2,
                    sublinear_tf=True,
                ),
            ),
            (
                "logreg",
                LogisticRegression(
                    C=2.0,
                    class_weight=class_weight,
                    max_iter=1000,
                    n_jobs=-1,
                    random_state=RANDOM_STATE,
                    solver="saga",
                ),
            ),
        ]
    )


def oversample_training_data(train_df: pd.DataFrame) -> pd.DataFrame:
    max_count = train_df["label_id"].value_counts().max()
    parts = []

    for label_id, part in train_df.groupby("label_id"):
        if len(part) < max_count:
            part = resample(
                part,
                replace=True,
                n_samples=max_count,
                random_state=RANDOM_STATE + int(label_id),
            )
        parts.append(part)

    balanced_df = pd.concat(parts, ignore_index=True)
    return balanced_df.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)


def train_model(train_df: pd.DataFrame, method: str) -> Pipeline:
    if method == "class_weight":
        model = build_model(class_weight="balanced")
        fit_df = train_df
    elif method == "oversampling":
        model = build_model(class_weight=None)
        fit_df = oversample_training_data(train_df)
    elif method == "none":
        model = build_model(class_weight=None)
        fit_df = train_df
    else:
        raise ValueError(f"Unknown balancing method: {method}")

    model.fit(fit_df["review"], fit_df["label_id"])
    return model


def evaluate_model(model: Pipeline, df: pd.DataFrame, method: str, split_name: str):
    y_true = df["label_id"].to_numpy()
    y_pred = model.predict(df["review"])
    proba = model.predict_proba(df["review"])

    metrics = {
        "model": BASELINE_NAME,
        "balancing_method": method,
        "split": split_name,
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "precision_weighted": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "recall_weighted": recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
    }

    report_df = pd.DataFrame(
        classification_report(
            y_true,
            y_pred,
            target_names=CLASS_NAMES,
            output_dict=True,
            zero_division=0,
        )
    ).transpose()
    report_df.insert(0, "split", split_name)
    report_df.insert(0, "balancing_method", method)

    predictions_df = df[["review", "sentiment", "label_id"]].copy()
    predictions_df["true_label"] = predictions_df["label_id"].map(ID_TO_LABEL)
    predictions_df["pred_label_id"] = y_pred
    predictions_df["pred_label"] = predictions_df["pred_label_id"].map(ID_TO_LABEL)
    predictions_df["confidence"] = proba.max(axis=1)
    for idx, label in ID_TO_LABEL.items():
        predictions_df[f"proba_{label}"] = proba[:, idx]

    return metrics, report_df, predictions_df, y_true, y_pred


def save_confusion_matrix(y_true, y_pred, method: str, split_name: str):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm)
    ax.set_xticks([0, 1, 2])
    ax.set_yticks([0, 1, 2])
    ax.set_xticklabels(CLASS_NAMES)
    ax.set_yticklabels(CLASS_NAMES)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title(f"Confusion Matrix - TF-IDF + LogReg ({method}, {split_name})")

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")

    fig.colorbar(im)
    plt.tight_layout()
    fig_path = FIGURES_DIR / f"confusion_matrix_tfidf_logreg_{method}_{split_name}.png"
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close()
    return fig_path


def save_misclassified(predictions_df: pd.DataFrame, method: str, split_name: str) -> pd.DataFrame:
    errors_df = predictions_df[predictions_df["label_id"] != predictions_df["pred_label_id"]].copy()
    errors_df["text_length_chars"] = errors_df["review"].astype(str).str.len()
    errors_df["text_length_tokens"] = errors_df["review"].astype(str).str.split().str.len()
    errors_df = errors_df.rename(columns={"review": "text", "label_id": "true_label_id"})

    errors_path = BASELINE_GENERATED_DIR / f"misclassified_examples_tfidf_logreg_{method}_{split_name}.csv"
    errors_df.to_csv(errors_path, index=False)

    summary_df = (
        errors_df.groupby(["true_label", "pred_label"])
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
    )
    summary_path = BASELINE_GENERATED_DIR / f"error_summary_tfidf_logreg_{method}_{split_name}.csv"
    summary_df.to_csv(summary_path, index=False)
    return errors_df


def plot_balancing_metrics(metrics_df: pd.DataFrame):
    test_metrics = metrics_df[metrics_df["split"] == "test"].set_index("balancing_method")
    selected = test_metrics[["accuracy", "f1_macro", "f1_weighted"]].loc[BALANCING_METHODS]

    ax = selected.plot(kind="bar", figsize=(10, 5))
    ax.set_xlabel("Balancing method")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1)
    ax.set_title("TF-IDF + Logistic Regression: Balancing Methods")
    ax.tick_params(axis="x", rotation=0)
    plt.tight_layout()
    fig_path = FIGURES_DIR / "balancing_methods_tfidf_logreg_metrics.png"
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close()
    return fig_path


def save_comparison_with_existing_models(metrics_df: pd.DataFrame):
    existing_paths = {
        "lstm": TABLES_DIR / "metrics_lstm.csv",
        "rubert": TABLES_DIR / "metrics_rubert.csv",
    }
    comparison_parts = []

    for model_name, path in existing_paths.items():
        if path.exists():
            part = pd.read_csv(path)
            part["model"] = model_name
            part["balancing_method"] = "not_applicable"
            comparison_parts.append(part)

    best_baseline = (
        metrics_df[metrics_df["split"] == "test"]
        .sort_values(["f1_macro", "accuracy"], ascending=False)
        .iloc[0]
    )
    baseline_part = pd.DataFrame(
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
                best_baseline["accuracy"],
                best_baseline["precision_macro"],
                best_baseline["recall_macro"],
                best_baseline["f1_macro"],
                best_baseline["precision_weighted"],
                best_baseline["recall_weighted"],
                best_baseline["f1_weighted"],
            ],
            "model": BASELINE_NAME,
            "balancing_method": best_baseline["balancing_method"],
        }
    )
    comparison_parts.append(baseline_part)

    comparison_df = pd.concat(comparison_parts, ignore_index=True)
    comparison_wide = comparison_df.pivot_table(
        index="metric",
        columns="model",
        values="value",
        aggfunc="first",
    ).reset_index()
    comparison_wide["tfidf_logreg_balancing_method"] = best_baseline["balancing_method"]

    comparison_path = TABLES_DIR / "model_comparison_with_tfidf_logreg.csv"
    comparison_wide.to_csv(comparison_path, index=False)

    plot_df = comparison_df[comparison_df["metric"].isin(["accuracy", "f1_macro", "f1_weighted"])]
    plot_wide = plot_df.pivot_table(index="model", columns="metric", values="value", aggfunc="first")
    plot_wide = plot_wide.reindex([BASELINE_NAME, "lstm", "rubert"]).dropna(how="all")

    ax = plot_wide.plot(kind="bar", figsize=(10, 5))
    ax.set_xlabel("Model")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1)
    ax.set_title("Model Comparison with TF-IDF + Logistic Regression")
    ax.tick_params(axis="x", rotation=0)
    plt.tight_layout()
    fig_path = FIGURES_DIR / "model_comparison_with_tfidf_logreg.png"
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close()

    return comparison_path, fig_path


def select_lime_examples(predictions_df: pd.DataFrame) -> pd.DataFrame:
    errors_df = predictions_df[predictions_df["label_id"] != predictions_df["pred_label_id"]].copy()
    errors_df = errors_df.sort_values("confidence", ascending=False)

    selected_parts = []
    for _, part in errors_df.groupby(["true_label", "pred_label"], sort=False):
        selected_parts.append(part.head(2))

    if selected_parts:
        selected_df = pd.concat(selected_parts, ignore_index=True).drop_duplicates(subset=["review"])
    else:
        selected_df = errors_df

    if len(selected_df) < LIME_EXAMPLES:
        remaining = errors_df[~errors_df["review"].isin(selected_df["review"])].head(LIME_EXAMPLES - len(selected_df))
        selected_df = pd.concat([selected_df, remaining], ignore_index=True)

    return selected_df.head(LIME_EXAMPLES).reset_index(drop=True)


def predict_proba_factory(model: Pipeline):
    def predict_proba(texts):
        return model.predict_proba(list(texts))

    return predict_proba


def pairwise_jaccard(feature_sets):
    values = []
    for left, right in combinations(feature_sets, 2):
        union = left | right
        values.append(len(left & right) / len(union) if union else 0.0)
    return float(np.mean(values)) if values else np.nan


def pairwise_weight_correlation(run_weights):
    values = []
    for left, right in combinations(run_weights, 2):
        features = sorted(set(left) | set(right))
        if len(features) < 2:
            continue
        left_values = pd.Series([left.get(feature, 0.0) for feature in features])
        right_values = pd.Series([right.get(feature, 0.0) for feature in features])
        corr = left_values.corr(right_values, method="spearman")
        if not pd.isna(corr):
            values.append(corr)
    return float(np.mean(values)) if values else np.nan


def run_lime_stability(model: Pipeline, predictions_df: pd.DataFrame, method: str):
    LIME_DIR.mkdir(parents=True, exist_ok=True)
    LIME_HTML_DIR.mkdir(parents=True, exist_ok=True)
    LIME_FIG_DIR.mkdir(parents=True, exist_ok=True)

    selected_df = select_lime_examples(predictions_df)
    selected_path = LIME_DIR / f"lime_selected_examples_tfidf_logreg_{method}.csv"
    selected_df.to_csv(selected_path, index=False)

    predict_proba = predict_proba_factory(model)
    explanation_rows = []
    stability_rows = []

    for example_idx, row in selected_df.iterrows():
        text = str(row["review"])
        pred_label = str(row["pred_label"])
        pred_label_id = LABEL_TO_ID[pred_label]
        feature_sets = []
        run_weights = []

        for run_idx in range(LIME_RUNS):
            seed = RANDOM_STATE + run_idx
            explainer = LimeTextExplainer(class_names=CLASS_NAMES, random_state=seed)
            explanation = explainer.explain_instance(
                text_instance=text,
                classifier_fn=predict_proba,
                labels=[pred_label_id],
                num_features=LIME_NUM_FEATURES,
                num_samples=LIME_NUM_SAMPLES,
            )

            html_path = LIME_HTML_DIR / f"lime_tfidf_logreg_{method}_example_{example_idx + 1}_run_{run_idx + 1}.html"
            explanation.save_to_file(str(html_path))

            if run_idx == 0:
                fig = explanation.as_pyplot_figure(label=pred_label_id)
                fig_path = LIME_FIG_DIR / f"lime_tfidf_logreg_{method}_example_{example_idx + 1}.png"
                fig.tight_layout()
                fig.savefig(fig_path, dpi=300, bbox_inches="tight")
                plt.close(fig)
            else:
                fig_path = None

            feature_list = explanation.as_list(label=pred_label_id)
            weights = {feature: float(weight) for feature, weight in feature_list}
            feature_sets.append(set(weights))
            run_weights.append(weights)

            for rank, (feature, weight) in enumerate(feature_list, start=1):
                explanation_rows.append(
                    {
                        "example_id": example_idx + 1,
                        "run_id": run_idx + 1,
                        "random_state": seed,
                        "text": text,
                        "true_label": row["true_label"],
                        "pred_label": pred_label,
                        "confidence": row["confidence"],
                        "feature_rank": rank,
                        "feature": feature,
                        "weight": weight,
                        "html_path": str(html_path),
                        "figure_path": str(fig_path) if fig_path else None,
                    }
                )

        stability_rows.append(
            {
                "example_id": example_idx + 1,
                "true_label": row["true_label"],
                "pred_label": pred_label,
                "confidence": row["confidence"],
                "lime_runs": LIME_RUNS,
                "mean_pairwise_top_feature_jaccard": pairwise_jaccard(feature_sets),
                "mean_pairwise_spearman_weight_correlation": pairwise_weight_correlation(run_weights),
                "unique_features_across_runs": len(set().union(*feature_sets)) if feature_sets else 0,
                "text": text,
            }
        )

        print(f"LIME stability: example {example_idx + 1}/{len(selected_df)}")

    explanations_df = pd.DataFrame(explanation_rows)
    explanations_path = LIME_DIR / f"lime_explanations_tfidf_logreg_{method}_multi_run.csv"
    explanations_df.to_csv(explanations_path, index=False)

    stability_df = pd.DataFrame(stability_rows)
    stability_path = LIME_DIR / f"lime_stability_summary_tfidf_logreg_{method}.csv"
    stability_df.to_csv(stability_path, index=False)

    plot_lime_stability(stability_df, method)
    return stability_df


def plot_lime_stability(stability_df: pd.DataFrame, method: str):
    if stability_df.empty:
        return None

    labels = [f"ex {example_id}" for example_id in stability_df["example_id"]]
    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(
        x - width / 2,
        stability_df["mean_pairwise_top_feature_jaccard"],
        width,
        label="Top-feature Jaccard",
    )
    ax.bar(
        x + width / 2,
        stability_df["mean_pairwise_spearman_weight_correlation"],
        width,
        label="Weight Spearman",
    )
    ax.set_xlabel("Example")
    ax.set_ylabel("Mean pairwise stability")
    ax.set_ylim(-1, 1)
    ax.set_title(f"LIME Explanation Stability - TF-IDF + LogReg ({method})")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.legend()
    plt.tight_layout()
    fig_path = FIGURES_DIR / f"lime_stability_tfidf_logreg_{method}.png"
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close()
    return fig_path


def main():
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    BASELINE_GENERATED_DIR.mkdir(parents=True, exist_ok=True)

    train_df = clean_dataframe(get_train_data())
    val_df = clean_dataframe(get_val_data())
    test_df = clean_dataframe(get_test_data())

    _, stats_path, stats_fig_path = save_class_imbalance_stats(train_df, val_df, test_df)
    print(f"Class imbalance stats saved to: {stats_path}")
    print(f"Class distribution figure saved to: {stats_fig_path}")

    all_metrics = []
    all_reports = []
    trained_models = {}
    test_predictions_by_method = {}

    for method in BALANCING_METHODS:
        print(f"\nTraining TF-IDF + Logistic Regression ({method})")
        model = train_model(train_df, method)
        trained_models[method] = model

        for split_name, split_df in [("val", val_df), ("test", test_df)]:
            metrics, report_df, predictions_df, y_true, y_pred = evaluate_model(model, split_df, method, split_name)
            all_metrics.append(metrics)
            all_reports.append(report_df)

            predictions_path = BASELINE_GENERATED_DIR / f"predictions_tfidf_logreg_{method}_{split_name}.csv"
            predictions_df.to_csv(predictions_path, index=False)
            save_misclassified(predictions_df, method, split_name)
            save_confusion_matrix(y_true, y_pred, method, split_name)

            if split_name == "test":
                test_predictions_by_method[method] = predictions_df

            print(
                f"{method} | {split_name}: "
                f"accuracy={metrics['accuracy']:.4f}, f1_macro={metrics['f1_macro']:.4f}"
            )

    metrics_df = pd.DataFrame(all_metrics)
    metrics_path = TABLES_DIR / "metrics_tfidf_logreg_balancing.csv"
    metrics_df.to_csv(metrics_path, index=False)

    report_df = pd.concat(all_reports)
    report_path = TABLES_DIR / "classification_report_tfidf_logreg_balancing.csv"
    report_df.to_csv(report_path)

    metrics_fig_path = plot_balancing_metrics(metrics_df)
    comparison_path, comparison_fig_path = save_comparison_with_existing_models(metrics_df)

    best_method = (
        metrics_df[metrics_df["split"] == "test"]
        .sort_values(["f1_macro", "accuracy"], ascending=False)
        .iloc[0]["balancing_method"]
    )
    print(f"\nBest test baseline by macro-F1: {best_method}")

    stability_df = run_lime_stability(
        trained_models[best_method],
        test_predictions_by_method[best_method],
        best_method,
    )
    stability_overview_path = TABLES_DIR / "lime_stability_tfidf_logreg_overview.csv"
    stability_df.drop(columns=["text"]).to_csv(stability_overview_path, index=False)

    print("\nSaved main files:")
    print(metrics_path)
    print(report_path)
    print(metrics_fig_path)
    print(comparison_path)
    print(comparison_fig_path)
    print(stability_overview_path)


if __name__ == "__main__":
    main()
