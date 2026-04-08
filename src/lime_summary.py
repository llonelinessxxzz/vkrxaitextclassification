import pandas as pd
import matplotlib.pyplot as plt

from config import FIGURES_DIR, GENERATED_DIR


LIME_SUMMARY_PATH = GENERATED_DIR / "lime_lstm" / "lime_explanations_lstm.csv"

TOP_FEATURES_ALL_PATH = GENERATED_DIR / "lime_lstm" / "lime_top_features_all.csv"
TOP_FEATURES_BY_PRED_PATH = GENERATED_DIR / "lime_lstm" / "lime_top_features_by_pred.csv"

TOP_POSITIVE_FIG_PATH = FIGURES_DIR / "lime_top_positive_features_lstm.png"
TOP_NEGATIVE_FIG_PATH = FIGURES_DIR / "lime_top_negative_features_lstm.png"
TOP_BY_CLASS_FIG_PATH = FIGURES_DIR / "lime_top_features_by_pred_class_lstm.png"


def load_lime_summary() -> pd.DataFrame:
    df = pd.read_csv(LIME_SUMMARY_PATH)
    return df


def aggregate_top_features(df: pd.DataFrame, top_n: int = 15):
    agg_df = (
        df.groupby("feature", as_index=False)
        .agg(
            mean_weight=("weight", "mean"),
            abs_mean_weight=("weight", lambda x: x.abs().mean()),
            count=("feature", "count"),
        )
        .sort_values(["abs_mean_weight", "count"], ascending=[False, False])
    )

    agg_df.to_csv(TOP_FEATURES_ALL_PATH, index=False)

    top_positive = agg_df.sort_values("mean_weight", ascending=False).head(top_n)
    top_negative = agg_df.sort_values("mean_weight", ascending=True).head(top_n)

    return agg_df, top_positive, top_negative


def aggregate_by_predicted_class(df: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    grouped = (
        df.groupby(["pred_label", "feature"], as_index=False)
        .agg(
            mean_weight=("weight", "mean"),
            abs_mean_weight=("weight", lambda x: x.abs().mean()),
            count=("feature", "count"),
        )
        .sort_values(["pred_label", "abs_mean_weight", "count"], ascending=[True, False, False])
    )

    top_by_class = (
        grouped.groupby("pred_label", group_keys=False)
        .head(top_n)
        .reset_index(drop=True)
    )

    top_by_class.to_csv(TOP_FEATURES_BY_PRED_PATH, index=False)
    return top_by_class


def plot_top_positive(top_positive: pd.DataFrame):
    plt.figure(figsize=(10, 6))
    plt.barh(top_positive["feature"], top_positive["mean_weight"])
    plt.xlabel("Mean LIME weight")
    plt.ylabel("Feature")
    plt.title("Top Tokens Supporting Wrong Predictions - Positive Direction")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(TOP_POSITIVE_FIG_PATH, dpi=300, bbox_inches="tight")
    plt.close()


def plot_top_negative(top_negative: pd.DataFrame):
    plt.figure(figsize=(10, 6))
    plt.barh(top_negative["feature"], top_negative["mean_weight"])
    plt.xlabel("Mean LIME weight")
    plt.ylabel("Feature")
    plt.title("Top Tokens Supporting Wrong Predictions - Negative Direction")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(TOP_NEGATIVE_FIG_PATH, dpi=300, bbox_inches="tight")
    plt.close()


def plot_top_by_class(top_by_class: pd.DataFrame):
    if top_by_class.empty:
        return

    class_names = top_by_class["pred_label"].unique().tolist()

    fig, axes = plt.subplots(len(class_names), 1, figsize=(10, 5 * len(class_names)))
    if len(class_names) == 1:
        axes = [axes]

    for ax, class_name in zip(axes, class_names):
        subset = top_by_class[top_by_class["pred_label"] == class_name].copy()
        subset = subset.sort_values("abs_mean_weight", ascending=True)

        ax.barh(subset["feature"], subset["abs_mean_weight"])
        ax.set_title(f"Top LIME Features for Wrong Predicted Class: {class_name}")
        ax.set_xlabel("Absolute mean LIME weight")
        ax.set_ylabel("Feature")

    plt.tight_layout()
    plt.savefig(TOP_BY_CLASS_FIG_PATH, dpi=300, bbox_inches="tight")
    plt.close()


def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    (GENERATED_DIR / "lime_lstm").mkdir(parents=True, exist_ok=True)

    df = load_lime_summary()

    agg_df, top_positive, top_negative = aggregate_top_features(df, top_n=15)
    top_by_class = aggregate_by_predicted_class(df, top_n=10)

    plot_top_positive(top_positive)
    plot_top_negative(top_negative)
    plot_top_by_class(top_by_class)

    print("LIME summary aggregation completed.\n")

    print("Top overall features:")
    print(agg_df.head(15))

    print("\nTop positive-direction features:")
    print(top_positive[["feature", "mean_weight", "count"]])

    print("\nTop negative-direction features:")
    print(top_negative[["feature", "mean_weight", "count"]])

    print("\nTop features by predicted class:")
    print(top_by_class[["pred_label", "feature", "abs_mean_weight", "count"]].head(20))

    print("\nSaved files:")
    print(TOP_FEATURES_ALL_PATH)
    print(TOP_FEATURES_BY_PRED_PATH)
    print(TOP_POSITIVE_FIG_PATH)
    print(TOP_NEGATIVE_FIG_PATH)
    print(TOP_BY_CLASS_FIG_PATH)


if __name__ == "__main__":
    main()