import re
import pandas as pd

from config import GENERATED_DIR


MISCLASSIFIED_PATH = GENERATED_DIR / "misclassified_examples_lstm.csv"
TAXONOMY_PATH = GENERATED_DIR / "taxonomy_lstm.csv"
TAXONOMY_SUMMARY_PATH = GENERATED_DIR / "taxonomy_summary_lstm.csv"


NEGATION_WORDS = {
    "не", "нет", "ни", "никто", "ничего", "никогда", "никакой", "никакая",
    "никакие", "никаким", "никаких", "без"
}

POSITIVE_WORDS = {
    "хорошо", "хороший", "отлично", "отличный", "прекрасно", "прекрасный",
    "супер", "класс", "круто", "нравится", "понравилось", "качественный",
    "удобно", "удобный", "красивый", "идеально", "люблю"
}

NEGATIVE_WORDS = {
    "плохо", "плохой", "ужас", "ужасно", "ужасный", "кошмар", "отвратительно",
    "отвратительный", "брак", "сломался", "сломано", "дырка", "грязный",
    "тонкий", "синтетика", "обман", "неприятно", "порван", "кривой",
    "маломерит", "большемерит", "неудобно", "не пришло", "не пришел"
}


def simple_tokenize(text: str) -> list[str]:
    text = str(text).lower().strip()
    text = re.sub(r"[^\w\sа-яё]", " ", text, flags=re.IGNORECASE)
    return text.split()


def contains_negation(tokens: list[str]) -> bool:
    return any(token in NEGATION_WORDS for token in tokens)


def contains_positive_lexicon(tokens: list[str]) -> bool:
    return any(token in POSITIVE_WORDS for token in tokens)


def contains_negative_lexicon(tokens: list[str]) -> bool:
    return any(token in NEGATIVE_WORDS for token in tokens)


def detect_error_type(text: str, text_length_tokens: int) -> tuple[str, str]:
    tokens = simple_tokenize(text)

    has_negation = contains_negation(tokens)
    has_positive = contains_positive_lexicon(tokens)
    has_negative = contains_negative_lexicon(tokens)

    if text_length_tokens <= 3:
        return "short_text", "Очень короткий текст, недостаток контекста"

    if text_length_tokens >= 40:
        return "long_text", "Длинный текст, возможная потеря ключевого контекста"

    if has_negation:
        return "negation", "В тексте присутствует отрицательная конструкция"

    if has_positive and has_negative:
        return "mixed_sentiment", "В тексте одновременно присутствуют положительные и отрицательные маркеры"

    if has_positive:
        return "lexical_positive_bias", "Модель могла опереться на положительно окрашенную лексику"

    if has_negative:
        return "lexical_negative_bias", "Модель могла опереться на отрицательно окрашенную лексику"

    return "other", "Ошибка не отнесена к основным эвристическим типам"


def main():
    df = pd.read_csv(MISCLASSIFIED_PATH)

    if df.empty:
        print("Misclassified examples file is empty.")
        return

    error_types = []
    explanations = []

    for _, row in df.iterrows():
        error_type, explanation = detect_error_type(
            text=row["text"],
            text_length_tokens=int(row["text_length_tokens"]),
        )
        error_types.append(error_type)
        explanations.append(explanation)

    df["error_type"] = error_types
    df["error_explanation"] = explanations

    summary_df = (
        df.groupby("error_type")
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
    )

    GENERATED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(TAXONOMY_PATH, index=False)
    summary_df.to_csv(TAXONOMY_SUMMARY_PATH, index=False)

    print("Taxonomy analysis completed.")
    print(f"Taxonomy file saved to: {TAXONOMY_PATH}")
    print(f"Taxonomy summary saved to: {TAXONOMY_SUMMARY_PATH}")

    print("\nError type distribution:")
    print(summary_df)

    print("\nSample taxonomy examples:")
    print(
        df[
            [
                "text",
                "true_label",
                "pred_label",
                "error_type",
                "error_explanation",
            ]
        ].head(10)
    )


if __name__ == "__main__":
    main()