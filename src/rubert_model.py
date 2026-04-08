from transformers import AutoModelForSequenceClassification, AutoTokenizer


MODEL_NAME = "DeepPavlov/rubert-base-cased"
NUM_LABELS = 3


def get_rubert_tokenizer(model_name: str = MODEL_NAME):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return tokenizer


def get_rubert_model(model_name: str = MODEL_NAME, num_labels: int = NUM_LABELS):
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
    )
    return model