"""Dataset loading, splitting, and tokenization."""
from datasets import DatasetDict, load_dataset
from config import DATASET_ID, DATASET_CONFIG, LABEL2ID, MAX_LENGTH

def load_financial_phrasebank(dataset_id=DATASET_ID, config_name=DATASET_CONFIG, seed=42):
    """Load the highest-agreement Financial PhraseBank split and create 80/10/10 splits."""
    raw = load_dataset(dataset_id, config_name)
    source = raw["train"]
    first = source.train_test_split(test_size=0.2, seed=seed)
    second = first["test"].train_test_split(test_size=0.5, seed=seed)
    return DatasetDict({
        "train": first["train"],
        "validation": second["train"],
        "test": second["test"],
    })

def tokenize_datasets(dataset_dict, tokenizer, max_length=MAX_LENGTH):
    """Tokenize each split. Dataset label IDs are expected to match negative/neutral/positive."""
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token
    tokenizer.padding_side = "right"

    def tokenize_batch(batch):
        return tokenizer(batch["sentence"], truncation=True, max_length=max_length)

    tokenized = dataset_dict.map(tokenize_batch, batched=True)
    return tokenized
