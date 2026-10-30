"""Central configuration for Financial PhraseBank sentiment classification."""
import os

BASE_MODEL_ID = os.getenv("BASE_MODEL_ID", "meta-llama/Llama-2-7b-hf")
AKIYA_VYRE_MODEL_ID = "Akiya-Vyre/QLora-LLama-Sentimental-Analysis"
INFERENCE_MODEL_ID = os.getenv(
    "INFERENCE_MODEL_ID", AKIYA_VYRE_MODEL_ID
)
DATASET_ID = os.getenv("DATASET_ID", "takala/financial_phrasebank")
DATASET_CONFIG = os.getenv("DATASET_CONFIG", "sentences_allagree")
MAX_LENGTH = int(os.getenv("MAX_LENGTH", "128"))
LABEL2ID = {"negative": 0, "neutral": 1, "positive": 2}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}
DEFAULT_OUTPUT_DIR = os.getenv("OUTPUT_DIR", "./artifacts/financial-llama-lora")
