"""Model loading: base Hugging Face model, optional 4-bit quantization, and PEFT adapter."""
import os
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from config import AKIYA_VYRE_MODEL_ID, BASE_MODEL_ID, LABEL2ID, ID2LABEL

def load_model_from_huggingface(
    model_id=BASE_MODEL_ID,
    token=None,
    use_4bit=False,
    adapter_id=None,
    num_labels=3,
):
    """
    Load a sequence-classification model from Hugging Face Hub.
    - model_id: base model repo ID, or a full fine-tuned model repo.
    - adapter_id: optional LoRA adapter repo/path; model_id is used as its base.
    - token: pass via HF_TOKEN environment variable or Hugging Face login; never hardcode it.
    """
    token = token or os.getenv("HF_TOKEN")
    tokenizer = AutoTokenizer.from_pretrained(model_id, token=token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token
    tokenizer.padding_side = "right"

    kwargs = {
        "num_labels": num_labels,
        "label2id": LABEL2ID,
        "id2label": ID2LABEL,
        "token": token,
    }
    if use_4bit:
        if not torch.cuda.is_available():
            raise RuntimeError("4-bit bitsandbytes loading requires a supported CUDA GPU.")
        compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True,
        )
        kwargs["device_map"] = "auto"
    else:
        kwargs["torch_dtype"] = (
            torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
            else (torch.float16 if torch.cuda.is_available() else torch.float32)
        )
        if torch.cuda.is_available():
            kwargs["device_map"] = "auto"

    model = AutoModelForSequenceClassification.from_pretrained(model_id, **kwargs)
    model.config.pad_token_id = tokenizer.pad_token_id
    if adapter_id:
        model = PeftModel.from_pretrained(model, adapter_id, token=token)
    model.eval()
    return tokenizer, model

def load_local_model(model_dir):
    """Convenience wrapper for a locally saved model/adapter directory."""
    return load_model_from_huggingface(model_id=model_dir, use_4bit=False)


def load_akiya_vyre_model(token=None, use_4bit=False):
    """Load the Akiya-Vyre LoRA adapter on top of its LLaMA base model."""
    return load_model_from_huggingface(
        model_id=BASE_MODEL_ID,
        token=token,
        use_4bit=use_4bit,
        adapter_id=AKIYA_VYRE_MODEL_ID,
    )
