"""Inference helpers shared by Streamlit and scripts."""
import torch
from model import load_model_from_huggingface

def predict_sentiment(text, tokenizer, model, max_length=128):
    if not text or not text.strip():
        raise ValueError("Please enter a financial text.")
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
    try:
        device = model.device
        inputs = {key: value.to(device) for key, value in inputs.items()}
    except (AttributeError, RuntimeError):
        pass
    with torch.inference_mode():
        logits = model(**inputs).logits
        probabilities = torch.softmax(logits, dim=-1)[0].detach().cpu()
    id2label = getattr(model.config, "id2label", None) or {0:"negative", 1:"neutral", 2:"positive"}
    # Transformers sometimes stores label keys as strings.
    label = id2label.get(int(probabilities.argmax()), id2label.get(str(int(probabilities.argmax())), "unknown"))
    scores = {}
    for i, score in enumerate(probabilities.tolist()):
        name = id2label.get(i, id2label.get(str(i), str(i)))
        scores[str(name).lower()] = float(score)
    return {"label": str(label).lower(), "confidence": float(probabilities.max()), "probabilities": scores}

def predict_with_hub(text, model_id, token=None, adapter_id=None, use_4bit=False):
    tokenizer, model = load_model_from_huggingface(
        model_id=model_id, token=token, adapter_id=adapter_id, use_4bit=use_4bit
    )
    return predict_sentiment(text, tokenizer, model)
