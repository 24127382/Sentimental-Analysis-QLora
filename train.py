"""Training entry points for QLoRA/LoRA and full fine-tuning."""
import os
import numpy as np
import torch
from transformers import Trainer, TrainingArguments, DataCollatorWithPadding
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from config import BASE_MODEL_ID, DEFAULT_OUTPUT_DIR, ID2LABEL, LABEL2ID
from data import load_financial_phrasebank, tokenize_datasets
from model import load_model_from_huggingface

def compute_metrics(eval_pred):
    from sklearn.metrics import accuracy_score, f1_score
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "f1_macro": float(f1_score(labels, predictions, average="macro")),
    }

def train_qlora(
    model_id=BASE_MODEL_ID,
    output_dir=DEFAULT_OUTPUT_DIR,
    epochs=3,
    learning_rate=1e-5,
    train_batch_size=4,
    eval_batch_size=4,
    gradient_accumulation_steps=2,
    max_length=128,
    token=None,
):
    """Train a 4-bit QLoRA classifier. Requires a supported CUDA GPU and HF access if gated."""
    if not torch.cuda.is_available():
        raise RuntimeError("QLoRA requires a supported CUDA GPU. Use Colab GPU or train a smaller model.")
    token = token or os.getenv("HF_TOKEN")
    tokenizer, model = load_model_from_huggingface(
        model_id=model_id, token=token, use_4bit=True
    )
    model = prepare_model_for_kbit_training(model)
    peft_config = LoraConfig(
        r=8, lora_alpha=16, target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05, bias="none", task_type="SEQ_CLS", modules_to_save=["score"]
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    datasets = load_financial_phrasebank()
    tokenized = tokenize_datasets(datasets, tokenizer, max_length=max_length)
    collator = DataCollatorWithPadding(tokenizer=tokenizer)
    args = TrainingArguments(
        output_dir=output_dir, learning_rate=learning_rate,
        per_device_train_batch_size=train_batch_size,
        per_device_eval_batch_size=eval_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        num_train_epochs=epochs, weight_decay=0.01,
        eval_strategy="epoch", save_strategy="epoch",
        load_best_model_at_end=True, metric_for_best_model="f1_macro",
        greater_is_better=True, fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(), logging_steps=10, report_to="none",
    )
    trainer = Trainer(
        model=model, args=args, train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"], processing_class=tokenizer,
        data_collator=collator, compute_metrics=compute_metrics,
    )
    trainer.train()
    test_result = trainer.evaluate(tokenized["test"], metric_key_prefix="test")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    return trainer, test_result

def train_full_finetune(
    model_id=BASE_MODEL_ID,
    output_dir="./artifacts/financial-llama-full",
    epochs=3,
    learning_rate=1e-5,
    train_batch_size=1,
    eval_batch_size=1,
    max_length=128,
    token=None,
):
    """Full fine-tuning; very memory-intensive for LLaMA-2 7B, even with a GPU."""
    token = token or os.getenv("HF_TOKEN")
    tokenizer, model = load_model_from_huggingface(model_id=model_id, token=token, use_4bit=False)
    datasets = tokenize_datasets(load_financial_phrasebank(), tokenizer, max_length=max_length)
    args = TrainingArguments(
        output_dir=output_dir, learning_rate=learning_rate,
        per_device_train_batch_size=train_batch_size,
        per_device_eval_batch_size=eval_batch_size,
        gradient_accumulation_steps=8, num_train_epochs=epochs, weight_decay=0.01,
        eval_strategy="epoch", save_strategy="epoch", load_best_model_at_end=True,
        metric_for_best_model="f1_macro", fp16=torch.cuda.is_available() and not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        logging_steps=10, report_to="none",
    )
    trainer = Trainer(
        model=model, args=args, train_dataset=datasets["train"],
        eval_dataset=datasets["validation"], processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer), compute_metrics=compute_metrics,
    )
    trainer.train()
    test_result = trainer.evaluate(datasets["test"], metric_key_prefix="test")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    return trainer, test_result
