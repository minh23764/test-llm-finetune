from pathlib import Path
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model

BASE_DIR = Path(__file__).parent.parent
DATA_PATH = BASE_DIR / "data" / "train-en.jsonl"
OUTPUT_DIR = BASE_DIR / "outputs" / "phi3_minh_lora"

MODEL_NAME = "microsoft/Phi-3-mini-4k-instruct"


def format_example(example):
    instr = example["instruction"]
    out = example["output"]
    example["text"] = f"User: {instr}\nAssistant: {out}"
    return example


def tokenize(example, tokenizer):
    tokens = tokenizer(
        example["text"],
        truncation=True,
        max_length=512,
        padding="max_length",
    )
    tokens["labels"] = tokens["input_ids"].copy()
    return tokens


def main():
    print(">>> Loading dataset:", DATA_PATH)
    ds_raw = load_dataset("json", data_files=str(DATA_PATH))
    train_ds = ds_raw["train"].map(format_example)

    print(">>> Load tokenizer/model:", MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    )

    model.to("cuda")

    print(">>> Tokenizing...")
    tokenized = train_ds.map(
        lambda x: tokenize(x, tokenizer),
        batched=True,
        remove_columns=train_ds.column_names
    )

    print(">>> Apply LoRA")
    lora = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        task_type="CAUSAL_LM",
        bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]
    )

    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    print(">>> Train")
    args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        per_device_train_batch_size=1,
        gradient_accumulation_steps=2,
        num_train_epochs=3,
        learning_rate=2e-4,
        save_steps=200,
        logging_steps=10,
        fp16=False,   # WINDOWS IMPORTANT
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized,
    )
    trainer.train()

    print(">>> Saving LoRA")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(OUTPUT_DIR))
    tokenizer.save_pretrained(str(OUTPUT_DIR))


if __name__ == "__main__":
    main()
