import pandas as pd
from datasets import load_dataset
from unsloth import FastLanguageModel
from peft import LoraConfig, get_peft_model
from transformers import TrainingArguments
from trl import SFTTrainer

# Load Excel file (ensure it's in the same directory as this script)
file_name = "SCHEMA file data (Recovered) (1).xlsx"

# Read and preprocess
df = pd.read_excel(file_name, sheet_name="questions_queries ", header=2)
df.columns = df.columns.str.strip()
df = df.rename(columns={"Question (Natural Language)": "prompt", "SQL Query": "response"})

# Save to JSONL
with open("nl_sql_data.jsonl", "w", encoding="utf-8") as f:
    for _, row in df.iterrows():
        prompt = str(row["prompt"]).replace('"', "'").strip()
        response = str(row["response"]).replace('"', "'").strip()
        f.write(f'{{"prompt": "{prompt}", "response": "{response}"}}\n')

# Load dataset
dataset = load_dataset("json", data_files="nl_sql_data.jsonl", split="train")

# Load base model
model_name = "unsloth/phi-3.5-mini-instruct"
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name,
    max_seq_length=256,
    dtype=None,
    load_in_4bit=True
)

# PEFT config
peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "fc1", "fc2"]
)

model = get_peft_model(model, peft_config)

# Tokenization function
def tokenize(example):
    tokens = tokenizer(
        example["prompt"],
        padding="max_length",
        truncation=True,
        max_length=256,
        return_token_type_ids=False,
    )
    tokens["labels"] = tokens["input_ids"].copy()
    return tokens

# Tokenize dataset
dataset = dataset.map(tokenize)

# Training arguments
training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    learning_rate=2e-5,
    logging_dir="./logs",
    logging_steps=10,
    save_steps=500,
    num_train_epochs=3,
    report_to="none",
    fp16=True  # Or set bf16=True if supported
)

# Trainer
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    args=training_args,
    max_seq_length=256
)

# Train the model
trainer.train()
