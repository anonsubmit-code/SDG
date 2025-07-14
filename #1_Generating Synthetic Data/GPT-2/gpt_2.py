import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer

# Join logs into one continuous string (simulate realistic sequence)
with open("kernel_trace_10K.txt", "r") as file:
    log_text = file.read()

# Make a single-column DataFrame
df = pd.DataFrame({"text": [log_text]})
dataset = Dataset.from_pandas(df)

# Load tokenizer and set pad token
tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

# Tokenize in blocks for causal LM
def tokenize_function(examples):
    return tokenizer(examples["text"])

tokenized_dataset = dataset.map(tokenize_function, batched=True)

# Group into blocks of 512 tokens (or 1024 if using full GPT-2 capacity)
block_size = 512

def group_texts(examples):
    concatenated = sum(examples["input_ids"], [])
    total_length = (len(concatenated) // block_size) * block_size
    result = {
        "input_ids": [concatenated[i:i+block_size] for i in range(0, total_length, block_size)],
        "attention_mask": [ [1]*block_size ] * (total_length // block_size),
    }
    result["labels"] = result["input_ids"].copy()
    return result

lm_dataset = tokenized_dataset.map(group_texts, batched=True, remove_columns=["text"])

from transformers import AutoModelForCausalLM, Trainer, TrainingArguments

model = AutoModelForCausalLM.from_pretrained("gpt2")

training_args = TrainingArguments(
    output_dir="results",
    overwrite_output_dir=True,
    evaluation_strategy="no",
    learning_rate=5e-5,
    per_device_train_batch_size=2,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="logs",
    save_total_limit=1,
    save_strategy="no"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=lm_dataset,
)

trainer.train()
model.save_pretrained("linuxLogsModel_10K")
tokenizer.save_pretrained("linuxLogsModel_10K")
