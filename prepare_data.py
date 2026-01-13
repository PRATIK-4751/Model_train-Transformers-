# prepare_data.py
from datasets import load_from_disk
from transformers import AutoTokenizer

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("model", trust_remote_code=True)

# Load dataset
dataset = load_from_disk("data")

# Format data
def format_example(examples):
    texts = [f"Task: {t}\nSolution:\n{c}" for t, c in zip(examples['text'], examples['code'])]
    return {"text": texts}

formatted = dataset.map(format_example, batched=True)

# Tokenize WITH LABELS
def tokenize(examples):
    tokenized = tokenizer(examples["text"], truncation=True, padding="max_length", max_length=256)
    
    # ADD THIS: Create labels (same as input_ids for language modeling)
    tokenized["labels"] = tokenized["input_ids"].copy()
    
    return tokenized

tokenized = formatted.map(tokenize, batched=True)

# Save
tokenized.save_to_disk("prepared_data")
print("Data prepared with labels")