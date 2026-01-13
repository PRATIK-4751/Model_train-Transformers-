from datasets import load_from_disk
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("model", trust_remote_code=True)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

train_data = load_from_disk("oasst1_train_10k")
val_data = load_from_disk("oasst1_val_1k")

def format_conversation(example):
    text = example['text']
    if 'Human:' in text and 'Assistant:' in text:
        text = text.replace('Human:', '<|user|>').replace('Assistant:', '<|assistant|>')
    formatted = f"<|system|>You are Pratik, an AI assistant.<|endofsystem|>\n{text}"
    return {"formatted_text": formatted}

train_data = train_data.map(format_conversation)
val_data = val_data.map(format_conversation)

def tokenize_function(examples):
    result = tokenizer(
        examples["formatted_text"],
        truncation=True,
        padding="max_length",
        max_length=512,
    )
    result["labels"] = result["input_ids"].copy()
    return result

train_tokenized = train_data.map(tokenize_function, batched=True, remove_columns=train_data.column_names)
val_tokenized = val_data.map(tokenize_function, batched=True, remove_columns=val_data.column_names)

train_tokenized.save_to_disk("oasst1_prepared_train")
val_tokenized.save_to_disk("oasst1_prepared_val")

print("Data prepared!")