from transformers import AutoModelForCausalLM, TrainingArguments, Trainer
from datasets import load_from_disk

# Load model
model = AutoModelForCausalLM.from_pretrained("model", trust_remote_code=True)

# Load your prepared data
dataset = load_from_disk("prepared_data")

# Use only 2 examples
train_dataset = dataset["train"].select(range(500))

# Check what's in the data
print("Example keys:", train_dataset[0].keys())

# Convert to correct format if needed
def convert_batch(examples):
    # Ensure all are lists (not nested lists)
    batch = {}
    for key in ["input_ids", "attention_mask", "labels"]:
        if key in examples:
            # Flatten if nested
            if isinstance(examples[key][0], list):
                batch[key] = examples[key]
            else:
                batch[key] = [examples[key]]
    return batch

# Simple training
training_args = TrainingArguments(
    output_dir="results",
    num_train_epochs=1,
    per_device_train_batch_size=1,
    fp16=True,
    logging_steps=1,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
)

print("Starting training...")
trainer.train()
print("Training complete!")