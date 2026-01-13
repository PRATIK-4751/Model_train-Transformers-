from datasets import load_dataset

dataset = load_dataset("OpenAssistant/oasst1")

train_limited = dataset["train"].shuffle(seed=42).select(range(10000))
val_limited = dataset["validation"].shuffle(seed=42).select(range(1000))

final_dataset = {
    "train": train_limited,
    "validation": val_limited
}

train_limited.save_to_disk("oasst1_train_10k")
val_limited.save_to_disk("oasst1_val_1k")

print("Saved train: 10k samples")
print("Saved validation: 1k samples")
