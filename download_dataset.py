from datasets import load_dataset

# Download MBPP dataset (only has 'default' config)

dataset = load_dataset("code-rag-bench/mbpp", "default")

# Save locally
dataset.save_to_disk("data")

print("Dataset downloaded")