from transformers import AutoModelForCausalLM, AutoTokenizer

# Download Qwen model
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-Coder-0.5B", trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-0.5B", trust_remote_code=True)

# Save locally

model.save_pretrained("model")
tokenizer.save_pretrained("model")

print("Model downloaded")