from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-Coder-1.5B", trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-1.5B", trust_remote_code=True)

model.save_pretrained("model")
tokenizer.save_pretrained("model")

print("Model downloaded")