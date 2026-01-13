# test_longer.py
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model = AutoModelForCausalLM.from_pretrained("results/checkpoint-500", trust_remote_code=True).to("cuda")
tokenizer = AutoTokenizer.from_pretrained("model", trust_remote_code=True, fix_mistral_regex=True)

prompt = "Task: Write a function to solve two sum problem with hashmap\nSolution:\n"

inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=200,      # Increased from 100
        temperature=0.1,         # Low temp = more deterministic
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,  # Stop at EOS token
    )

full = tokenizer.decode(outputs[0], skip_special_tokens=True)
code = full[len(prompt):].strip()

print("Generated Code:")
print("```python")
print(code)
print("```")

# Check completeness
if code.count('def ') == code.count('return '):
    print("✅ Complete function")
else:
    print("⚠️  Might be incomplete")