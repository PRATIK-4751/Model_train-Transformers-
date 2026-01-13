from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

tokenizer = AutoTokenizer.from_pretrained("model", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("oasst_results/final_model", trust_remote_code=True).to("cuda")

print("""
╔═══════════════════════════════════════╗
║   Pratik AI Assistant (プラティク)    ║
╚═══════════════════════════════════════╝
""")

chat_history = []

while True:
    user_input = input("\nYou: ").strip()
    
    if not user_input or user_input.lower() in ["exit", "quit"]:
        print("\nPratik: Goodbye!")
        break
    
    chat_history.append(f"<|user|>{user_input}<|endofuser|>")
    
    conversation = "<|system|>You are Pratik, an AI assistant.<|endofsystem|>\n"
    conversation += "\n".join(chat_history[-3:])
    conversation += "\n<|assistant|>"
    
    inputs = tokenizer(conversation, return_tensors="pt", truncation=True, max_length=512).to("cuda")
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=150,
            temperature=0.7,
            do_sample=True,
            top_p=0.9,
            repetition_penalty=1.2,
            pad_token_id=tokenizer.eos_token_id,
        )
    
    full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    ai_response = full_response.split("<|assistant|>")[-1].strip()
    
    if not ai_response:
        ai_response = "Could you rephrase that?"
    
    print(f"\nPratik: {ai_response}")
    chat_history.append(f"<|assistant|>{ai_response}<|endofassistant|>")