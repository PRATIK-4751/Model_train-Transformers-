from transformers import AutoModelForCausalLM, TrainingArguments, Trainer
from datasets import load_from_disk
import torch
import os

from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'

if __name__ == '__main__':
    print("Training...")
    
    torch.cuda.empty_cache()

    model = AutoModelForCausalLM.from_pretrained(
        "model", 
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map='auto',
        load_in_4bit=True,
        offload_folder="./offload",
        use_cache=False
    )
    
    model = prepare_model_for_kbit_training(model)
    
    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    model = get_peft_model(model, lora_config)

    train_dataset = load_from_disk("oasst1_prepared_train").select(range(1000))
    val_dataset = load_from_disk("oasst1_prepared_val").select(range(100))

    print(f"Training: {len(train_dataset)} | Validation: {len(val_dataset)}")

    training_args = TrainingArguments(
        output_dir="oasst_results",
        num_train_epochs=3,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=8,
        eval_strategy="steps",
        eval_steps=250,
        save_steps=500,
        logging_steps=50,
        fp16=True,
        learning_rate=2e-5,
        warmup_steps=100,
        optim="adamw_8bit",
        dataloader_num_workers=0,
        gradient_checkpointing=True,
        max_grad_norm=0.3,
        save_total_limit=1,
        dataloader_pin_memory=False,
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )

    print("Training started...")
    trainer.train()
    trainer.save_model("oasst_results/final_model")
    print("Done!")