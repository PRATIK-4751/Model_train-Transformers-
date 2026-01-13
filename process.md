╔══════════════════════════════════════════════════════════════════╗
║                    MODEL TRAINING GUIDE                          ║
║                   ══════════════════                           ║
║  ┌─┐┌─┐┬─┐┬┌─┐┌─┐  ┌─┐┬ ┬┌─┐┬─┐  ┌─┐┬─┐┌─┐┌─┐               ║
║  │  ├─┤├┬┘│├┤ └─┐  ├┤ │ │├─┤├┬┘  ├┤ ├┬┘├┤ └─┐               ║
║  └─┘┴ ┴┴└─┴└─┘└─┘  └  └─┘┴ ┴┴└─  └─┘┴└─└─┘└─┘               ║
║                                                                ║
║    Fine-tuning a Coding Model with Quantization                ║
╚══════════════════════════════════════════════════════════════════╝

Quantization means converting the model into a smaller version
that works similarly but with slightly less accurate output
and fewer parameters compared to the main model.

┌─────────────────────────────────────────────────────────────────┐
│ STEP 1: PREREQUISITES                                           │
└─────────────────────────────────────────────────────────────────┘

[✓] Need a good GPU for model training (e.g., RTX 3050 with 6GB VRAM)
    Note: In CUDA training, only VRAM works, not shared memory.

[✓] Quantized the model to 4-bit to reduce memory usage

[✓] Python environment: Python 3.10 (not 3.14!)
    (Transformers don't support newer Python versions)

[✓] Core Libraries:
    ┌─────────────────────────────────────────────────────────────┐
    │ [+] torch            → GPU acceleration (PyTorch)           │
    │ [+] transformers     → Hugging Face models                │
    │ [+] datasets         → Load training data                 │
    │ [+] accelerate       → Memory optimization                │
    │ [+] peft             → LoRA for efficient training        │
    │ [+] bitsandbytes     → 4-bit quantization                 │
    └─────────────────────────────────────────────────────────────┘

[✓] Model Used: Qwen/Qwen2.5-Coder-0.5B → Local model folder

┌─────────────────────────────────────────────────────────────────┐
│ DATASET SELECTION                                               │
└─────────────────────────────────────────────────────────────────┘

[+] Used MBPP (Mostly Basic Python Programming)
[+] 974 examples
[+] Format: Clean, no solutions in prompts
[+] Can try different datasets as your device allows

┌─────────────────────────────────────────────────────────────────┐
│ DATA PREPARATION FLOW                                           │
└─────────────────────────────────────────────────────────────────┘

Raw text → Formatting → Tokenization → Training ready

┌─────────────────────────────────────────────────────────────────┐
│ TRAINING PROCESS                                                  │
└─────────────────────────────────────────────────────────────────┘

Common Problems & Solutions:
[!] CUDA out of memory → 4-bit quantization
[!] Can't fine-tune 4-bit → LoRA adapters
[!] String format error → Fixed tokenization
[!] GPU not used → Force with .to("cuda")

┌─────────────────────────────────────────────────────────────────┐
│ CODE SNIPPETS EXPLANATION                                       │
└─────────────────────────────────────────────────────────────────┘

[+] download_model.py:
    ┌─────────────────────────────────────────────────────────────┐
    │ Downloads pre-trained model from Hugging Face               │
    │ [•] transformers lib has all HF models                      │
    │ [•] AutoModelForCausalLM = text generation model            │
    │ [•] AutoTokenizer = text to number converter                │
    │ [•] Model download:                                         │
    │     model = AutoModelForCausalLM.from_pretrained(           │
    │       "Qwen/Qwen2.5-Coder-0.5B", trust_remote_code=True)   │
    └─────────────────────────────────────────────────────────────┘

    • HOW: Connects to internet, downloads 1.8GB of weights
    • WHY "Qwen/Qwen2.5-Coder-0.5B": Small coding model for 6GB GPU
    • WHY trust_remote_code=True: Qwen has custom code needing
      permission

    • Download tokenizer (text processor)
      tokenizer = AutoTokenizer.from_pretrained(
        "Qwen/Qwen2.5-Coder-0.5B", trust_remote_code=True)
    • Saves weights and tokenizer configs locally

[+] download_dataset.py:
    • Imports dataset loader tool (datasets library)
    • Downloads MBPP dataset:
      dataset = load_dataset("code-rag-bench/mbpp", "default")
    • Saves dataset locally for offline access
      dataset.save_to_disk("data")

[+] prepare_data.py:
    • Imports necessary tools (load_from_disk, AutoTokenizer)
    • Loads tokenizer from your downloaded model
    • Loads dataset from local storage
    • Formats examples with template: "Task: [desc]\nSolution:\n[code]"
    • Tokenizes text (converts to numbers)
    • Saves prepared data for training
      tokenized.save_to_disk("prepared_data")

[+] train.py:
    • Imports training components (AutoModelForCausalLM,
      TrainingArguments, Trainer, torch)
    • Loads model and moves to GPU
    • Configures training parameters
    • Sets up Trainer with model, args, and train_dataset
    • Starts training process: trainer.train()
    • Monitors loss values and saves checkpoints

[+] Verify Model Directory:
    • Import OS module for file operations
    • Check if model exists at specified path
    • Verify weights file (.safetensors format) exists
    • Output: ✓ if found, ✗ if missing