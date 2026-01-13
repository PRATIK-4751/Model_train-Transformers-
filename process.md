i tried to fine tune a model  , it's a coding small model that i finetuned . you can say that the model is quantized , quantized means the model is converted into smaller model , it will works same but less better output and the parameters are less comparison to the main model 


step 1 . 


1. I need a good GPU for the model trainning as i used Rtx 3050 with 6gb  VRAM , keep in mind in CUDA trainning the VRAM will only work not the shared memory .

2. I quantized the model to 4-bit to reduce memory 

3. python env -> : Python 3.10 (not 3.14!) (Because Transformers not support the latest version of Python mostly from 9 to 11 )

4. core Libraries :-
===============================

# Essential packages:
1.  torch            # GPU acceleration (PyTorch)
2.  transformers     # Hugging Face models
3.  datasets         # Load training data
4.  accelerate       # Memory optimization
5.  peft             # LoRA for efficient training
6.  bitsandbytes     # 4-bit quantization


# The model I used 
Qwen/Qwen2.5-Coder-0.5B → Your local model folder


## Now we need to select a Dataset for fine tuning !
-> I used MBPP "Mostly Basic Python Programming"
-> 974 examples 
->format : clean and no solution in prompts 
# you can try different dataset as you wish and  the device is capable to handle 


# Data preparation flow  ! 

Raw text → Formatting → Tokenization → Training ready


Trainning process : - 

!! I have faced some problem that you might face !!
CUDA out of memory → 4-bit quantization

Can't fine-tune 4-bit → LoRA adapters

String format error → Fixed tokenization

GPU not used → .to("cuda") force


# code snippets for understanding !

for the download_model.py 

-> we are downloading the pre-trained model from Hugging face 

import the tools -> transformers library has all the hf models 
AutoModelForCasualLm  is text gen model 

AutoTokenizer   is text to number converter 

then we will download the model  

      

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-Coder-0.5B", trust_remote_code=True)


# HOW: Connects to internet, downloads 1.8GB of neural network weights
# WHY "Qwen/Qwen2.5-Coder-0.5B": Small coding model that fits 6GB GPU
# WHY trust_remote_code=True: Qwen has custom code that needs permission

# Line 5: Download tokenizer (text processor)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-0.5B", trust_remote_code=True)
# WHAT: Tokenizer splits text into pieces (tokens) and converts to numbers
# EXAMPLE: "def add():" → [1754, 123, 567, 891]

# Line 7-8: Save locally so you don't need internet next time
model.save_pretrained("model")      # Saves weights to ./model/model.safetensors
tokenizer.save_pretrained("model")  # Saves tokenizer configs to ./model/
# WHY: Faster loading, work offline, customize later


for download_dataset.py
# Line 1: Import the dataset loader tool
from datasets import load_dataset
# -> datasets library has access to thousands of ready-to-use datasets
# -> load_dataset function downloads datasets from Hugging Face Hub

# Line 3: Download the MBPP dataset
dataset = load_dataset("code-rag-bench/mbpp", "default")
# -> "code-rag-bench/mbpp" is the dataset name on Hugging Face
# -> MBPP = "Mostly Basic Python Programming" dataset
# -> "default" is the configuration name (the dataset only has one version)
# -> Contains 974 Python programming tasks with solutions

# Line 4: Save the dataset locally to your computer
dataset.save_to_disk("data")
# -> Creates a folder called "data" in your project
# -> Saves all dataset files so you don't need to download again
# -> Faster access for training, works offline

for prepare_data.py
# Line 1-2: Import necessary tools
from datasets import load_from_disk
from transformers import AutoTokenizer
# -> load_from_disk loads datasets saved locally (from download_dataset.py)
# -> AutoTokenizer loads the tokenizer from your downloaded model

# Line 4-5: Load the tokenizer from your downloaded model
tokenizer = AutoTokenizer.from_pretrained("model", trust_remote_code=True)
# -> Loads the tokenizer from "./model/" folder (where you saved it)
# -> Must use same tokenizer the model was originally trained with
# -> trust_remote_code=True needed for Qwen's custom code

# Line 7-8: Load the dataset you saved earlier
dataset = load_from_disk("data")
# -> Loads the MBPP dataset from "./data/" folder
# -> Now you have 974 training examples in memory

# Line 10-13: Function to format each example
def format_example(examples):
    texts = [f"Task: {t}\nSolution:\n{c}" for t, c in zip(examples['text'], examples['code'])]
    return {"text": texts}
# -> Creates a template: "Task: [description]\nSolution:\n[code]"
# -> zip() pairs each task description with its solution code
# -> \n creates newlines so model learns the structure
# -> Returns dictionary with "text" key containing formatted strings

# Line 15: Apply formatting to entire dataset
formatted = dataset.map(format_example, batched=True)
# -> .map() runs format_example function on every example
# -> batched=True processes 100 examples at once (faster)
# -> Creates new dataset with formatted text

# Line 17-23: Function to tokenize text (convert to numbers)
def tokenize(examples):
    tokenized = tokenizer(
        examples["text"],                # The formatted text to tokenize
        truncation=True,                 # Cut text if longer than max_length
        padding="max_length",           # Add padding if shorter than max_length
        max_length=256,                 # Maximum 256 tokens per example
        return_tensors=None             # Return Python lists (not PyTorch tensors)
    )
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized
# -> tokenizer() converts text like "Task: Write..." to numbers like [123, 456, ...]
# -> truncation=True ensures all examples are same length for training
# -> padding="max_length" fills shorter sequences with zeros
# -> max_length=256 limits memory usage (model can handle up to 2048, but we use less)
# -> labels = input_ids.copy() because model learns to predict its own input (language modeling)
# -> This is called "causal language modeling" or "next token prediction"

# Line 25: Apply tokenization to formatted dataset
tokenized = formatted.map(tokenize, batched=True)
# -> Converts all text examples to tokenized number sequences
# -> Now dataset has: input_ids, attention_mask, and labels

# Line 27: Save the prepared data
tokenized.save_to_disk("prepared_data")
# -> Saves tokenized dataset to "./prepared_data/" folder
# -> Now ready for training - contains numbers model understands

for train.py 
# Line 1-3: Import training components
from transformers import AutoModelForCausalLM, TrainingArguments, Trainer
from datasets import load_from_disk
import torch
# -> AutoModelForCausalLM loads model architecture for text generation
# -> TrainingArguments sets training configuration (epochs, batch size, etc.)
# -> Trainer handles the training loop automatically
# -> torch is PyTorch library for GPU computation

# Line 5-7: Load model and move to GPU
model = AutoModelForCausalLM.from_pretrained("model", trust_remote_code=True).to("cuda")
# -> Loads model from "./model/" folder
# -> .to("cuda") moves model from CPU to GPU memory
# -> "cuda" is NVIDIA's GPU computing platform
# -> Without this, training would use CPU (100x slower)

# Line 9-10: Load prepared data, use only 2 examples for testing
dataset = load_from_disk("prepared_data")
train_dataset = dataset["train"].select(range(2))
# -> Loads tokenized data from "./prepared_data/"
# -> .select(range(2)) takes only first 2 examples
# -> Why 2? To test if training works before committing to full training
# -> After testing, change to range(50) or range(100) for real training

# Line 12-21: Configure training parameters
training_args = TrainingArguments(
    output_dir="results",           # Directory to save training checkpoints
    num_train_epochs=1,            # Number of times to go through entire dataset
    per_device_train_batch_size=1, # Batch size per GPU (1 due to 6GB limit)
    fp16=True,                     # Use 16-bit floating point (half precision)
    save_steps=50,                 # Save model checkpoint every 50 steps
    logging_steps=1,               # Print training progress every step
    remove_unused_columns=False,   # Keep all data columns (needed for labels)
)
# -> output_dir="results": Saves model to "./results/" folder
# -> per_device_train_batch_size=1: Your 6GB GPU can only handle 1 example at a time
# -> fp16=True: Uses half the memory of normal 32-bit floats
# -> num_train_epochs=1: One pass through data is enough for testing

# Line 23-27: Set up the Trainer
trainer = Trainer(
    model=model,                    # The model to train
    args=training_args,             # Training configuration
    train_dataset=train_dataset,    # Training data (2 examples)
)
# -> Trainer automates: forward pass, loss calculation, backward pass, optimization
# -> Handles gradient accumulation, logging, checkpoint saving
# -> Abstracts away complex training loop code

# Line 29-30: Start the training process
trainer.train()
print("Training complete!")
# -> trainer.train() begins training loop
# -> For each example: predicts → compares → adjusts → repeats
# -> Saves checkpoints to "./results/checkpoint-*/"
# -> Prints loss values to monitor learning progress

for verifying the model is in the directory we can do this 

# Line 1: Import OS module for file operations
import os
# -> os module lets you check files/folders on your computer

# Line 3: Define model path
model_path = "models/Qwen2.5-Coder-0.5B"
# -> Path where model should be saved
# -> Important: Must match actual folder name exactly

# Line 5: Check if folder exists
if os.path.exists(model_path):
    # -> os.path.exists() returns True if folder exists
    
    # Line 7: List all files in folder
    files = os.listdir(model_path)
    # -> os.listdir() gets all files in directory
    
    # Line 8: Check if weights file exists
    has_weights = any(f.endswith('.safetensors') for f in files)
    # -> any() returns True if any file ends with .safetensors
    # -> .safetensors is Hugging Face's safe tensor format (like .bin)
    
    # Line 9: Print result
    print(f"Weights file: {'✓' if has_weights else '✗'}")
    # -> Shows checkmark if weights exist, X if missing
    
# Line 11-12: Handle missing folder
else:
    print(f"Model folder not found at: {model_path}")
    # -> Error message if folder doesn't exist