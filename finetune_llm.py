#This script uses huggingface/transformers and peft with bitsandbytes for QLoRA

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments
from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training
from trl import SFTTrainer
from datasets import Dataset
import json
import os
from tqdm import tqdm

# --- Configuration ---
MODEL_NAME = "microsoft/phi-2" # Or "mistralai/Mistral-7B-Instruct-v0.2", "microsoft/Phi-3-mini-4k-instruct" (ensure compatibility)
ADAPTERS_OUTPUT_DIR = "models/fine_tuned_llm_adapters"
FINE_TUNING_DATA_PATH = "finetuning_data/cyber_qa_pairs.jsonl" # Your tiny dataset

# QLoRA configuration
NF4_CONFIG = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16 # Use bfloat16 for better performance on modern GPUs (T4 supports it)
)

# LoRA configuration
LORA_CONFIG = LoraConfig(
    r=16, # Rank, keep low for free tier
    lora_alpha=32, # Alpha, typically 2*r
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"], # Common target modules for attention and feed-forward
    bias="none",
    task_type="CAUSAL_LM",
)

# Training arguments
TRAINING_ARGS = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3, # Keep low, 1-3 epochs
    per_device_train_batch_size=1, # Batch size 1 is common for QLoRA on small GPUs
    gradient_accumulation_steps=4, # Accumulate gradients to simulate larger batch size (1*4=4)
    optim="paged_adamw_8bit", # Optimized AdamW for quantized models
    logging_steps=10,
    learning_rate=2e-4,
    fp16=False, # Set to True if using NVIDIA GPU, bfloat16 is better if supported
    bf16=True, # Use bfloat16 for supported GPUs (T4), overrides fp16
    save_strategy="epoch",
    do_eval=False, # Disable evaluation to save time/compute
    gradient_checkpointing=True, # Saves memory, potentially slower
    report_to="none" # Disable logging to external services
)

# --- Data Preparation for Fine-tuning ---
def load_fine_tuning_data(filepath):
    """Loads a tiny dataset of instruction-response pairs."""
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data

def format_prompt(sample):
    """Formats a sample into a conversational prompt for instruction tuning."""
    # This format depends on the base model. Phi-2/3 typically uses:
    # "Instruction: {instruction}\nOutput: {output}"
    # Mistral uses: "<s>[INST] {instruction} [/INST] {output}</s>"
    # Adjust according to your chosen LLM's chat template.
    if MODEL_NAME == "microsoft/phi-2":
        return f"Instruction: {sample['instruction']}\nOutput: {sample['output']}"
    elif "mistral" in MODEL_NAME.lower(): # For Mistral models
        return f"<s>[INST] {sample['instruction']} [/INST] {sample['output']}</s>"
    elif "phi-3" in MODEL_NAME.lower(): # For Phi-3 models
        return f"<|user|>\n{sample['instruction']}<|end|>\n<|assistant|>\n{sample['output']}<|end|>"
    else:
        raise ValueError(f"Unknown model name {MODEL_NAME} for prompt formatting.")

if __name__ == "__main__":
    print(f"Starting fine-tuning for {MODEL_NAME}...")

    # Load data
    raw_data = load_fine_tuning_data(FINE_TUNING_DATA_PATH)
    if not raw_data:
        print(f"Error: No data found in {FINE_TUNING_DATA_PATH}. Please populate it.")
        exit()
    
    # Create Hugging Face Dataset
    data = [{"text": format_prompt(sample)} for sample in raw_data]
    dataset = Dataset.from_list(data)
    print(f"Loaded {len(dataset)} samples for fine-tuning.")

    # Load base model and tokenizer
    print(f"Loading base model {MODEL_NAME} with 4-bit quantization...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=NF4_CONFIG,
        trust_remote_code=True,
        device_map="auto" # Auto-map to GPU if available, otherwise CPU
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token # Important for some models
    tokenizer.padding_side = "right" # Important for training

    # Prepare model for k-bit training
    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)
    model.config.use_cache = False # Disable cache for gradient checkpointing

    # Add LoRA adapters
    model = PeftModel(model, LORA_CONFIG) # Apply LoRA config
    model.print_trainable_parameters() # Show how many parameters are trainable

    # Setup SFTTrainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=LORA_CONFIG,
        tokenizer=tokenizer,
        args=TRAINING_ARGS,
        max_seq_length=512, # Keep sequence length low to save memory
        packing=False, # Set to True for higher throughput if dataset allows, but can be complex for small datasets
    )

    # Train
    print("Starting training...")
    trainer.train()

    # Save fine-tuned adapters
    os.makedirs(ADAPTERS_OUTPUT_DIR, exist_ok=True)
    trainer.model.save_pretrained(ADAPTERS_OUTPUT_DIR)
    print(f"Fine-tuned LoRA adapters saved to {ADAPTERS_OUTPUT_DIR}")

    # Optional: Merge and save the full model (requires more RAM, not for free tier inference)
    # print("Merging adapters and saving full model (requires significant RAM)...")
    # merged_model = model.merge_and_unload()
    # merged_model_path = os.path.join(ADAPTERS_OUTPUT_DIR, "merged_model")
    # merged_model.save_pretrained(merged_model_path)
    # tokenizer.save_pretrained(merged_model_path)
    # print(f"Merged model saved to {merged_model_path}")
