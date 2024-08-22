"""
**Fine-Tuning a Language Model Using James Clear's Writings as the Dataset**

This Python program demonstrates the fine-tuning of a large language model (LLM) using a dataset inspired by James Clear's writings. 
The goal is to enhance the LLM's ability to respond with insights similar to those of James Clear. 
We will use the "meta-llama/Meta-Llama-3.1-8B" model for this task. 
The fine-tuning process employs Low-Rank Adaptation (LoRA) for parameter-efficient training, combined with 4-bit quantization for memory optimization. 

**Steps:**

1. **Dataset Preparation:**
    - The dataset, formatted in JSONL, contains questions and answers reflective of James Clear's style.

2. **Fine-Tuning:**
    - The LLM is fine-tuned using LoRA, focusing on adapting the model to generate responses in the style of James Clear.
    - 4-bit quantization is applied to handle memory constraints during training.

3. **Evaluation:**
    - The performance of the fine-tuned model is evaluated by generating responses to a sample question, both before and after fine-tuning.

4. **Timing Analysis:**
    - The code measures and reports the time taken for the fine-tuning process and individual inference tasks.

**Comparison:**
    - Compare the outputs before and after fine-tuning to assess the improvement in the model's ability to generate James Clear-like responses.

This program was run a NVIDIA GeForce RTX 3060 Ti

Few observations:
- Training with 500 steps took about 15 mins on this hardware. 
- Training steps impacted the quality of the model: 
  -- Less than 100 trainig steps mostly produced garbage
  -- More than 600 training steps seems to be producing over-fitting effect where a few answers were verbatim from the dataset or there were repeated answers.
  -- 500 steps worked beautifully where I was getting the responses in the tone of James Clear

"""

import json
import random
import time, datetime
from datetime import datetime
import os
import torch
import logging
import wandb
from torch.utils.data import Dataset
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments, Trainer
from peft import get_peft_model, LoraConfig, prepare_model_for_kbit_training
from datasets import load_dataset
from dotenv import load_dotenv
from accelerate import FullyShardedDataParallelPlugin, Accelerator
from torch.distributed.fsdp.fully_sharded_data_parallel import FullOptimStateDictConfig, FullStateDictConfig


# Constants
DEVICE = "cuda:0"
SPLIT_RATIO = 0.8
MODEL_NAME = "meta-llama/Meta-Llama-3.1-8B"
OUTPUT_DIR_SUFFIX = "-james_clear-finetune"
PROJECT = "james-clear-finetune"
    
# Load model and tokenizer
def load_model_and_tokenizer(model_name):
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=bnb_config, device_map=DEVICE)
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left", add_eos_token=True, add_bos_token=True)
    tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer
    
# Split dataset into training and evaluation sets
def split_dataset(jsonl_file, train_data_file, eval_data_file, split_ratio=SPLIT_RATIO):
    data = []

    try:
        with open(jsonl_file, 'r', encoding='utf-8') as file:
            data = [json.loads(line) for line in file]
    except FileNotFoundError:
        logging.error(f"File {jsonl_file} not found.")
        return
    
    random.shuffle(data)
    split_index = int(split_ratio * len(data))
    train_data = data[:split_index]
    eval_data = data[split_index:]

    # Write the training data to a new JSONL file
    with open(train_data_file, 'w', encoding='utf-8') as train_file:
        for item in train_data:
            train_file.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    # Write the evaluation data to a new JSONL file
    with open(eval_data_file, 'w', encoding='utf-8') as eval_file:
        for item in eval_data:
            eval_file.write(json.dumps(item, ensure_ascii=False) + '\n')

# Apply PEFT using LoRA
def apply_lora(model, lora_r=32, lora_alpha=64, lora_dropout=0.05):
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model = prepare_model_for_kbit_training(model)
    
    config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj", "lm_head"
        ],
        bias="none",
        lora_dropout=lora_dropout,
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, config)
    return model


# Format input data for training

def formatting_func(input):
    return f"Question: {input['prompt']}. Answer as if you were James Clear:: {input['completion']}"

# Tokenize input data
def generate_and_tokenize_prompt(prompt, tokenizer, max_length=14):
    #print("Prompt: ", prompt)
    result = tokenizer(
        formatting_func(prompt),
        truncation=True,
        max_length=max_length,
        padding="max_length",
    )
    result["labels"] = result["input_ids"].copy()
    return result

# Train the model
def train_model(model, tokenizer, train_dataset, eval_dataset, output_dir="./results", max_steps=500):
    run_name = MODEL_NAME + "-" + PROJECT
    tokenizer.pad_token = tokenizer.eos_token
    training_args = TrainingArguments(
        output_dir=output_dir,
        warmup_steps=1,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=1,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={'use_reentrant':False},
        max_steps=max_steps,
        learning_rate=2.5e-5,
        bf16=True,
        optim="paged_adamw_8bit",
        logging_dir="./logs",        # Directory for storing logs
        logging_steps=25,
        save_strategy="steps",
        save_steps=25,
        eval_strategy="steps",
        eval_steps=25,
        do_eval=True,
        report_to="wandb",           # Comment this out if you don't want to use weights & baises
        run_name=f"{run_name}-{datetime.now().strftime('%Y-%m-%d-%H-%M')}"          # Name of the W&B run (optional)
    )
    data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),

    trainer = Trainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=training_args,
        data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )
    model.config.use_cache = False
    start_time = time.time()
    trainer.train()
    
    # Calculate and print training time
    end_time = time.time()
    elapsed_time = end_time - start_time
    minutes, seconds = divmod(elapsed_time, 60)
    print(f"Total training time: {int(minutes)} minutes and {int(seconds)} seconds")

# Perform inference with the fine-tuned model
def perform_inference(model, tokenizer, prompt):
    model_input = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    model.eval()
    with torch.no_grad():
        start_time = time.time()
        outputs = model.generate(**model_input, max_new_tokens=300, repetition_penalty=1.5, do_sample=True, temperature=0.7)
        end_time = time.time()
        print(f"Inference time: {end_time - start_time:.2f} seconds")
        return tokenizer.decode(outputs[0], skip_special_tokens=True)
    
def main():

    # Load environment variables
    load_dotenv()
    huggingface_token = os.getenv('HUGGINGFACE_API_KEY')
    if not huggingface_token:
        logging.error("Hugging Face API key not found.")
        return
    wandb.init(project=PROJECT)
    
    # Input files
    input_file = os.path.join(os.path.dirname(__file__), "james_clear_qa21.jsonl")
    train_data_file = 'jc-train-data.jsonl'
    eval_data_file = 'jc-eval-data.jsonl'
    
    # Output directory
    dir_model_name = MODEL_NAME.split("/")[-1]
    output_dir = f"{dir_model_name}{OUTPUT_DIR_SUFFIX}"
    
    # Split dataset into training and evaluation sets
    split_dataset(input_file, train_data_file, eval_data_file)

    # Load datasets
    train_dataset = load_dataset('json', data_files=train_data_file, split='train')
    eval_dataset = load_dataset('json', data_files=eval_data_file, split='train')
    
    #print(train_dataset[0:3])

    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(MODEL_NAME)

    # Inference before fine-tuning
    eval_prompt = "Question: What might be some obstacles you encounter while pursuing mastery of your craft? Answer as if you were James Clear:"
    before_finetune_output = perform_inference(model, tokenizer, eval_prompt)
   

    # Apply LoRA
    model = apply_lora(model)

    # Integrate Accelerator with FSDP
    fsdp_plugin = FullyShardedDataParallelPlugin(
        state_dict_config=FullStateDictConfig(offload_to_cpu=True, rank0_only=False),
        optim_state_dict_config=FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=False),
    )

    accelerator = Accelerator(fsdp_plugin=fsdp_plugin)
    model = accelerator.prepare_model(model)


    # Tokenize datasets
    tokenized_train_dataset = train_dataset.map(lambda x: generate_and_tokenize_prompt(x, tokenizer))
    tokenized_eval_dataset = eval_dataset.map(lambda x: generate_and_tokenize_prompt(x, tokenizer))

    # Train the model
    train_model(model, tokenizer, tokenized_train_dataset, tokenized_eval_dataset, output_dir)

    # Load fine-tuned model for inference
    base_model, tokenizer = load_model_and_tokenizer(MODEL_NAME)
    from peft import PeftModel
    checkpoint_path = f"{output_dir}/checkpoint-500"
    ft_model = PeftModel.from_pretrained(base_model, checkpoint_path)
    

    # Perform inference with the fine-tuned model
    output = perform_inference(ft_model, tokenizer, eval_prompt)

    print("###########\n Inference before fine-tuning:\n", before_finetune_output)
    print(f"###########\n Fine-tuned Model Output:\n{output}\n")
    
if __name__ == "__main__":
    main()
