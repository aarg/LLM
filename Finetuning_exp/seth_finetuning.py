"""
**Finetuning demonstration using Seth Godin Blogs as the dataset**

This Python program demonstrates the impact of finetuning on a large language model (LLM). 
We will use the "seth godin blogs dataset" to illustrate how fine-tuning improves the LLM's ability to answer questions like Seth Godin. We will be using "meta-llama/Llama-2-7b-hf" model for finetuning. QLoRA (Quantized Low-Rank Adaptation) for fine-tuning, this is required to fit the model in the available memory.  

**Steps:**

1. **Prepare the Dataset:**
    - Download and pre-process the "seth godin blogs dataset" from [here](https://www.kaggle.com/datasets/glushko/seth-godins-blogs-dataset).
    - We will  go through cleaning, formatting and tokenizing the text for the chosen LLM.
2. **Fine-Tuning:**
    - Define a fine-tuning task. In this example, we want LLM to respond like the personality of Seth Godin and say things he's likely to say. 
    - Split the dataset into training and validation
    - Fine-tune the LLM on the training set. Monitor performance on the validation set during training.
3. **Evaluation:**
    - Evaluate the performance of the fine-tuned model using an example question. 
4. **Comparison:**
    - Compare the results of the output before and after finetuning on an unseen Q&A.

"""

'''
This program was run on:

Hardware: NVIDIA GeForce RTX 3060 Ti
For me, this required specific versions of libraries to work correctly: 
- Build cuda_12.1.r12.1/. Installed from Cuda website.
- pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
- pip3 install bitsandbytes==0.41.2.post2 --prefer-binary --extra-index-url=https://jllllll.github.io/bitsandbytes-windows-webui

'''


####### Imports #######

import csv
import json
import random
import datasets


####### Data Prep #######
# Copy the raw dataset file as: '/content/seth-data.csv'

csv_file = './content/seth-data.csv'
jsonl_file = './content/seth-data.jsonl'
train_data_file = './content/seth-train-data.jsonl'
eval_data_file = './content/seth-eval-data.jsonl'

# Using 2 columns from the blogs: 1. Title 2. Content as training dataset. Convert to .jsonl format for easier manipulation
def convert_csv_to_jsonl(csv_file, jsonl_file):
    with open(csv_file, 'r', newline='', encoding='utf-8') as csvfile, open(jsonl_file, 'w', encoding='utf-8') as jsonlfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            json_data = {'title': row['title'], 'content': row['content_plain']}
            jsonlfile.write(json.dumps(json_data) + '\n')

convert_csv_to_jsonl(csv_file, jsonl_file)



# 80:20 split the jsonl file into training and eval set
data = []
with open(jsonl_file, 'r', encoding='utf-8') as file:
    for line in file:
        data.append(json.loads(line))

# Shuffle the data
random.shuffle(data)

# Split the data into training and evaluation sets
split_index = int(0.8 * len(data))
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


# Uncomment to see the sample output from files

'''
# Function to read the first 5 entries from a JSON Lines file
def print_first_5_entries(file_name):
    with open(file_name, 'r', encoding='utf-8') as datafile:
        for _ in range(5):
            line = datafile.readline()
            if not line:
                break
            data = json.loads(line)
            print(data)

print(f'\n First 5 lines of full dataset: \n')
#print_first_5_entries(jsonl_file)

print(f'\n First 5 lines of training dataset: \n')
#print_first_5_entries(train_data_file)

print(f'\n First 5 lines of Eval dataset: \n')
#print_first_5_entries(eval_data_file)
''' 

#load dataset
from datasets import load_dataset

train_dataset = load_dataset('json', data_files=train_data_file, split='train')
eval_dataset = load_dataset('json', data_files=eval_data_file, split='train')

def formatting_func(input):
    text = f"### Seth would say: # {input['title']}"
    return text

#example_input = train_dataset[0]
#print(example_input)
#formatted_input = formatting_func(example_input)
#print(formatted_input)

# Load quantized model
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

base_model_id = "meta-llama/Llama-2-7b-hf"
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

model = AutoModelForCausalLM.from_pretrained(base_model_id, quantization_config=bnb_config, device_map="cuda:0")

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    base_model_id,
    padding_side="left",
    add_eos_token=True,
    add_bos_token=True,
)
tokenizer.pad_token = tokenizer.eos_token

def generate_and_tokenize_prompt(prompt):
    return tokenizer(formatting_func(prompt))

tokenized_train_dataset = train_dataset.map(generate_and_tokenize_prompt)
tokenized_val_dataset = eval_dataset.map(generate_and_tokenize_prompt)

#print(tokenized_train_dataset[1])


import matplotlib.pyplot as plt

def plot_data_lengths(tokenize_train_dataset, tokenized_val_dataset):
    lengths = [len(x['input_ids']) for x in tokenized_train_dataset]
    lengths += [len(x['input_ids']) for x in tokenized_val_dataset]
    print(len(lengths))

    # Plotting the histogram
    plt.figure(figsize=(10, 6))
    plt.hist(lengths, bins=20, alpha=0.7, color='blue')
    plt.xlabel('Length of input_ids')
    plt.ylabel('Frequency')
    plt.title('Distribution of Lengths of input_ids')
    plt.show()

#plot the dataset to see the distributiion of tokens
#plot_data_lengths(tokenized_train_dataset, tokenized_val_dataset)

max_length = 14 # This was an appropriate max length for my dataset, somewhere in the middle of distribution

def generate_and_tokenize_prompt2(prompt):
    result = tokenizer(
        formatting_func(prompt),
        truncation=True,
        max_length=max_length,
        padding="max_length",
    )
    result["labels"] = result["input_ids"].copy()
    return result

tokenized_train_dataset = train_dataset.map(generate_and_tokenize_prompt2)
tokenized_val_dataset = eval_dataset.map(generate_and_tokenize_prompt2)

#print(tokenized_train_dataset[1]['input_ids'])

#plot_data_lengths(tokenized_train_dataset, tokenized_val_dataset)

# Get an output from the pretrained model. We will ask the same question to the finetuned model later. 
print("\n##########")
eval_prompt = "What does Seth Godin say about getting unstuck? \n"

model_input = tokenizer(eval_prompt, return_tensors="pt").to("cuda:0")

model.eval()
with torch.no_grad():
    print(tokenizer.decode(model.generate(**model_input, max_new_tokens=300, pad_token_id=2)[0], skip_special_tokens=True))

print("\n##########\n")

# Use PEFT (Parameter-Efficient Fine-Tuning) to only fine-tunes a small subset of additional parameters compared to the entire LLM. 

from peft import prepare_model_for_kbit_training

model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model)

def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )

#print(model)

from accelerate import FullyShardedDataParallelPlugin, Accelerator
from torch.distributed.fsdp.fully_sharded_data_parallel import FullOptimStateDictConfig, FullStateDictConfig

fsdp_plugin = FullyShardedDataParallelPlugin(
    state_dict_config=FullStateDictConfig(offload_to_cpu=True, rank0_only=False),
    optim_state_dict_config=FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=False),
)

accelerator = Accelerator(fsdp_plugin=fsdp_plugin)

from peft import LoraConfig, get_peft_model

#Load LoRA config
config = LoraConfig(
    r=32,
    lora_alpha=64,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
        "lm_head",
    ],
    bias="none",
    lora_dropout=0.05,  # Conventional
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, config)
print_trainable_parameters(model)

# Apply the accelerator. You can comment this out to remove the accelerator.
model = accelerator.prepare_model(model)

#print(model)

# Uncomment the following to sync results to wandb account
'''
import wandb, os
wandb.login()

wandb_project = "seth-finetune"
if len(wandb_project) > 0:
    os.environ["WANDB_PROJECT"] = wandb_project
''' 

if torch.cuda.device_count() > 1: # If more than 1 GPU
    model.is_parallelizable = True
    model.model_parallel = True

print("##### BEGIN TRAINING ######")
    
import transformers
from datetime import datetime

project = "seth-finetune"
base_model_name = "llama2-7b"
run_name = base_model_name + "-" + project
output_dir = "./" + run_name

tokenizer.pad_token = tokenizer.eos_token

trainer = transformers.Trainer(
    model=model,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_val_dataset,
    args=transformers.TrainingArguments(
        output_dir=output_dir,
        warmup_steps=1,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=1,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={'use_reentrant':False},
        max_steps=500,
        learning_rate=2.5e-5, # Want a small lr for finetuning
        bf16=True,
        optim="paged_adamw_8bit",
        logging_steps=25,
        logging_dir="./logs",        # Directory for storing logs
        save_strategy="steps",       # Save the model checkpoint every logging step
        save_steps=25,                # Save checkpoints every 50 steps
        evaluation_strategy="steps", # Evaluate the model every logging step
        eval_steps=25,               # Evaluate and save checkpoints every 50 steps
        do_eval=True,                # Perform evaluation at the end of training
        report_to="wandb",           # Comment this out if you don't want to use weights & baises
        run_name=f"{run_name}-{datetime.now().strftime('%Y-%m-%d-%H-%M')}"          # Name of the W&B run (optional)
    ),
    data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
)

model.config.use_cache = False  # silence the warnings. Please re-enable for inference!
trainer.train()

print ("##### START INFERENCE #####")


import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

base_model_id = "meta-llama/Llama-2-7b-hf"

base_model = AutoModelForCausalLM.from_pretrained(
    base_model_id,  # Llama 2 7B, same as before
    quantization_config=bnb_config,  # Same quantization config as before
    device_map="cuda:0",
    trust_remote_code=True,
)

tokenizer = AutoTokenizer.from_pretrained(base_model_id, add_bos_token=True, trust_remote_code=True)

from peft import PeftModel

ft_model = PeftModel.from_pretrained(base_model, "llama2-7b-seth-finetune/checkpoint-500")

print("\n##########\n")
eval_prompt = "What does Seth Godin say about getting unstuck? \n"
model_input = tokenizer(eval_prompt, return_tensors="pt").to("cuda:0")

ft_model.eval()
with torch.no_grad():
    print(tokenizer.decode(ft_model.generate(**model_input, max_new_tokens=300)[0], skip_special_tokens=True))

print("\n##########\n")
# Compare the results from previous input. These results would be much closer to the expected outcome.
# It is likely that we need further tuning to improve the performance for this to become a ship-able product.
