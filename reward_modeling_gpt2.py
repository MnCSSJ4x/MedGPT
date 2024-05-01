import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from tqdm.notebook import tqdm
import numpy as np
from datasets import load_dataset
from datasets import Dataset 
from torch.utils.data import Dataset
from trl import (
    RewardTrainer,
)
from transformers import TrainingArguments
from peft import LoraConfig, TaskType
from transformers import (
    GPT2Config,
    GPT2ForSequenceClassification,
    GPT2Tokenizer,
)
from datasets import DatasetDict

# Configurations
device = "cuda" if torch.cuda.is_available() else "cpu"
HF_TOKEN = "YOUR HF TOKEN"
configuration = GPT2Config()
peft_config = LoraConfig(
    task_type=TaskType.SEQ_CLS,
    inference_mode=False,
    r=8,
    lora_alpha=32,
    lora_dropout=0.1,
)
training_args = TrainingArguments(
    output_dir="./outputs_500",  # Adjust output directory as needed
    overwrite_output_dir=True,  # Set to False if resuming training
    per_device_train_batch_size=4,  # Adjust batch size based on your GPU memory
    per_device_eval_batch_size=2,
    save_steps=1060,
    num_train_epochs=10,
    fp16=True,  # Adjust training epochs
    logging_dir="./runs/reward_modeling_500",
    logging_steps=530,
    load_best_model_at_end=True,
    evaluation_strategy="steps",
    eval_steps = 1060
)

reward_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
reward_tokenizer.pad_token = reward_tokenizer.eos_token
# instantiate the model
reward_model = (
    GPT2ForSequenceClassification(configuration).from_pretrained("gpt2").to(device)
)
# set the pad token of the model's configuration
reward_model.config.pad_token_id = reward_model.config.eos_token_id

def preprocess_function(examples):
    new_examples = {
        "input_ids_chosen": [],
        "attention_mask_chosen": [],
        "input_ids_rejected": [],
        "attention_mask_rejected": [],
    }
    for chosen, rejected in zip(examples["Answer"], examples["generated"]):
        tokenized_chosen = reward_tokenizer(chosen, padding=True, truncation=True)
        tokenized_rejected = reward_tokenizer(rejected,padding=True, truncation=True)
        new_examples["input_ids_chosen"].append(tokenized_chosen["input_ids"])
        new_examples["attention_mask_chosen"].append(tokenized_chosen["attention_mask"])
        new_examples["input_ids_rejected"].append(tokenized_rejected["input_ids"])
        new_examples["attention_mask_rejected"].append(
            tokenized_rejected["attention_mask"]
        )
    return new_examples


def train():
    torch.cuda.empty_cache()
    # Convert data to Hugging Face dataset format
    reward_dataset = load_dataset('csv',data_files= {"train": "../datasets/reward_dataset_500/reward_train_set.csv", "test":"../datasets/reward_dataset_500/reward_eval_set.csv"})
    reward_dataset = reward_dataset.map(preprocess_function, batched=True)
    reward_trainer = RewardTrainer(
        model=reward_model,
        args=training_args,
        tokenizer=reward_tokenizer,
        train_dataset=reward_dataset['train'],
        eval_dataset = reward_dataset['test'],
        peft_config=peft_config,
        
    )
    reward_trainer.train()
    reward_trainer.save_model("gpt2_reward_model_500")
    
if __name__ == "__main__":
    train() 
