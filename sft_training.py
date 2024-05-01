import pandas as pd 
import numpy as np 
import torch 
import torch.nn as nn 
from tqdm.notebook import tqdm 
import numpy as np 
from datasets import Dataset
from trl import (
    ModelConfig,
    RichProgressCallback,
    SFTTrainer,
    get_peft_config,
    get_quantization_config,
    get_kbit_device_map,
    RewardTrainer, 
    RewardConfig,
    PPOConfig
    AutoModelForCausalLMWithValueHead,
    PPOTrainer
)
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import LoraConfig, TaskType
from transformers import AutoModelForSequenceClassification
import gc 
from tqdm import tqdm


df_sft = pd.read_csv('ENTER DATASET LOCATION')
sft_dataset = Dataset.from_pandas(df_sft[['abstract']])

peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

sft_model = AutoModelForCausalLM.from_pretrained(
    "gpt2",
    device_map="auto",
)
sft_tokenizer = AutoTokenizer.from_pretrained('gpt2')

sft_tokenizer.add_special_tokens({'pad_token':'[PAD]'})
sft_model.resize_token_embeddings(len(sft_tokenizer))

training_args = TrainingArguments(
    output_dir="./outputs",  # Adjust output directory as needed
    overwrite_output_dir=True,  # Set to False if resuming training
    per_device_train_batch_size=4,  # Adjust batch size based on your GPU memory
    save_steps=500,
    num_train_epochs=3,
    fp16 = True# Adjust training epochs
)

sft_trainer = SFTTrainer(
    sft_model,
    train_dataset=sft_dataset,
    dataset_text_field="abstract",
    peft_config=peft_config,
    args = training_args,
    tokenizer = sft_tokenizer
)
sft_trainer.train()

sft_model.save_pretrained(output_dir)


