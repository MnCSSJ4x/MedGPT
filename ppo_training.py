import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from tqdm.notebook import tqdm
import numpy as np
from datasets import load_dataset
from torch.utils.data import Dataset
from trl import (
    PPOConfig,
    AutoModelForCausalLMWithValueHead,
    PPOTrainer,
)
from transformers import AutoTokenizer
from peft import LoraConfig
from transformers import (
    BitsAndBytesConfig,
)
from tqdm import tqdm
from trl.core import LengthSampler
from transformers import pipeline


# Configs
device = "cuda" if torch.cuda.is_available() else "cpu"
access_token = "ENTER YOUR HF TOKEN"  # Replace with your access token
config = PPOConfig(
    model_name="vicgalle/gpt2-open-instruct-v1",
    learning_rate=1.41e-5,
    batch_size=32,
    mini_batch_size=16,
    is_peft_model=True,
    log_with='tensorboard',
    project_kwargs={"logging_dir": "./runs/ppo_trainer_500"}
)

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
quantization_config = BitsAndBytesConfig(load_in_8bit=True, signed=True)


model = AutoModelForCausalLMWithValueHead.from_pretrained(
        config.model_name,
        device_map="auto",
        use_auth_token=access_token,
        peft_config=lora_config,
        #     qunatization_config = quantization_config
    )

ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(
    config.model_name,
    device_map="auto",
    use_auth_token=access_token,
    peft_config=lora_config,
)

ppo_tokenizer = AutoTokenizer.from_pretrained(
    config.model_name,
    use_auth_token=access_token,
    max_length=512,
    padding=True,
    truncation=True,
)

ppo_tokenizer.pad_token = ppo_tokenizer.eos_token
reward_model = pipeline("text-classification", model="gpt2_reward_model_500")


generation_kwargs = {
    "min_length": -1,
    "top_k": 0.0,
    "top_p": 1.0,
    "do_sample": True,
    "pad_token_id": ppo_tokenizer.eos_token_id,
    'max_length': 400,
    "num_beams": 1,
    "batch_size": 16, 
#     "max_time":1, 
}


def build_dataset_PPO(tokenizer, ppo_dataset) -> Dataset:
    train_ds = ppo_dataset
    def tokenize(example):
        example["input_ids"] = tokenizer.encode(example["Question"])
        example["query"] = tokenizer.decode(example["input_ids"])
        return example
    def convert_to_prompt(example):
        prompt_template = """
        Below is an instruction that describes a task. Write a response that appropriately completes the request.

        ### Instruction:
        Pretend you are a medical expert and answer to the following question - {query}

        ### Response: 

        """
        example["Question"] = prompt_template.format(query=example["Question"])
        return example 
    
    
    train_ds = train_ds.map(convert_to_prompt, batched=False)
    train_ds = train_ds.map(tokenize, batched=False)
    train_ds.set_format(type="torch")
    return train_ds


def train():
    torch.cuda.empty_cache()

    
    ppo_dataset = load_dataset("csv", data_files= "/datasets/reward_dataset_500/reward_dataset_500.csv")
    ppo_dataset = build_dataset_PPO(ppo_tokenizer, ppo_dataset)

    ppo_trainer = PPOTrainer(
        model=model,
        ref_model=ref_model,
        config=config,
        dataset=ppo_dataset['train'],
        tokenizer=ppo_tokenizer,
        data_collator=lambda x: dict((key, [d[key] for d in x]) for key in x[0]),
    )
    generation_kwargs["pad_token_id"] = ppo_tokenizer.eos_token_id
    epochs = 1
    for epoch in tqdm(range(epochs), "epoch: "):
        for batch in tqdm(ppo_trainer.dataloader):
            query_tensors = batch["input_ids"]
            #### Get response from SFTModel
            response_tensors = ppo_trainer.generate(
                query_tensors, **generation_kwargs, return_prompt=False
            )
            batch["response"] = [
                ppo_tokenizer.decode(r.squeeze()) for r in response_tensors
            ]

            #### Compute reward score
            texts = [q + r for q, r in zip(batch["query"], batch["response"])]
            pipe_outputs = reward_model(texts)
            rewards = [torch.tensor(output["score"]) for output in pipe_outputs]

            #### Run PPO step
            stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
            ppo_trainer.log_stats(stats, batch, rewards)
            # break
        # break 
    #### Save model
    # model.push_to_hub(repo_id="AK232003/gpt2_ppo_model_200", overwrite=True, access_token=HF_TOKEN)
    model.save_pretrained("gpt2PPO_500")
    # ppo_trainer.save_model("gpt2PPO_500")
    

if __name__ == "__main__":
    train()
