import pandas as pd
import torch
from tqdm.notebook import tqdm
import numpy as np
from datasets import load_dataset
from torch.utils.data import Dataset
from trl import (
    AutoModelForCausalLMWithValueHead,
)
from transformers import AutoTokenizer
from peft import LoraConfig
from tqdm import tqdm
from transformers import pipeline

HF_TOKEN = "ENTER TOKEN"

ppo_tokenizer = AutoTokenizer.from_pretrained(
    "vicgalle/gpt2-open-instruct-v1",
    use_auth_token=HF_TOKEN,
    max_length=512,
    padding=True,
    truncation=True,
)

eval_generation_kwargs = {
    "min_length": -1,
    "top_k": 0.0,
    "top_p": 1.0,
    "do_sample": True,
    "pad_token_id": ppo_tokenizer.eos_token_id,
    'max_length': 400,
    "num_beams": 1,
}
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
sent_kwargs = {"return_all_scores": True, "function_to_apply": "none", "max_length":400}

model = AutoModelForCausalLMWithValueHead.from_pretrained(
        "gpt2PPO_500",
        device_map="auto",
        use_auth_token=HF_TOKEN,
        peft_config=lora_config,
    )

ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(
    "vicgalle/gpt2-open-instruct-v1",
    device_map="auto",
    use_auth_token=HF_TOKEN,
    peft_config=lora_config,
)
ppo_tokenizer.pad_token = ppo_tokenizer.eos_token


reward_model = pipeline("text-classification", model="gpt2_reward_model_500")

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

bs = 128
device = "cuda" if torch.cuda.is_available() else "cpu"

def eval():
    torch.cuda.empty_cache()    
    ppo_dataset = load_dataset("csv",data_files= "datasets/reward_dataset_500/reward_dataset_500.csv")
    ppo_dataset = build_dataset_PPO(ppo_tokenizer, ppo_dataset)

    #### Model Inspection 
    game_data = dict()
    ppo_dataset.set_format("pandas")
    df_batch = ppo_dataset['train'][:].sample(bs)
    game_data["query"] = df_batch["Question"].tolist()
    query_tensors = df_batch["input_ids"].tolist()

    response_tensors_ref, response_tensors = [], []

    #### get response from gpt2 and gpt2_ref
    for i in tqdm(range(bs)):
    #     gen_len = 512
        output = ref_model.generate(
            torch.tensor(query_tensors[i]).unsqueeze(dim=0).to(device), **eval_generation_kwargs
        ).squeeze()
        response_tensors_ref.append(output)
        output = model.generate(
            torch.tensor(query_tensors[i]).unsqueeze(dim=0).to(device), **eval_generation_kwargs
        ).squeeze()
        response_tensors.append(output)
    #### decode responses
    game_data["response (before)"] = [ppo_tokenizer.decode(response_tensors_ref[i]) for i in range(bs)]
    game_data["response (after)"] = [ppo_tokenizer.decode(response_tensors[i]) for i in range(bs)]

    #### sentiment analysis of query/response pairs before/after
    texts = [q + r for q, r in zip(game_data["query"], game_data["response (before)"])]
    game_data["rewards (before)"] = [output[1]["score"] for output in reward_model(texts, **sent_kwargs)]

    texts = [q + r for q, r in zip(game_data["query"], game_data["response (after)"])]
    game_data["rewards (after)"] = [output[1]["score"] for output in reward_model(texts, **sent_kwargs)]

    # store results in a dataframe
    df_results = pd.DataFrame(game_data)
    df_results.to_csv('EvalOutput500_1.csv')
    print("mean:")
    print(df_results[["rewards (before)", "rewards (after)"]].mean())
    print()
    print("median:")
    print(df_results[["rewards (before)", "rewards (after)"]].median())


if __name__=="__main__":
    eval()
