import gradio as gr
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel

from trl import (
    AutoModelForCausalLMWithValueHead,
)
from transformers import AutoTokenizer
from peft import LoraConfig

# from tqdm import tqdm
# from transformers import pipeline

HF_TOKEN = "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"
device = "cuda"
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
    "max_length": 400,
    "num_beams": 1,
}
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
sent_kwargs = {
    "return_all_scores": True,
    "function_to_apply": "none",
    "max_length": 400,
}
ppo_tokenizer.pad_token = ppo_tokenizer.eos_token

model_vanila = AutoModelForCausalLMWithValueHead.from_pretrained(
    "vicgalle/gpt2-open-instruct-v1",
    device_map="auto",
    use_auth_token=HF_TOKEN,
    peft_config=lora_config,
)

model500 = AutoModelForCausalLMWithValueHead.from_pretrained(
    "monjoychoudhury29/gpt2PPO",
    device_map="auto",
    use_auth_token=HF_TOKEN,
    peft_config=lora_config,
)

model200 = AutoModelForCausalLMWithValueHead.from_pretrained(
    "monjoychoudhury29/gpt2PPO200",
    device_map="auto",
    use_auth_token=HF_TOKEN,
    peft_config=lora_config,
)


def convert_to_prompt(example):
    prompt_template = """
        Below is an instruction that describes a task. Write a response that appropriately completes the request.

        ### Instruction:
        Pretend you are a medical expert and answer to the following question - {query}

        ### Response: 

        """
    example = prompt_template.format(query=example)
    return example


# Function to generate response using GPT-2 model
def generate_response_gpt2(prompt):
    prompt = convert_to_prompt(prompt)
    input_ids = ppo_tokenizer.encode(prompt, return_tensors="pt").to(device)
    output_vanila = model_vanila.generate(input_ids, **eval_generation_kwargs)
    output_200 = model200.generate(input_ids, **eval_generation_kwargs)
    output_500 = model500.generate(input_ids, **eval_generation_kwargs)

    response_vanila = ppo_tokenizer.decode(output_vanila[0], skip_special_tokens=True)
    response_200 = ppo_tokenizer.decode(output_200[0], skip_special_tokens=True)
    response_500 = ppo_tokenizer.decode(output_500[0], skip_special_tokens=True)

    final_vanila = response_vanila.split("Response:")[1]
    final_str_200 = response_200.split("Response:")[1]
    final_str_500 = response_500.split("Response:")[1]

    return final_vanila, final_str_200, final_str_500


# Create Gradio interface
gr.Interface(
    fn=generate_response_gpt2,
    title="RLHF Experiment",
    description="Enter a prompt and get responses from different models.",
    inputs=[gr.Textbox(label="Input Prompt")],
    outputs=[
        gr.Textbox(label="Standard Fine-Tuning (SFT) GPT-2 Response"),
        gr.Textbox(label="GPT-2 200-Version Response"),
        gr.Textbox(label="GPT-2 400-Version Response"),
    ],
).launch(share=True)
