import torch
import pandas as pd
from tqdm import tqdm
import gc
from transformers import AutoTokenizer, AutoModelForCausalLM,BitsAndBytesConfig


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
hf_token = "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"
model_id = "vicgalle/gpt2-open-instruct-v1"

quantization_config = BitsAndBytesConfig(    
    load_in_8bit=True
    )

def get_completion(query_list, model, tokenizer) -> str:
    prompt_template = """
    Below is an instruction that describes a task. Write a response that appropriately completes the request.

    ### Instruction:
    Pretend you are a medical expert and answer to the following question - {query}
    
    ### Response: 

    """
    prompt_list = []
    for query in query_list:
        prompt = prompt_template.format(query=query)
        prompt_list.append(prompt)

    encodeds = tokenizer(prompt_list, return_tensors="pt", max_length=1024, padding=True, truncation=True)
    model_inputs = encodeds.to(device)
    generated_ids = model.generate(**model_inputs, max_length=250, do_sample=True, pad_token_id=tokenizer.eos_token_id, num_beams=1, num_return_sequences=1)
    decoded = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

    return decoded

tokenizer = AutoTokenizer.from_pretrained(model_id, token= hf_token, use_fast=True)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    token= hf_token,
    torch_dtype=torch.float16,
)

def generate_diagnosis_given_text(df):
        diagnosis_list = []
        progress_bar = tqdm(range(df.shape[0]//batch_size))
        query_list = []
        for i, row in df.iterrows():
            if ((i + 1) % batch_size == 0) or (i == df.shape[0] - 1):
                query_list.append(row['Question'])
                results = get_completion(query_list, model=model, tokenizer=tokenizer)
                query_list = []
                diagnosis_list.extend(result.split('Response')[1] for result in results)
                progress_bar.update()
                torch.cuda.empty_cache()
                gc.collect()
            else:
                query_list.append(row['Question'])  
        return diagnosis_list

batch_size = 128



if __name__ == '__main__':
    df = pd.read_csv('filtered_train.csv')
    diagnosis_list = generate_diagnosis_given_text(df)
    df['generated'] = diagnosis_list 
    df.to_csv('reward_dataset.csv')