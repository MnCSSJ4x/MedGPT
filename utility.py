import pandas as pd 

from tqdm import tqdm
df_500 = pd.read_csv('/home/sunnidhya/rlhf-experiments/rlhf/rlhf-experiments/rlhf-experiments/datasets/reward_dataset_500/reward_dataset_500.csv')
df_generated = pd.read_csv('/home/sunnidhya/rlhf-experiments/rlhf/rlhf-experiments/rlhf-experiments/src/gemini_eval_500.csv')

def generate_text(prompt):
  prompt_template = """
    Below is an instruction that describes a task. Write a response that appropriately completes the request.

    ### Instruction:
    Pretend you are a medical expert and answer to the following question - {query}
    
    ### Response: 
    """
  return prompt_template.format(query=prompt) 


df_500["Generated Question"] = df_500["Question"].apply(generate_text)

# Initialize an empty list to store the matched values
matched_values = []

# Iterate over each row in df1
for i, row1 in tqdm(df_generated.iterrows()):
    query = row1['query']
    matched = False
    # Iterate over each row in df_500
    for j, row2 in df_500.iterrows():
        question = row2['Question']
        # Check if question partially matches query
        if question in query or query in question:
            matched_values.append(row2['Answer'])
            matched = True
            break
    # If no match found, append None
    if not matched:
        matched_values.append(None)

df_generated['Answer'] = matched_values
df_generated.to_csv('full_out.csv')