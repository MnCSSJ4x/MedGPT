import pandas as pd 
df = pd.read_csv('reward_dataset_500/reward_dataset_500.csv')

train_size = int(0.8 * len(df))  
train_data = df[:train_size]
eval_data = df[train_size:]

train_data.to_csv('reward_dataset_500/reward_train_set.csv')
eval_data.to_csv('reward_dataset_500/reward_eval_set.csv')