import pandas as pd
from BARTScore.SUM.bart_score import BARTScorer
import numpy as np 

device = "cuda" 
bart_model_name = 'facebook/bart-large-cnn'
def compute_bart_score(df):
    reference = df["Answer"].tolist()
    prediction_1 = df["response (before)"].tolist()
    prediction_2 = df["response (after)"].tolist()

    bartscore = BARTScorer(device, max_length=512, checkpoint='facebook/bart-large-cnn')
    bartscore.load()
    
    results_gpt2 = bartscore.score(prediction_1, reference, batch_size=2)
    results_rlhf = bartscore.score(prediction_2,reference,batch_size=2)
    print("GPT2 Without RLHF")
    print(results_gpt2)
    print("Mean ", np.array(results_gpt2).mean())
    print("GPT2 With RLHF")
    print(results_rlhf)
    print("Mean ", np.array(results_rlhf).mean())

if __name__ == "__main__":
    df = pd.read_csv("full_out.csv")
    compute_bart_score(df)