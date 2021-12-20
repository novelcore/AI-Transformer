from transformers import GPT2LMHeadModel, GPT2Config, GPT2TokenizerFast
from transformers import set_seed
import os
import gradio as gr
import torch
import pandas as pd 
from collections import Counter

set_seed(42)

model_name = 'gpt2'
max_length = 242

# if not available then run on cpu 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


#Load a trained model and vocabulary that you have fine-tuned
configuration = GPT2Config.from_json_file("/root/model_save/config.json")
model = GPT2LMHeadModel.from_pretrained("/root/model_save", config=configuration)
tokenizer = GPT2TokenizerFast.from_pretrained("/root/model_save")

model = model.to(device)

"""1. Get the Data"""
# read from csv file 
portuguese_ids = pd.read_csv('test_portuguese_ids.csv')

"""# 2. Format Data """

def format_data_porto(df):
    # Drop columns with no information
    df = df.drop(['Unnamed: 0'], axis=1)

    # split column Parents to get beter results  
    df_tmp = pd.DataFrame(df['Parents'].str.split("|", 1).tolist(), columns=["Mother Parents", "Father Parents"])    
    df = pd.concat([df, df_tmp], axis=1).drop(["Parents"], axis=1)
    

    # remove whitespaces from those columns
    df['Mother Parents'] = df['Mother Parents'].str.strip()
    df['Father Parents'] = df['Father Parents'].str.strip()

    # set all dtypes to object   
    df = df.applymap(str)

    # replace space with '/' in dates
    df["Date of Birth"] =  df["Date of Birth"].str.replace(' ', '/', regex=True)
    df["Date of Expire"] =  df["Date of Expire"].str.replace(' ', '/', regex=True)
   
    # convert it dictionary for easier 
    dictQA = df.to_dict(orient = 'records')

    # Proper Format of Data for Q.A. task
    # load all dataset to list object 
    X = []
    y = []
    
    for index in dictQA:
        for q, a in index.items():
            line = ' '.join(index.values()) +", " +q +":\n"

            y.append(a)
            X.append(line)

    return X, y
    
    
X_test, y_test = format_data_porto(portuguese_ids)


print("X test length: ", len(X_test))
print("y test length: ", len(y_test))

""" 3. get max length """
# Get the max_length of tokens 
max_length = max([len(tokenizer.encode(row)) for row in X_test])
max_length = max_length +1

print(f'The longest text is {max_length} tokens long.')


""" 4. predictions """
preds = []

for line in X_test:
  model.eval()
  input_ids = tokenizer(line, return_tensors="pt").input_ids
  input_ids = input_ids.to(device)
  input_len = len(input_ids[0])
  generated_outputs = model.generate(input_ids, 
                                      do_sample=True,   
                                      top_k=50, 
                                      max_length = max_length,
                                      top_p=0.95, 
                                      num_return_sequences=1
                                    )
  
  for i, seq in enumerate(generated_outputs):
    pred = tokenizer.decode(seq[input_len:], skip_special_tokens=True)
  
  preds.append(pred)


""" 4. compute_f1_score """
# function that computes f1 score of model
def compute_f1(c_answer, pred_answer):
    gold_toks = c_answer
    pred_toks = pred_answer
    
    common = Counter(gold_toks) & Counter(pred_toks)
    
    total = 0
    for i in common.values():
        total += i

    num_same = total

    if len(gold_toks) == 0 or len(pred_toks) == 0:
      # If either is no-answer, then F1 is 1 if they agree, 0 otherwise
      return int(gold_toks == pred_toks)


    
    if num_same == 0:
      return 0

    precision = 1.0 * num_same / len(pred_toks)
    recall = 1.0 * num_same / len(gold_toks)
    f1_score = (2*precision *recall) / (precision + recall)

    return f1_score

""" 6. Avg f1 score"""
#calculate the avg
i = 0
sum = 0

for correct_as, prediction_as in zip(y_test, preds):
   sum = sum + compute_f1(correct_as, prediction_as)
   i = i + 1

avg = sum/i

print("Avg f1 score is ", avg)
print('ok!')
