import pandas as pd
import random
import numpy as np
from transformers import GPT2TokenizerFast, GPT2LMHeadModel, GPT2Config
import torch
from torch.utils.data import Dataset, DataLoader, random_split, RandomSampler, SequentialSampler
from transformers import AdamW, get_linear_schedule_with_warmup
import random
import time
import datetime
import os

model_name = 'gpt2'        # set model name 
batch_size = 4              # Set batch size
seed_val = 42               # This step is optional but it wiil enable reproducible runs

"""## Training parameters"""
epochs = 4			# same as batchsize 
learning_rate = 5e-4
warmup_steps = 1e2
epsilon = 1e-6
sample_every = 100          # this produces sample output every 100 steps

"""1. Get the Data"""
# read from csv file 
turk_ids = pd.read_csv('train_turk_ids.csv') 
greek_ids = pd.read_csv('train_greek_ids.csv')
portuguese_ids = pd.read_csv('train_portuguese_ids.csv')

"""# 2. Format Data """

def format_data_tr(df):
    # Drop columns with no information
    df = df.drop(['Unnamed: 0', 'Issuing Authority'], axis=1)
    
    # its important to have all dtypes in sting format 
    df['Identity No'] = df['Identity No'].apply(str)
  
    # remove '.' becasuse produce conflicts with ',' 
    df["Date of Birth"] =  df["Date of Birth"].str.replace('.', '/', regex=True)
    df["Date of Expire"] =  df["Date of Expire"].str.replace('.', '/', regex=True)
    
    # convert it dictionary for easier 
    dictQA = df.to_dict(orient = 'records')

    # Proper Format of Data for Q.A. task
    # load all dataset to list object 
    data = []
    
    for index in dictQA:
        for k, v in index.items():
            line = ', '.join(index.values()) +", " +k +":\n" +v +"<|endoftext|>"
            data.append(line)

    return data


def format_data_gr(df):
    # Drop columns with no information
    df = df.drop(['Unnamed: 0', 'Ύψος', 'Issuing Authority'], axis=1)
    
    # convert it dictionary for easier 
    dictQA = df.to_dict(orient = 'records')

    # Proper Format of Data for Q.A. task
    # load all dataset to list object 
    data = []
    
    for index in dictQA:
        for k, v in index.items():
            line = ', '.join(index.values()) +", " +k +":\n" +v +"<|endoftext|>"
            data.append(line)

    return data


def format_data_pr(df):
    # Drop columns with no information
    df = df.drop(['Unnamed: 0'], axis=1)

    # split column Parents to get beter results  
    df_tmp = pd.DataFrame(df['Parents'].str.split("|", 1).tolist(), columns=["Mother's Parents", "Father's Parents"])    
    df = pd.concat([df, df_tmp], axis=1).drop(["Parents"], axis=1)
    

    # remove whitespaces from those columns
    df["Mother's Parents"] = df["Mother's Parents"].str.strip()
    df["Father's Parents"] = df["Father's Parents"].str.strip()

    # set all dtypes to object   
    df = df.applymap(str)

     # replace space with '/' in dates
    df["Date of Birth"] =  df["Date of Birth"].str.replace(' ', '/', regex=True)
    df["Date of Expire"] =  df["Date of Expire"].str.replace(' ', '/', regex=True)

    # convert it dictionary for easier 
    dictQA = df.to_dict(orient = 'records')

    # Proper Format of Data for Q.A. task
    # load all dataset to list object 
    data = []
    
    for index in dictQA:
        for k, v in index.items():
            line = ', '.join(index.values()) +", " +k +":\n" +v +"<|endoftext|>"
            data.append(line)

    return data
    
    
    
    
gr = format_data_gr(greek_ids)
tr = format_data_tr(turk_ids)
pr = format_data_pr(portuguese_ids)

# concatenate all data to single input
data = gr + tr + pr


"""3. Tokenization"""

# load tokenizer from our pre-trained model
# add extra special tokens
tokenizer = GPT2TokenizerFast.from_pretrained(model_name,
                                          eos_token='<|endoftext|>', 
                                          pad_token='<|pad|>',
                                          add_prefix_space=True )

# set pad token to eos token because it auto sets while generating 
tokenizer.pad_token = tokenizer.eos_token 

# Re-define amx length
max_length = max([len(tokenizer.encode(row)) for row in data])
max_length = max_length+1
print(f'The longest text is {max_length} tokens long.')

class IDDataset(Dataset):

  def __init__(self, df, tokenizer, gpt2_type="gpt2", max_length=max_length):

    self.tokenizer = tokenizer
    self.input_ids = []
    self.attn_masks = []

    for line in df:
      encodings_dict = tokenizer(line, truncation=True, max_length=max_length, padding="max_length")

      self.input_ids.append(torch.tensor(encodings_dict['input_ids']))
      self.attn_masks.append(torch.tensor(encodings_dict['attention_mask']))
    
  def __len__(self):
    return len(self.input_ids)

  def __getitem__(self, idx):
    return self.input_ids[idx], self.attn_masks[idx]


# Create Dataset 
dataset = IDDataset(data, tokenizer, max_length=max_length)

"""5. Split my Data into train and validatons """

train_size = int(0.9 * len(dataset))
val_size = len(dataset)-train_size

train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
#f'There are {train_size} samples for training, and {val_size} samples for validation testing'
print('{:>5,} training samples'.format(train_size))
print('{:>5,} validation samples'.format(val_size))


"""6. DataLoader """

train_dataloader = DataLoader(
    train_dataset,
    sampler = RandomSampler(train_dataset), # Sampling for training is Random
    batch_size = batch_size
)

validation_dataloader = DataLoader(
    val_dataset,
    sampler = SequentialSampler(val_dataset),
    batch_size = batch_size
)

"""# Finetune GPT2 Language Model"""

"""# Set GPU or CPU"""
# cleaning the occupied cuda memory
torch.cuda.empty_cache()
# Tell pytorch to run this model on the GPU 
# if not available then run on cpu 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Loading the model configuration and setting it to the GPT2 standard settings.
configuration = GPT2Config.from_pretrained(model_name, output_hiden_states=False)


# Create the instance of the model 
model = GPT2LMHeadModel.from_pretrained(model_name, config=configuration)
# set the token size embedding length
model.resize_token_embeddings(len(tokenizer))

model.to(device)

random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)

# AdamW is a class from the huggingface library, it is the optimizer we will be using, and we will only be instantiating it with the default parameters. 
optimizer = AdamW(model.parameters(),
                  lr = learning_rate,
                  eps = epsilon
                )

total_steps = len(train_dataloader) * epochs

"""
We can set a variable learning rate which will help scan larger areas of the 
problem space at higher LR earlier, then fine tune to find the exact model minima 
at lower LR later in training.
"""
scheduler = get_linear_schedule_with_warmup(optimizer, 
                                            num_warmup_steps = warmup_steps, 
                                            num_training_steps = total_steps)


def format_time(elapsed):
    return str(datetime.timedelta(seconds=int(round((elapsed)))))


"""## model train"""

total_t0 = time.time()
training_stats = []

model = model.to(device)

for epoch_i in range(0, epochs):

    # ========================================
    #               Training
    # ========================================

    print("")
    print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
    print('Training...')

    t0 = time.time()

    total_train_loss = 0

    model.train()

    for step, batch in enumerate(train_dataloader):

        b_input_ids = batch[0].to(device)
        b_labels = batch[0].to(device)
        b_masks = batch[1].to(device)

        model.zero_grad()        

        outputs = model(  b_input_ids,
                          labels=b_labels, 
                          attention_mask = b_masks,
                          token_type_ids=None
                        )

        loss = outputs[0]  

        batch_loss = loss.item()
        total_train_loss += batch_loss

        # Get sample every x batches.
        if step % sample_every == 0 and not step == 0:

            elapsed = format_time(time.time() - t0)
            print('  Batch {:>5,}  of  {:>5,}. Loss: {:>5,}.   Elapsed: {:}.'.format(step, len(train_dataloader), batch_loss, elapsed))

            model.eval()

            sample_outputs = model.generate(
                                    bos_token_id=random.randint(1,30000),
                                    do_sample=True,   
                                    top_k=50, 
                                    max_length = 200,
                                    top_p=0.95, 
                                    num_return_sequences=1
                                )
            for i, sample_output in enumerate(sample_outputs):
                  print("{}: {}".format(i, tokenizer.decode(sample_output, skip_special_tokens=True)))
            
            model.train()

        loss.backward()

        optimizer.step()

        scheduler.step()

    # Calculate the average loss over all of the batches.
    avg_train_loss = total_train_loss / len(train_dataloader)       
    
    # Measure how long this epoch took.
    training_time = format_time(time.time() - t0)

    print("")
    print("  Average training loss: {0:.2f}".format(avg_train_loss))
    print("  Training epoch took: {:}".format(training_time))
        
    # ========================================
    #               Validation
    # ========================================

    print("")
    print("Running Validation...")

    t0 = time.time()

    model.eval()

    total_eval_loss = 0
    nb_eval_steps = 0

    # Evaluate data for one epoch
    for batch in validation_dataloader:
        
        b_input_ids = batch[0].to(device)
        b_labels = batch[0].to(device)
        b_masks = batch[1].to(device)
        
        with torch.no_grad():        

            outputs  = model(b_input_ids, 
#                            token_type_ids=None, 
                             attention_mask = b_masks,
                            labels=b_labels)
          
            loss = outputs[0]  
            
        batch_loss = loss.item()
        total_eval_loss += batch_loss        

    avg_val_loss = total_eval_loss / len(validation_dataloader)
    
    validation_time = format_time(time.time() - t0)    

    print("  Validation Loss: {0:.2f}".format(avg_val_loss))
    print("  Validation took: {:}".format(validation_time))

    # Record all statistics from this epoch.
    training_stats.append(
        {
            'epoch': epoch_i + 1,
            'Training Loss': avg_train_loss,
            'Valid. Loss': avg_val_loss,
            'Training Time': training_time,
            'Validation Time': validation_time
        }
    )


print("")
print("Training complete!")
print("Total training took {:} (h:mm:ss)".format(format_time(time.time()-total_t0)))

"""# Results - Score, Loss table
"""
# Display floats with two decimal places.
pd.set_option('precision', 2)

# Create a DataFrame from our training statistics.
df_stats = pd.DataFrame(data=training_stats)

# Use the 'epoch' as the row index.
df_stats = df_stats.set_index('epoch')

print('\n')
print(df_stats)
print('\n')

# Model Info
# Get all of the model's parameters as a list of tuples.
params = list(model.named_parameters())

"""# Save model"""
output_dir = "/root/model_save/"

# Create output directory if needed
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

print("Saving model to %s" % output_dir)

# Save a trained model, configuration and tokenizer using `save_pretrained()`.
# They can then be reloaded using `from_pretrained()`
model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
model_to_save.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)

# Good practice: save your training arguments together with the trained model
# torch.save(args, os.path.join(output_dir, 'training_args.bin'))
#print(dataset[2])
print("ok!")
