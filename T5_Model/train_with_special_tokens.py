import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch import cuda
from tqdm import tqdm
from transformers import BertConfig
from transformers import BertModel
import json
import matplotlib.pyplot as plt
import os
import pickle

from transformers import AutoTokenizer, T5ForConditionalGeneration

PRETRAINED_MODEL_NAME = "t5-small"
EPOCHS = 9
TRAIN_BATCH_SIZE = 512//2
VALID_BATCH_SIZE = 512//2
SEQ_LEN=18

model = T5ForConditionalGeneration.from_pretrained(PRETRAINED_MODEL_NAME)
tokenizer = AutoTokenizer.from_pretrained(
    PRETRAINED_MODEL_NAME,
    padding_side="left"
)
tokenizer_r = AutoTokenizer.from_pretrained(
    PRETRAINED_MODEL_NAME,
    padding_side="right"
)

train_data = pd.read_csv("new_17resid_train_data.txt", encoding='unicode_escape', names=['Seq'])
train_target = pd.read_csv("Ntrain_3states_target.txt", encoding='unicode_escape', names=['target'])

test_data = pd.read_csv("new_17resid_test_data.txt", encoding='unicode_escape', names=['Seq'])
test_target = pd.read_csv("Ntest_3states_target.txt", encoding='unicode_escape', names=['target'])

df = pd.concat([train_data, train_target], axis=1)
df_ = pd.concat([test_data, test_target], axis=1)

df["Seq"] = df.Seq.apply(lambda val: " ".join([x.replace("B", tokenizer.pad_token) for x in val]))
df_["Seq"] = df_.Seq.apply(lambda val: " ".join([x.replace("B", tokenizer.pad_token) for x in val]))

device = 'cuda' if cuda.is_available() else 'cpu'

from torch.utils.data import TensorDataset, DataLoader

def batchify(l, batch_size):
    i = 0
    while i < len(l):
        i += batch_size
        yield l[i-batch_size:i]

def to_tensor_dataset(data, tokenizer, kind):
    
    input_ids = []
    labels = []
    
    outputs = tokenizer_r(["Helyx", "Coil", "Strand"], padding="longest")
    outputs = {i:v for i,v in enumerate(outputs.input_ids)}
    
    labels = data.target.apply(lambda x: outputs[x])#[outputs[row.target] for _,row in tqdm(data.iterrows(), total=len(data))]
    labels = torch.tensor(labels)
    
    print(labels.shape)
        
    batch_size = 200
    
    for sentences in tqdm(batchify(data.Seq, batch_size), total=len(data)//batch_size):
        
        tok_out = tokenizer(sentences.tolist(), add_special_tokens=True, padding="longest")
        tok_out = [
            [tokenizer.bos_id]+[x for x in seq if (x != 3) and (x != 0)]+[tokenizer.eos_token_id]
            for seq in tok_out.input_ids
        ]
        tok_out = [
            seq + [tokenizer.pad_token_id]*(SEQ_LEN-len(seq))
            for seq in tok_out
        ]
        
        input_ids += tok_out
    
    input_ids = torch.tensor(input_ids)
    
    print(input_ids.shape)
    
    return TensorDataset(input_ids, labels)

import pickle
import os

if not os.path.exists("train_dataset_special.pkl"):

    train_dataset = to_tensor_dataset(df, tokenizer, "TRAIN")
    test_dataset = to_tensor_dataset(df_, tokenizer, "TEST")
    # test_dataloader = DataLoader(test_dataset, batch_size=8)

    with open("train_dataset_special.pkl", "wb") as f:
        pickle.dump(train_dataset, f)
    with open("test_dataset_special.pkl", "wb") as f:
        pickle.dump(test_dataset, f)
        
with open("train_dataset_special.pkl", "rb") as f:
    train_dataset = pickle.load(f)
#with open("test_dataset.pkl", "rb") as f:
#    test_dataset = pickle.load(f)
    
import transformers

training_args = transformers.TrainingArguments(
    # ------------------------------------------------------- [epochs and batch size]
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=TRAIN_BATCH_SIZE*2,
    gradient_accumulation_steps=1,
    # ------------------------------------------------------- [hyperparams]
    warmup_steps=100, 
    weight_decay=0.01,
    # ------------------------------------------------------- [save and logging]
    output_dir=".", 
    overwrite_output_dir = True,
    do_eval = False,
    logging_strategy="epoch", # activate if interested
    save_strategy="epoch",
    save_total_limit = None,
    # -------------------------------------------------------
)
trainer = transformers.Trainer(
    model=model, 
    args=training_args, 
    train_dataset=train_dataset,
    data_collator = lambda data: {
        'input_ids': torch.stack([f[0] for f in data]), 
        # 'attention_mask': torch.stack([f[1] for f in data]), 
        'labels': torch.stack([f[1] for f in data]),
    }
)

trainer.train()

trainer.save_model("t5_special_9_epochs")



