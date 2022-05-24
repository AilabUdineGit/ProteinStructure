import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
from transformers import BertTokenizerFast, BertConfig, BertForTokenClassification
from torch import cuda
from tqdm import tqdm
from transformers import BertConfig
from transformers import BertModel
import json
import matplotlib.pyplot as plt
import os
import pickle


FINETUNED_MODEL_PATH = "pretraining_code/Protein/model_output_files/maxseq_64_bs_256x8_ep_50_nhidden_4"

OUTPUT_MODEL_PATH = "finetuned_model/maxseq_17_tokencls_lr_1e-06"

config = BertConfig.from_pretrained(FINETUNED_MODEL_PATH)
config.num_labels = 3

# EMBEDDING_SIZE = 768
MAX_LEN = 17#config.max_position_embeddings
TRAIN_BATCH_SIZE = 128
VALID_BATCH_SIZE = 128
EPOCHS = 20
LEARNING_RATE = 1e-06
MAX_GRAD_NORM = 10
ADD_SPECIAL_TOKENS = False


def load_json(path):
    with open(path, "r") as f:
        data = json.load(f)
    return data


fast_tokenizer_args = load_json(
    "pretraining_code/Protein/tokenizer_output_files/BertWordPieceTokenizer/fast_tokenizer_args.json")
tokenizer = BertTokenizerFast.from_pretrained(
    "pretraining_code/Protein/tokenizer_output_files/BertWordPieceTokenizer",
    **fast_tokenizer_args)

train_data = pd.read_csv("new_17resid_train_data.txt", encoding='unicode_escape', names=['Seq'])
train_target = pd.read_csv("Ntrain_3states_target.txt", encoding='unicode_escape', names=['target'])

test_data = pd.read_csv("new_17resid_test_data.txt", encoding='unicode_escape', names=['Seq'])
test_target = pd.read_csv("Ntest_3states_target.txt", encoding='unicode_escape', names=['target'])

df = pd.concat([train_data, train_target], axis=1)
df_ = pd.concat([test_data, test_target], axis=1)
device = 'cuda' if cuda.is_available() else 'cpu'

label_set = df.target.unique().tolist()  # set("".join(df.target.tolist()))


# labels_to_ids = {k: v for v, k in enumerate(label)}
# ids_to_labels = {v: k for v, k in enumerate(label)}


def batchify(l, batch_size):
    i = 0
    while i < len(l):
        i += batch_size
        yield l[i - batch_size:i]


def make_dataset(df):
    train_x = []
    train_y = []

    for _, row in tqdm(df.iterrows(), total=len(df)):

        window_size = len(list(row.Seq.strip()))

        sent = " ".join(list(row.Seq.strip()))
        sent = sent.replace("B", "[PAD]")
        train_x.append(sent)

        word_labels = row.target
        labels = [word_labels]  # [labels_to_ids[label] for label in word_labels]

        if ADD_SPECIAL_TOKENS == True:
            labels = [-100] + labels

        labels = [-100] * (window_size // 2) + labels
        labels = labels + [-100] * (MAX_LEN - len(labels))
        labels = labels[:MAX_LEN]
        train_y.append(labels)

    input_ids_train = []
    attention_mask_train = []

    for batch in tqdm(batchify(train_x, 500), total=len(train_x) // 500):
        tok = tokenizer(
            batch,
            return_attention_mask=True,
            padding='max_length',
            add_special_tokens=ADD_SPECIAL_TOKENS,
            truncation=True,
            max_length=MAX_LEN
        )
        input_ids_train += tok["input_ids"]
        attention_mask_train += tok["attention_mask"]

    input_ids_train = torch.LongTensor(input_ids_train)
    attention_mask_train = torch.LongTensor(attention_mask_train)
    train_y = torch.LongTensor(train_y)
    training_set = TensorDataset(input_ids_train, attention_mask_train, train_y)
    return training_set


def read_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def save_pickle(obj, path):
    with open(path, "wb") as f:
        pickle.dump(obj, f)


if os.path.exists("cached_training_set.pickle"):
    print("\t LOADING training_set FROM CACHE")
    training_set = read_pickle("cached_training_set.pickle")
else:
    training_set = make_dataset(df)
    save_pickle(training_set, "cached_training_set.pickle")

if os.path.exists("cached_testing_set.pickle"):
    print("\t LOADING testing_set FROM CACHE")
    testing_set = read_pickle("cached_testing_set.pickle")
else:
    testing_set = make_dataset(df_)
    save_pickle(testing_set, "cached_testing_set.pickle")

train_params = {'batch_size': TRAIN_BATCH_SIZE,
                'shuffle': True,
                'num_workers': os.cpu_count()//2
                }
test_params = {'batch_size': VALID_BATCH_SIZE,
               'shuffle': False,
               'num_workers': os.cpu_count()//2
               }
training_loader = DataLoader(training_set, **train_params)
testing_loader = DataLoader(testing_set, **test_params)
# model = BertForTokenClassification.from_pretrained("/mnt/HDD/VE_Saida/Transformers_3states/dilbert-main/Protein/model_output_fil   es/Protein/checkpoint-93443", config=config)
model = BertForTokenClassification.from_pretrained(FINETUNED_MODEL_PATH, config=config)

model.to(device)

optimizer = torch.optim.Adam(params=model.parameters(), lr=LEARNING_RATE)


# Evaluating the model
def valid(model, testing_loader):
    # put model in evaluation mode
    model.eval()
    eval_loss, eval_accuracy = 0, 0
    nb_eval_examples, nb_eval_steps = 0, 0
    eval_preds, eval_labels = [], []

    with torch.no_grad():
        for idx, batch in enumerate(testing_loader):
            ids, mask, labels = batch
            ids = ids.to(device)
            mask = mask.to(device)
            labels = labels.to(device)

            out = model(input_ids=ids, attention_mask=mask, labels=labels)
            loss = out["loss"]
            eval_logits = out["logits"]
            eval_loss += loss.item()

            nb_eval_steps += 1
            nb_eval_examples += labels.size(0)

            # if idx % 100 == 0:
            #     loss_step = eval_loss / nb_eval_steps
            #     print(f"Validation loss per 100 evaluation steps: {loss_step}")

            # compute evaluation accuracy
            flattened_targets = labels.view(-1)  # shape (batch_size * seq_len,)
            active_logits = eval_logits.view(-1, model.num_labels)  # shape (batch_size * seq_len, num_labels)
            flattened_predictions = torch.argmax(active_logits, axis=1)  # shape (batch_size * seq_len,)

            # only compute accuracy at active labels
            active_accuracy = labels.view(-1) != -100  # shape (batch_size, seq_len)

            labels = torch.masked_select(flattened_targets, active_accuracy)
            predictions = torch.masked_select(flattened_predictions, active_accuracy)

            eval_labels.extend(labels)
            eval_preds.extend(predictions)

            tmp_eval_accuracy = accuracy_score(labels.cpu().numpy(), predictions.cpu().numpy())
            eval_accuracy += tmp_eval_accuracy

    labels = [id.item() for id in eval_labels]
    predictions = [id.item() for id in eval_preds]

    eval_loss = eval_loss / nb_eval_steps
    eval_accuracy = eval_accuracy / nb_eval_steps

    print(f"Validation Loss: {eval_loss}")
    print(f"Validation Accuracy: {eval_accuracy}")

    return labels, predictions, eval_loss, eval_accuracy


# Defining the training function on the 80% of the dataset for tuning the bert model
def train(epoch):
    tr_loss, tr_accuracy = 0, 0
    nb_tr_examples, nb_tr_steps = 0, 0
    tr_preds, tr_labels = [], []
    # put model in training mode
    model.train()
    temp = 0
    for idx, batch in tqdm(enumerate(training_loader), total=5_000):#len(training_loader)):
        
        if idx == 5_000:
            break

        # ids = batch['input_ids'].to(device, dtype=torch.long)
        # mask = batch['attention_mask'].to(device, dtype=torch.long)
        # labels = batch['labels'].to(device, dtype=torch.long)
        temp += len(batch)

        ids, mask, labels = batch
        ids = ids.to(device)
        mask = mask.to(device)
        labels = labels.to(device)

        out = model(input_ids=ids, attention_mask=mask, labels=labels)
        loss = out["loss"]
        tr_logits = out["logits"]
        tr_loss += loss.item()

        nb_tr_steps += 1
        nb_tr_examples += labels.size(0)
        
        
        # if idx%5000==0:
        #     model.save_pretrained(f"{OUTPUT_MODEL_PATH}/checkpoint_epoch{epoch}_step{idx}")

        # if idx % 100 == 0:
        #     loss_step = tr_loss / nb_tr_steps
        #     print(f"Training loss per 100 training steps: {loss_step}")

        # compute training accuracy
        flattened_targets = labels.view(-1)  # shape (batch_size * seq_len,)
        active_logits = tr_logits.view(-1, model.num_labels)  # shape (batch_size * seq_len, num_labels)
        flattened_predictions = torch.argmax(active_logits, axis=1)  # shape (batch_size * seq_len,)

        # only compute accuracy at active labels
        active_accuracy = labels.view(-1) != -100  # shape (batch_size, seq_len)
        # active_labels = torch.where(active_accuracy, labels.view(-1), torch.tensor(-100).type_as(labels))

        labels = torch.masked_select(flattened_targets, active_accuracy)
        predictions = torch.masked_select(flattened_predictions, active_accuracy)

        tr_labels.extend(labels)
        tr_preds.extend(predictions)

        tmp_tr_accuracy = accuracy_score(labels.cpu().numpy(), predictions.cpu().numpy())
        tr_accuracy += tmp_tr_accuracy

        # gradient clipping
        torch.nn.utils.clip_grad_norm_(
            parameters=model.parameters(), max_norm=MAX_GRAD_NORM
        )

        # backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # if temp == 5000:

        #   break

    epoch_loss = tr_loss / nb_tr_steps
    tr_accuracy = tr_accuracy / nb_tr_steps

    print(f"Training loss epoch: {epoch_loss}")
    print(f"Training accuracy epoch: {tr_accuracy}")
    labels, predictions, eval_loss, eval_accuracy = valid(model, testing_loader)
    return epoch_loss, tr_accuracy, eval_loss, eval_accuracy


eval_losses = []
eval_accs = []
tr_losses = []
tr_accs = []
for epoch in range(EPOCHS):
    print(f"Training epoch: {epoch + 1}")
    epoch_loss, tr_accuracy, eval_loss, eval_accuracy = train(epoch)
    model.save_pretrained(f"{OUTPUT_MODEL_PATH}/checkpoint_epoch{epoch}")
    eval_losses.append(eval_loss)
    eval_accs.append(eval_accuracy)
    tr_losses.append(epoch_loss)
    tr_accs.append(tr_accuracy)

d1 = {'Train_Acc': tr_accs, 'Test_Acc': eval_accs}
d2 = {'Train_loss': tr_losses, 'Test_loss': eval_losses}
df = pd.DataFrame(d1)
df_ = pd.DataFrame(d2)

plot1 = plt.figure(1)
plt.plot(range(EPOCHS), tr_accs)
plt.plot(range(EPOCHS), eval_accs)
plt.legend(["Train_accuracy", "Test_accuracy"])
plt.savefig(f'{OUTPUT_MODEL_PATH}/F2.png')

plot2 = plt.figure(2)
plt.plot(range(EPOCHS), tr_losses)
plt.plot(range(EPOCHS), eval_losses)
plt.legend(["Train_loss", "Test_loss"])
plt.savefig(f'{OUTPUT_MODEL_PATH}/F1.png')

# plt.show()

df.to_csv(f"{OUTPUT_MODEL_PATH}/accuracy.csv", sep='\t')
df_.to_csv(f"{OUTPUT_MODEL_PATH}/losses.csv", sep='\t')