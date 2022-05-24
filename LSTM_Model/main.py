import torch
from torch.autograd import Variable
import numpy as np
from torch.utils.data import Dataset, ConcatDataset
import torch.nn as nn
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import LinearRegression
from torch.optim import SGD, Adam
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm

MODEL_NAME = "CORRECT_LSTM"

class CustomDataset(Dataset):
    def __init__(self, path1, path2):

        with open(path1, 'r') as f:
            self.data_info = f.readlines()
        with open(path2, 'r') as f:
            self.label_info = f.readlines()
    def __getitem__(self, index):
        single_data = self.data_info[index].rstrip('\n')
        line = single_data.split()  # to deal with blank
        if line:  # lines (ie skip them)
            line = [int(i) for i in line]
        single_data = line

        single_label = self.label_info[index].rstrip('\n')
        line = single_label.split()  # to deal with blank
        if line:  # lines (ie skip them)
            line = [int(i) for i in line]
        single_label = line

        sample = [single_data, single_label]#[torch.FloatTensor(single_data), torch.FloatTensor(single_label)]

        return sample

    def __len__(self):
        return len(self.data_info)


training_set = CustomDataset('one__hot_Ntrain_data.txt', 'Ntrain_targets.txt')
print("Number of the whole training dataset:", len(training_set))
test_set = CustomDataset('one__hot_Ntest_data.txt', 'Ntest_targets.txt')
print("Number of the test dataset:", len(test_set))


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
CUDA = torch.cuda.is_available()
print(f"working on GPU: {CUDA}")

import os

def run_gradient_descent(model, data_train, data_val, batch_size=64, learning_rate=0.01, weight_decay=0, num_epochs=20):
    # criterion = nn.CrossEntropyLoss()
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    iters, losses = [], []
    iters_sub, train_acc, val_acc = [], [] ,[]
    print(batch_size)
    train_loader = torch.utils.data.DataLoader(
        data_train,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda d: ([x[0] for x in d], [x[1] for x in d]),
        num_workers=os.cpu_count()//2
    )

    # training
    n = 0 # the number of iterations
    for epoch in tqdm(range(num_epochs), desc="epoch"):
        correct = 0
        total = 0
        for xs, ts in tqdm(train_loader, desc="train"):
            xs = torch.FloatTensor(xs).to(device)
            ts = torch.FloatTensor(ts).to(device)
            # if len(ts) != batch_size:
            #     print("ops")
            #     continue
            model.train()
            model = model.to(device)
            zs = model(xs)
            zs = zs.to(device)
            
            loss = criterion(zs, ts)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            iters.append(n)
            loss.detach().cpu()
            losses.append(float(loss)/len(ts)) # compute *average* loss
            pred = zs.max(1, keepdim=True)[1] # get the index of the max logit
            target = ts.max(1, keepdim=True)[1]
            correct += pred.eq(target).sum().item()
            total += int(ts.shape[0])
            acc = correct / total

            if (n % (len(train_loader)//2) == 0) and n>0:
                test_acc = get_accuracy(model, data_val)
                iters_sub.append(n)
                train_acc.append(acc)
                val_acc.append(test_acc)
                print("Epoch", epoch, "train_acc", acc)
                print("Epoch", epoch, "test_acc", test_acc)
             # increment the iteration number
            n += 1
        torch.save(model.state_dict(), f"{MODEL_NAME}/checkpoint_epoch{epoch}.pt")


    # plotting
    plt.title("Training Curve (batch_size={}, lr={})".format(batch_size, learning_rate))
    plt.plot(iters, losses, label="Train")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.savefig(f"{MODEL_NAME}/training_loss.png")
    # plt.show()
    plt.title("Training Curve (batch_size={}, lr={})".format(batch_size, learning_rate))
    plt.plot(iters_sub, train_acc, label="Train")
    plt.plot(iters_sub, val_acc, label="Test")
    plt.xlabel("Iterations")
    plt.ylabel("Accuracy")
    plt.legend(loc='best')
    plt.savefig(f"{MODEL_NAME}/training_acc.png")
    # plt.show()
    return model


def get_accuracy(model, data):
    loader = torch.utils.data.DataLoader(
        data,
        batch_size=64,
        collate_fn=lambda d: ([x[0] for x in d], [x[1] for x in d]),
        num_workers=0#os.cpu_count()//2
    )
    correct, total = 0, 0
    model.eval()
    with torch.no_grad():
        for xs, ts in tqdm(loader, desc="eval"):
            xs = torch.FloatTensor(xs).to(device)
            ts = torch.FloatTensor(ts).to(device)
            
            zs = model(xs)
            zs.detach().cpu()
            pred = zs.max(1, keepdim=True)[1]  # get the index of the max logit
            target = ts.max(1, keepdim=True)[1]
            correct += pred.eq(target).sum().item()
            total += int(ts.shape[0])
    return correct / total


class PPS(nn.Module):
    def __init__(self):
        super(PPS, self).__init__()
        self.n_layers = 1
        self.embedding_size = 20 #340
        self.n_hidden = 64
        self.num_labels = 3

        self.lstm1 = nn.LSTM(
            input_size=self.embedding_size,
            hidden_size=self.n_hidden,
            num_layers=self.n_layers,
            bidirectional=True,
            batch_first=True, dropout=0.2)

        self.fc1 = nn.Linear(self.n_hidden * 2, 32)
        self.fc2 = nn.Linear(32, self.num_labels)
        self.sof = nn.LogSoftmax(dim=-1)
        self.relu = nn.ReLU()

    def forward(self, x):
        
        x = x.view((-1,17,20))

        # x = x.squeeze(-1).unsqueeze(1)
        _, (out, _) = self.lstm1(x)
        
        out = torch.cat((out[0], out[1]), dim=1)

        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.sof(out)
        # out = out.squeeze()
        return out

    
path = f"{MODEL_NAME}/checkpoint"

if not os.path.exists(MODEL_NAME):
    os.makedirs(MODEL_NAME)

model = PPS()
run_gradient_descent(
    model,
    training_set,
    test_set,
    batch_size=128,
    learning_rate=1e-1,
    num_epochs=50
)

train_acc = get_accuracy(model, training_set)
test_acc = get_accuracy(model, test_set)
print("\n\n", train_acc, test_acc)

torch.save(model.state_dict(), path)
