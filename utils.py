import torch.utils
from torch.autograd import Variable
import numpy as np
from torch.utils.data import Dataset, ConcatDataset
import torch.nn as nn
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import LinearRegression
from torch.optim import SGD, Adam
import torch.optim as optim
import matplotlib.pyplot as plt
from captum.attr import DeepLift
import os
from tqdm import tqdm
import torch


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


class MyRegression(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MyRegression, self).__init__()
        # One layer
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.linear(x)


class PPS(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(PPS, self).__init__()
        self.n_layers = 1
        self.embedding_size = 20
        self.window_dim = input_dim // 20
        self.n_hidden = 64
        self.num_labels = output_dim

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

        x = x.view((-1,self.window_dim,20))
        _, (out, _) = self.lstm1(x)
        out = torch.cat((out[0], out[1]), dim=1)

        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.sof(out)
        return out
    
NUM_TO_NAME =  {
    0: "Helix",
    1: "Coil",
    2: "Strand",
    3: "All",
}

def get_accuracy(model, data):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    loader = torch.utils.data.DataLoader(
        data,
        batch_size=500,
        collate_fn=lambda d: ([x[0] for x in d], [x[1] for x in d]),
        num_workers=0#os.cpu_count()//2
    )
    
    all_real = []
    all_pred = []
    
    correct, total = 0, 0
    model.eval()
    with torch.no_grad():
        for xs, ts in tqdm(loader, desc="eval"):
            xs = torch.FloatTensor(xs).to(device)
            ts = torch.FloatTensor(ts).to(device)
            
            zs = model(xs)
            zs = zs.to(device)
            pred = zs.max(1, keepdim=True)[1]  # get the index of the max logit
            target = ts.max(1, keepdim=True)[1]
            
            all_real += target.detach().cpu().tolist()
            all_pred += pred.detach().cpu().tolist()
            
            correct += pred.eq(target).sum().item()
            total += int(ts.shape[0])
    
    all_real = [x[0] for x in all_real]
    all_pred = [x[0] for x in all_pred]
    return correct / total, all_real, all_pred


def batchify(l, batch_size):
    i = 0
    while i < len(l):
        i += batch_size
        yield l[i-batch_size:i]

def get_attributions(dl, X_test, y_test, target, absolute=False):
    
    attr_list = []
    
    if type(target)==list:
        
        for batch in batchify(list(zip(X_test, target)), 1_000):
            
            x = torch.cat([b[0] for b in batch])
            y = torch.cat([b[1] for b in batch])
            
            dl_attr_test = dl.attribute(x, target=y)
            # dl_attr_test_sum = dl_attr_test.detach().cpu().numpy().sum(0)
            # dl_attr_test_norm_sum_0 = dl_attr_test_sum / np.linalg.norm(dl_attr_test_sum, ord=1)
            dl_attr_test = np.absolute(dl_attr_test)
            dl_attr_test_norm_sum_0=dl_attr_test.detach().cpu().numpy().mean(0)
            attr_list.append(dl_attr_test_norm_sum_0)
        
    else:
    
        for batch in batchify(X_test, 1_000):

            dl_attr_test = dl.attribute(batch, target=target)
            # dl_attr_test_sum = dl_attr_test.detach().cpu().numpy().sum(0)
            # dl_attr_test_norm_sum_0 = dl_attr_test_sum / np.linalg.norm(dl_attr_test_sum, ord=1)
            l_attr_test = torch.abs(dl_attr_test)
            dl_attr_test_norm_sum_0=dl_attr_test.detach().cpu().numpy().mean(0)
            attr_list.append(dl_attr_test_norm_sum_0)
        
    dl_attr_test_norm_sum_0 = np.mean(attr_list, axis=0)
    dl_attr_test_norm_sum_17_0 = np.mean(np.reshape(dl_attr_test_norm_sum_0, (-1,20)), axis=1)
    
    if absolute:
        dl_attr_test_norm_sum_0 = np.abs(dl_attr_test_norm_sum_0)
        dl_attr_test_norm_sum_17_0 = np.abs(dl_attr_test_norm_sum_17_0)
        
        
    abs_dl_attr_test_norm_sum_0 = np.abs(dl_attr_test_norm_sum_0)
    abs_dl_attr_test_norm_sum_17_0 = np.mean(np.reshape(abs_dl_attr_test_norm_sum_0, (-1,20)), axis=1)
    
    return dl_attr_test_norm_sum_0.tolist(), dl_attr_test_norm_sum_17_0.tolist(), abs_dl_attr_test_norm_sum_0.tolist(), abs_dl_attr_test_norm_sum_17_0.tolist()



def get_sign_color(w0, pos="steelblue", neg="steelblue"):#(w0, pos="steelblue", neg="tomato"):
    return [neg if x<0 else pos for x in w0]




def PLOT(dl, X_test, y_test, absolute=False):
    
    fig, axs = plt.subplots(4,3, figsize=(10, 8), sharey=True, sharex=True)
    
    for target in range(3):
        for c in range(4):

            X = c
            Y = target
            if X == 2:
                X = 1
            elif X == 1:
                X = 2
            if Y == 2:
                Y = 1
            elif Y == 1:
                Y = 2
            ax = axs[X,Y]

            ax.yaxis.tick_right()
            if Y==2:
                ax.yaxis.set_tick_params(labelright=True, labelleft=False)
                pass
            else:

                for label in ax.get_yticklabels():
                    label.set_visible(False)

            if c==0:
                ax.set_title(f"{NUM_TO_NAME[target]} Neuron")
            if target==0:
                ax.set_ylabel(f"{NUM_TO_NAME[c]} Samples")

            if c==3:
                attr0, attr17_0, abs_attr0, abs_attr17_0 = get_attributions(dl, X_test, y_test, target)
                num_samp0 = len(X_test)
                x_axis_data = list(range(len(attr17_0)))
            else:
                attr0, attr17_0, abs_attr0, abs_attr17_0 = get_attributions(dl, X_test[y_test[:,c]==1], y_test, target)
                num_samp0 = len(X_test[y_test[:,c]==1])
                x_axis_data = list(range(len(attr17_0)))

            y_values = [0]+abs_attr17_0+[0] if absolute else [0]+attr17_0+[0]
            ax.bar(
                [-1]+x_axis_data+[len(attr17_0)],
                y_values,
                color= get_sign_color([0]+attr17_0+[0]),
                lw=1,
                alpha=1 if (target==c or c==3) else .5,
                width=.9,
            )

            ax.set_xlim(-1,len(attr17_0))
            ax.set_xticks(x_axis_data)
            
    plt.tight_layout()
    fig.subplots_adjust(hspace=0.1, wspace=0)
    
    
    
    
def PLOT_DIAG(dl, X_test, y_test, absolute=False, avg=True):
    
    LABELS = ["-"+str(x) for x in range(8,0,-1)]+[str(0)]+[str(x) for x in range(1,9,1)]

    
    fig, axs = plt.subplots(1,3, figsize=(12, 3), sharey=True, sharex=True)
    
    all_attributions = []
    
    for target in range(3):
        #for c in range(4):

            c=0
            X = 0
            Y = target
            ax = axs[Y]

            ax.yaxis.tick_right()
            if Y==2:
                ax.yaxis.set_tick_params(labelright=True, labelleft=False)
                pass
            else:

                for label in ax.get_yticklabels():
                    label.set_visible(False)

            ax.set_title(f"{NUM_TO_NAME[target]} Neuron")

            if c==3:
                attr0, attr17_0, abs_attr0, abs_attr17_0 = get_attributions(dl, X_test, y_test, target)
                num_samp0 = len(X_test)
            else:
                attr0, attr17_0, abs_attr0, abs_attr17_0 = get_attributions(dl, X_test[y_test[:,target]==1], y_test, target)
                num_samp0 = len(X_test[y_test[:,c]==1])

            all_attributions.append(abs_attr17_0)
                
            if avg:
                x_axis_data = list(range(len(attr17_0)))
                y_values = [0]+abs_attr17_0+[0] if absolute else [0]+attr17_0+[0]
                ax.bar(
                    [-1]+x_axis_data+[len(attr17_0)],
                    y_values,
                    color= get_sign_color([0]+attr17_0+[0]),
                    lw=1,
                    alpha=1,
                    width=.9,
                )

                ax.set_xlim(-1,len(attr17_0))
                ax.set_xticks(x_axis_data)
                ax.set_xticklabels(LABELS)
            else:
                x_axis_data = list(range(len(attr0)))
                y_values = [0]+abs_attr0+[0] if absolute else [0]+attr0+[0]
                ax.bar(
                    [-1]+x_axis_data+[len(attr0)],
                    y_values,
                    color= get_sign_color([0]+attr0+[0]),
                    lw=1,
                    alpha=1,
                    width=1,
                )

                # ax.set_xlim(-1,len(attr0))
                ax.set_xticks([x for x in x_axis_data if x%20==0])
                ax.set_xticklabels(LABELS)
                # ax.tick_params(axis='x', rotation=90)

    plt.tight_layout()
    fig.subplots_adjust(hspace=0.1, wspace=0)
    return all_attributions
    
    
def PLOT_DIAG2(dl, X_test, y_test, absolute=False, avg=True):
    
    LABELS = ["-"+str(x) for x in range(8,0,-1)]+[str(0)]+[str(x) for x in range(1,9,1)]

    
    fig, axs = plt.subplots(1,3, figsize=(12, 3), sharey=True, sharex=True)
    
    for target in range(3):
        #for c in range(4):

            c=0
            X = 0
            Y = target
            ax = axs[Y]

            ax.yaxis.tick_right()
            if Y==2:
                ax.yaxis.set_tick_params(labelright=True, labelleft=False)
                pass
            else:

                for label in ax.get_yticklabels():
                    label.set_visible(False)

            ax.set_title(f"{NUM_TO_NAME[target]} Neuron")

            if c==3:
                attr0, attr17_0, abs_attr0, abs_attr17_0 = get_attributions(dl, X_test, y_test, target)
                num_samp0 = len(X_test)
            else:
                x_samples = X_test[y_test[:,target]==1]
                attr0, attr17_0, abs_attr0, abs_attr17_0 = get_attributions(dl, x_samples, y_test, [target]*len(x_samples))
                num_samp0 = len(X_test[y_test[:,c]==1])
                

            if avg:
                x_axis_data = list(range(len(attr17_0)))
                y_values = [0]+abs_attr17_0+[0] if absolute else [0]+attr17_0+[0]
                ax.bar(
                    [-1]+x_axis_data+[len(attr17_0)],
                    y_values,
                    color= get_sign_color([0]+attr17_0+[0]),
                    lw=1,
                    alpha=1,
                    width=.9,
                )

                ax.set_xlim(-1,len(attr17_0))
                ax.set_xticks(x_axis_data)
                ax.set_xticklabels(LABELS)
            else:
                x_axis_data = list(range(len(attr0)))
                y_values = [0]+abs_attr0+[0] if absolute else [0]+attr0+[0]
                ax.bar(
                    [-1]+x_axis_data+[len(attr0)],
                    y_values,
                    color= get_sign_color([0]+attr0+[0]),
                    lw=1,
                    alpha=1,
                    width=1,
                )

                # ax.set_xlim(-1,len(attr0))
                ax.set_xticks([x for x in x_axis_data if x%20==0])
                ax.set_xticklabels(LABELS)
                # ax.tick_params(axis='x', rotation=90)

    plt.tight_layout()
    fig.subplots_adjust(hspace=0.1, wspace=0)