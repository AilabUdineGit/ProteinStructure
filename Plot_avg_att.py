import csv
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr


#Linear
with open('Linear_avg_abs_att.csv', 'r') as read_obj:

    csv_reader = csv.reader(read_obj)

    list_of_csv = list(csv_reader)

list_numbers = list (map(float, list_of_csv[0]))
print(len(list_numbers))
avg_att_linear = np.mean(np.reshape(list_numbers, (-1,20)), axis=1)

def visualize_importances(feature_names, importances, title="Average Feature Importances", plot=True,
                          axis_title="Features", log=False):
    print(title)
    x_pos = [x for x in feature_names]
    if plot:
        # plt.figure(figsize=(6, 4))
        #plt.plot(x_pos, importances)
        plt.bar(feature_names, importances, align='center')
        plt.xticks(x_pos, feature_names, wrap=True)
        plt.xlabel(axis_title)
        if log:
            plt.yscale('log')
        plt.title(title)
        plt.tight_layout()
        plt.show()


visualize_importances([-8,-7,-6,-5,-4,-3,-2,-1,0,1,2,3,4,5,6,7,8], avg_att_linear, title='Linear Model Average Feature Importances', log = False)

#LSTM
with open('LSTM_avg_abs_att.csv', 'r') as read_obj:

    csv_reader = csv.reader(read_obj)

    list_of_csv = list(csv_reader)


list_numbers_1 = list(map(float, list_of_csv[0]))
print(len(list_numbers_1))
avg_att_lstm= np.mean(np.reshape(list_numbers_1, (-1,20)), axis=1)
visualize_importances([-8,-7,-6,-5,-4,-3,-2,-1,0,1,2,3,4,5,6,7,8], avg_att_lstm, title='LSTM Model Average Feature Importances', log = False)

#Bert

with open('Bert_avg_abs_all_classes.csv', 'r') as read_obj:

    csv_reader = csv.reader(read_obj)

    list_of_csv = list(csv_reader)


list_numbers = list (map(float, list_of_csv[0]))
print(len(list_numbers))
avg_att_bert = list_numbers
visualize_importances([-8,-7,-6,-5,-4,-3,-2,-1,0,1,2,3,4,5,6,7,8], list_numbers, title = "Bert Model Average Feature Importances",log = False)

#

def get_sign_color(w0, pos="steelblue", neg="steelblue"):
    return [neg if x < 0 else pos for x in w0]


LABELS = ["-"+str(x) for x in range(8,0,-1)]+[str(0)]+[str(x) for x in range(1,9,1)]
fig, axs = plt.subplots(1, 3, figsize=(12, 3), sharey=True)
for i, ax in enumerate(axs):
    ax.yaxis.tick_right()
    ax.set_xticks(range(len(avg_att_bert)))
    ax.set_xticklabels(LABELS)
    # ax.tick_params(axis='x', rotation=90)
    if i == 2:
        ax.yaxis.set_tick_params(labelright=True, labelleft=False)
    else:
        for label in ax.get_yticklabels():
            label.set_visible(False)
axs[0].bar(range(17), avg_att_linear, color=get_sign_color(avg_att_linear), width=.9)
axs[1].bar(range(17 ), avg_att_lstm, color=get_sign_color(avg_att_lstm), width=.9)
axs[2].bar(range(17 ), avg_att_bert, color=get_sign_color(avg_att_bert), width=.9)
axs[0].set_title(f"Linear Model")
axs[1].set_title(f"LSTM Model")
axs[2].set_title(f"Bert Model")

plt.tight_layout()
fig.subplots_adjust(hspace=0.1, wspace=0)
#plt.savefig(f"{model_type}.weights.all.pdf", dpi=300)
#plt.title("Average Features Importances")
plt.show()

#linear Regression
import pandas as pd

weights = pd.read_csv("/mnt/HDD/VE_Saida/ProteinStructure-master/Linear_Regressor/coef3.out", sep=" ", header=None)
weights = weights.drop(columns=[340])
w0,w1,w2 = weights.values[0], weights.values[1], weights.values[2]
avg_att_reg = np.mean(np.array([abs(w0), abs(w1), abs(w2)]), axis=0)
avg_att_reg = np.mean(np.reshape(avg_att_reg, (-1,20)), axis=1)

print(len(avg_att_reg))


#mutual_coff
def flatten(A):
    rt = []
    for i in A:
        if isinstance(i,list): rt.extend(flatten(i))
        else: rt.append(i)
    return rt


with open('/mnt/HDD/VE_Saida/ProteinStructure-master/Mutual_Information/mutual_information.txt', 'r') as read_obj:

    csv_reader = csv.reader(read_obj)

    list_of_csv = list(csv_reader)
#flat_list = [float(item) for sublist in list_numbers for item in sublist]
flat_list = flatten(list_of_csv)
mututal_info = []
for i in flat_list:
    r = i[2:]
    print(i)
    y = float(r)
    mututal_info.append(y)

#calculate prearsonr
corr_,_= pearsonr(avg_att_linear,avg_att_lstm)
corr_1,_ =pearsonr(avg_att_linear,avg_att_bert)

corr_3,_=pearsonr(avg_att_lstm,avg_att_bert)
corr_4,_=pearsonr(avg_att_lstm,mututal_info)


corr_5,_ =pearsonr(avg_att_linear,avg_att_reg)
corr_6,_ = pearsonr(avg_att_lstm,avg_att_reg)
corr_7,_ = pearsonr(avg_att_bert,avg_att_reg)

corr_8,_ =pearsonr(avg_att_linear,mututal_info)
corr_9,_ = pearsonr(avg_att_lstm,mututal_info)
corr_10,_ = pearsonr(avg_att_bert,mututal_info)
corr_11,_ = pearsonr(avg_att_reg,mututal_info)

M = np.array([[1, corr_,corr_1,corr_5, corr_8],[corr_, 1,corr_4, corr_6, corr_9],[corr_1,corr_3,1,corr_7,corr_10],[corr_5, corr_6, corr_7,1,corr_11], [corr_8,corr_9,corr_10,corr_11,1]])
print(M[0,0])
heatmap = plt.imshow(M, cmap="PuBu")
plt.xlabel("Pesrson coeff")
plt.ylabel("Pesrson coef")
plt.xticks([0, 1, 2, 3, 4], ["Linear Model", "Bi-LSTM ", "Bert", "Linear Reg", "Mutual Information"])
plt.yticks([0, 1, 2 ,3,4], ["Linear Model", "Bi-LSTM", "Bert", "Linear Reg", "Mutual Information"])
plt.title("Pearson correlation coefficient between three model Feature Importances")
for x in range(5):
    for y in range(5):
        print(x)
        print(y)
        perc = round( M[x,y], 4)
        print(perc)
        plt.text(
            x, y,
            f"{round(M[y, x], 4)}",
            #f"{perc*100}%\n({round(M[y, x],4)})",
            ha="center",
            va="center",
            c="w" if M[y, x] > 50 else "k"
        )
model_type = "Pearson_coeff"
plt.colorbar(heatmap)
plt.savefig(f"{model_type}.confmat_test.pdf", dpi=300)
plt.show()

# the graphs without mutual information


M = np.array([[1, corr_,corr_1,corr_5],[corr_, 1,corr_4, corr_6],[corr_1,corr_3,1,corr_7],[corr_5, corr_6, corr_7,1]])
print(M[0,0])
heatmap = plt.imshow(M, cmap="PuBu")
plt.xlabel("Pesrson coeff")
plt.ylabel("Pesrson coef")
plt.xticks([0, 1, 2, 3], ["Linear Model", "Bi-LSTM ", "Bert", "Linear Reg"])
plt.yticks([0, 1, 2 ,3], ["Linear Model", "Bi-LSTM", "Bert", "Linear Reg"])
plt.title("Pearson correlation coefficient between three model Feature Importances")
for x in range(4):
    for y in range(4):
        print(x)
        print(y)
        perc = round( M[x,y], 4)
        print(perc)
        plt.text(
            x, y,
            f"{round(M[y, x], 4)}",
            #f"{perc*100}%\n({round(M[y, x],4)})",
            ha="center",
            va="center",
            c="w" if M[y, x] > 50 else "k"
        )
model_type = "Pearson_coeff"
plt.colorbar(heatmap)
plt.savefig(f"{model_type}.confmat_test.pdf", dpi=300)
plt.show()
