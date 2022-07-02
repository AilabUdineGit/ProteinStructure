import csv
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr


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


#visualize_importances([-8,-7,-6,-5,-4,-3,-2,-1,0,1,2,3,4,5,6,7,8], avg_att_linear, title='Linear Model Average Feature Importances', log = False)

import pandas as pd
mi_h = pd.read_csv("mi_new_set_h.txt", sep=" ", header=None)
mi_l= pd.read_csv("mi_new_set_l.txt", sep=" ", header=None)
mi_e= pd.read_csv("mi_new_set_e.txt", sep=" ", header=None)

#visualize_importances([-8,-7,-6,-5,-4,-3,-2,-1,0,1,2,3,4,5,6,7,8], list(weights_h.iloc[:, 0]), title='Mutual information _Average Feature Importances', log = False)
#visualize_importances([-8,-7,-6,-5,-4,-3,-2,-1,0,1,2,3,4,5,6,7,8], list(weights_l.iloc[:, 0]), title='Mutual information _Average Feature Importances', log = False)
#visualize_importances([-8,-7,-6,-5,-4,-3,-2,-1,0,1,2,3,4,5,6,7,8], list(weights_e.iloc[:, 0]), title='Mutual information _Average Feature Importances', log = False)


def get_sign_color(w0, pos="steelblue", neg="steelblue"):
    return [neg if x < 0 else pos for x in w0]


LABELS = ["-"+str(x) for x in range(8,0,-1)]+[str(0)]+[str(x) for x in range(1,9,1)]

fig, axs = plt.subplots(1, 3, figsize=(12, 3), sharey=True)
for i, ax in enumerate(axs):
    ax.yaxis.tick_right()
    ax.set_xticks(range(len(list(mi_h.iloc[:, 0]))))
    ax.set_xticklabels(LABELS)
    # ax.tick_params(axis='x', rotation=90)
    if i == 2:
        ax.yaxis.set_tick_params(labelright=True, labelleft=False)
    else:
        for label in ax.get_yticklabels():
            label.set_visible(False)
axs[0].bar(range(17), list(mi_h.iloc[:, 0]), color=get_sign_color(list(mi_h.iloc[:, 0])), width=.9)
axs[1].bar(range(17 ), list(mi_l.iloc[:, 0]), color=get_sign_color(list(mi_l.iloc[:, 0])), width=.9)
axs[2].bar(range(17 ), list(mi_e.iloc[:, 0]), color=get_sign_color(list(mi_e.iloc[:, 0])), width=.9)
axs[0].set_title(f"Helix Neuron")
axs[1].set_title(f"Coil Neuron")
axs[2].set_title(f"Strand Neuron")

plt.tight_layout()
fig.subplots_adjust(hspace=0.1, wspace=0)
plt.show()

################################################################

#linear Regression coff

#Helix_Neuron
weights_h = pd.read_csv("coef3_new_set_h.txt", sep=" ", header=None)
weights_h = weights_h.drop(columns=[340])
print(len(list(weights_h)))
avg_att_reg = np.mean(np.array([abs(weights_h)]), axis=0)
avg_att_reg_h = np.mean(np.reshape(avg_att_reg, (-1,20)), axis=1)


#Coil_Neuron
weights_l = pd.read_csv("coef3_new_set_l.txt", sep=" ", header=None)
weights_l = weights_l.drop(columns=[340])
avg_att_reg = np.mean(np.array([abs(weights_l)]), axis=0)
avg_att_reg_l = np.mean(np.reshape(avg_att_reg, (-1,20)), axis=1)

#Strand_Neuron
weights_e = pd.read_csv("coef3_new_set_e.txt", sep=" ", header=None)
weights_e = weights_e.drop(columns=[340])
avg_att_reg = np.mean(np.array([abs(weights_e)]), axis=0)
avg_att_reg_e = np.mean(np.reshape(avg_att_reg, (-1,20)), axis=1)



fig, axs = plt.subplots(1, 3, figsize=(12, 3), sharey=True)
for i, ax in enumerate(axs):
    ax.yaxis.tick_right()
    ax.set_xticks(range(len(avg_att_reg_h)))
    ax.set_xticklabels(LABELS)
    # ax.tick_params(axis='x', rotation=90)
    if i == 2:
        ax.yaxis.set_tick_params(labelright=True, labelleft=False)
    else:
        for label in ax.get_yticklabels():
            label.set_visible(False)
axs[0].bar(range(17), avg_att_reg_h , color=get_sign_color(avg_att_reg_h ), width=.9)
axs[1].bar(range(17 ), avg_att_reg_l , color=get_sign_color(avg_att_reg_l ), width=.9)
axs[2].bar(range(17 ), avg_att_reg_e , color=get_sign_color(avg_att_reg_e ), width=.9)
axs[0].set_title(f"Helix Neuron")
axs[1].set_title(f"Coil Neuron")
axs[2].set_title(f"Strand Neuron")

plt.tight_layout()
fig.subplots_adjust(hspace=0.1, wspace=0)
plt.show()
