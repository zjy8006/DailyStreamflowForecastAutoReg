import os
root_path = os.path.dirname(os.path.abspath('__file__'))
# root_path = os.path.abspath(os.path.join(root_path, os.path.pardir))
# root_path = os.path.abspath(os.path.join(root_path, os.path.pardir))
print(10*'-'+' Root Path: {}'.format(root_path))

data_path = parent_path + '\\data\\'
graph_path = grandpa_path+'\\graph\\'

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import re
def sort_key(s):
    if s:
        try:
            c = re.findall('LSTM-IMF1-LR[\d+$]-HU[8]-EPS[500]-BS[512]-DR[0.0]-DC[0.0]-SEED[1].csv', s)[0]
        except:
            c = -1
        return int(c)
def strsort(alist):
    alist.sort(key=sort_key,reverse=True)
    return alist

MODEL_ID = 1
model_path = current_path+'\\lstm-models-history\\imf'+str(MODEL_ID)+'\\'
plt.figure(figsize=(8,4))
pred_files_list=[]
for files in os.listdir(model_path):
    if files.find('HU[16]-EPS[500]-BS[512]-DR[0.0]-DC[0.0]-SEED[1]-HISTORY')>0:
        pred_files_list.append(files)
        # print(files)
pred_files_list=strsort(pred_files_list)
# for IMF1
pred_files_list=[
    'LSTM-IMF1-LR[0.0001]-HU[8]-EPS[500]-BS[512]-DR[0.0]-DC[0.0]-SEED[1]-HISTORY-TRAIN-TEST.csv',
    'LSTM-IMF1-LR[0.0003]-HU[8]-EPS[500]-BS[512]-DR[0.0]-DC[0.0]-SEED[1]-HISTORY-TRAIN-TEST.csv',
    'LSTM-IMF1-LR[0.0007]-HU[8]-EPS[500]-BS[512]-DR[0.0]-DC[0.0]-SEED[1]-HISTORY-TRAIN-TEST.csv',
    'LSTM-IMF1-LR[0.001]-HU[8]-EPS[500]-BS[512]-DR[0.0]-DC[0.0]-SEED[1]-HISTORY-TRAIN-TEST.csv',
    'LSTM-IMF1-LR[0.003]-HU[8]-EPS[500]-BS[512]-DR[0.0]-DC[0.0]-SEED[1]-HISTORY-TRAIN-TEST.csv',
    'LSTM-IMF1-LR[0.007]-HU[8]-EPS[500]-BS[512]-DR[0.0]-DC[0.0]-SEED[1]-HISTORY-TRAIN-TEST.csv',
    'LSTM-IMF1-LR[0.01]-HU[8]-EPS[500]-BS[512]-DR[0.0]-DC[0.0]-SEED[1]-HISTORY-TRAIN-TEST.csv',
    'LSTM-IMF1-LR[0.03]-HU[8]-EPS[500]-BS[512]-DR[0.0]-DC[0.0]-SEED[1]-HISTORY-TRAIN-TEST.csv',
    'LSTM-IMF1-LR[0.07]-HU[8]-EPS[500]-BS[512]-DR[0.0]-DC[0.0]-SEED[1]-HISTORY-TRAIN-TEST.csv',
    'LSTM-IMF1-LR[0.1]-HU[8]-EPS[500]-BS[512]-DR[0.0]-DC[0.0]-SEED[1]-HISTORY-TRAIN-TEST.csv',
]
for i in range(0,len(pred_files_list)):
    print(pred_files_list[i])
    data = pd.read_csv(model_path+pred_files_list[i])
    plt.xticks(fontsize=10.5)
    plt.yticks(fontsize=10.5)
    plt.xlabel('Epochs', fontsize=10.5)
    plt.ylabel("MSE", fontsize=10.5)
    if i==0 or i==1  or i==2:
        plt.plot(data['loss'],label='learning rate = '+pred_files_list[i][13:19])
    elif i>=3 and i<=5:
        plt.plot(data['loss'],label='learning rate = '+pred_files_list[i][13:18])
    elif i>=6 and i<=7:
        plt.plot(data['loss'],label='learning rate = '+pred_files_list[i][13:17])
    elif i==9:
        plt.plot(data['loss'],label='learning rate = '+pred_files_list[i][13:16])
    plt.ylim([-0.0001,0.0025])
    plt.legend(
    # loc='upper left',
    loc=0,
    # bbox_to_anchor=(0.05,1),
    shadow=False,
    frameon=False,
    fontsize=10.5)
plt.subplots_adjust(left=0.12, bottom=0.12, right=0.98, top=0.96, hspace=0.2, wspace=0.2)
if MODEL_ID==1:
    plt.savefig(graph_path+'\\imf'+str(MODEL_ID)+'_learn_rate_ana.eps', transparent=False, format='EPS', dpi=2000)
else:
    plt.savefig(graph_path+'\\imf'+str(MODEL_ID)+'_learn_rate_ana.png', transparent=False, format='PNG', dpi=2000)

plt.show()


