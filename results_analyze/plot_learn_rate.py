import os
root_path = os.path.dirname(os.path.abspath('__file__'))
# root_path = os.path.abspath(os.path.join(root_path, os.path.pardir))
# root_path = os.path.abspath(os.path.join(root_path, os.path.pardir))
# data_path = parent_path + '\\data\\'
graph_path = root_path+'\\graph\\'

print(graph_path)

import matplotlib.pyplot as plt
# plt.rcParams['figure.figsize']=(10,8)
plt.rcParams['font.size']=6
# plt.rcParams["figure.figsize"] = [12, 8]
# plt.rcParams['image.cmap']='plasma'
# plt.rcParams['axes.linewidth']=0.8
import pandas as pd
import numpy as np
import re
def sort_key(s):
    if s:
        try:
            c = re.findall('LSTM-S1-LR[\d+$]-HU[8]-EPS[500]-BS[512]-DR[0.0]-DC[0.0]-SEED[1].csv', s)[0]
        except:
            c = -1
        return int(c)
def strsort(alist):
    alist.sort(key=sort_key,reverse=True)
    return alist

MODEL_ID = 1
model_path = root_path+'\\zjs_vmd\\projects\\lstm-models-history\\learning_rate_tuning_mse\\history\\s'+str(MODEL_ID)+'\\'
plt.figure(figsize=(5.51,3))
# pred_files_list=[]
# for files in os.listdir(model_path):
#     if files.find('HU[16]-EPS[500]-BS[512]-DR[0.0]-DC[0.0]-SEED[1]-HISTORY')>0:
#         pred_files_list.append(files)
#         # print(files)
# pred_files_list=strsort(pred_files_list)
# for IMF1
pred_files_list=[
    'LSTM-S1-LR[0.0001]-HU[8]-EPS[500]-BS[512]-DR[0.0]-DC[0.0]-SEED[1]-HISTORY-TRAIN-TEST.csv',
    'LSTM-S1-LR[0.0003]-HU[8]-EPS[500]-BS[512]-DR[0.0]-DC[0.0]-SEED[1]-HISTORY-TRAIN-TEST.csv',
    'LSTM-S1-LR[0.0007]-HU[8]-EPS[500]-BS[512]-DR[0.0]-DC[0.0]-SEED[1]-HISTORY-TRAIN-TEST.csv',
    'LSTM-S1-LR[0.001]-HU[8]-EPS[500]-BS[512]-DR[0.0]-DC[0.0]-SEED[1]-HISTORY-TRAIN-TEST.csv',
    'LSTM-S1-LR[0.003]-HU[8]-EPS[500]-BS[512]-DR[0.0]-DC[0.0]-SEED[1]-HISTORY-TRAIN-TEST.csv',
    'LSTM-S1-LR[0.007]-HU[8]-EPS[500]-BS[512]-DR[0.0]-DC[0.0]-SEED[1]-HISTORY-TRAIN-TEST.csv',
    'LSTM-S1-LR[0.01]-HU[8]-EPS[500]-BS[512]-DR[0.0]-DC[0.0]-SEED[1]-HISTORY-TRAIN-TEST.csv',
    'LSTM-S1-LR[0.03]-HU[8]-EPS[500]-BS[512]-DR[0.0]-DC[0.0]-SEED[1]-HISTORY-TRAIN-TEST.csv',
    'LSTM-S1-LR[0.07]-HU[8]-EPS[500]-BS[512]-DR[0.0]-DC[0.0]-SEED[1]-HISTORY-TRAIN-TEST.csv',
    'LSTM-S1-LR[0.1]-HU[8]-EPS[500]-BS[512]-DR[0.0]-DC[0.0]-SEED[1]-HISTORY-TRAIN-TEST.csv',
]
for i in range(0,len(pred_files_list)):
    print(pred_files_list[i])
    data = pd.read_csv(model_path+pred_files_list[i])
    plt.xticks()
    plt.yticks()
    plt.xlabel(r'$Epochs$', )
    plt.ylabel(r"$Loss$", )
    if i==0 or i==1  or i==2:
        plt.plot(data['loss'],label='Learning rate = '+pred_files_list[i][11:17])
    elif i>=3 and i<=5:
        plt.plot(data['loss'],label='Learning rate = '+pred_files_list[i][11:16])
    elif i>=6 and i<=7:
        plt.plot(data['loss'],label='Learning rate = '+pred_files_list[i][11:15])
    elif i==9:
        plt.plot(data['loss'],label='Learning rate = '+pred_files_list[i][11:14])
    plt.ylim([-0.0001,0.0025])
    plt.legend(
    # loc='upper left',
    loc=0,
    # bbox_to_anchor=(0.05,1),
    shadow=False,
    frameon=False,
    )
plt.subplots_adjust(left=0.12, bottom=0.12, right=0.98, top=0.96, hspace=0.2, wspace=0.2)
plt.savefig(graph_path+'\\zjs_imf'+str(MODEL_ID)+'_learn_rate_ana.eps', transparent=False, format='EPS', dpi=2000)
plt.savefig(graph_path+'\\zjs_imf'+str(MODEL_ID)+'_learn_rate_ana.tif', transparent=False, format='TIFF', dpi=1200)
plt.show()


