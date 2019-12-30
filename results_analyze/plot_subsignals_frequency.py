import matplotlib.pyplot as plt
plt.rcParams['font.size']=6
import math
import pandas as pd
import numpy as np
from scipy.fftpack import fft
import os
root_path = os.path.dirname(os.path.abspath('__file__'))
# root_path = os.path.abspath(os.path.join(root_path,os.path.pardir)) # For run in CMD
graph_path = root_path+'/graph/'

yx_vmd_train = pd.read_csv(root_path+"/yx_vmd/data/VMD_TRAIN.csv")
yx_eemd_train = pd.read_csv(root_path+"/yx_eemd/data/EEMD_TRAIN.csv")
yx_dwt_train = pd.read_csv(root_path+'/yx_wd/data/db45-3/WD_TRAIN.csv')

zjs_vmd_train = pd.read_csv(root_path+"/zjs_vmd/data/VMD_TRAIN.csv")
zjs_eemd_train = pd.read_csv(root_path+"/zjs_eemd/data/EEMD_TRAIN.csv")
zjs_dwt_train = pd.read_csv(root_path+'/zjs_wd/data/db45-3/WD_TRAIN.csv')


signals=[
    zjs_vmd_train['IMF11'],zjs_eemd_train['IMF1'],zjs_dwt_train['D1'],
    yx_vmd_train['IMF9'],yx_eemd_train['IMF1'],yx_dwt_train['D1'],
    ]
titles=[
    r"$IMF_{11}$ of VMD at ZJS",r"$IMF_{1}$ of EEMD at ZJS",r"$D_{1}$ of DWT (db45-3) at ZJS",
    r"$IMF_{9}$ of VMD at YX",r"$IMF_{1}$ of EEMD at YX",r"$D_{1}$ of DWT (db45-3) at YX",
]

plt.figure(figsize=(7.48,3.48))
for i in range(len(signals)):
    T=signals[i].shape[0]
    fs=1/T
    t = np.arange(start=1,stop=T+1,step=1,dtype=np.float)/T
    freqs = t-0.5-1/T
    plt.subplot(2,3,i+1)
    plt.title(titles[i])
    plt.plot(freqs,abs(fft(signals[i])),c='b',lw=0.8)
    plt.xlabel('Frequence (1/day)')
    plt.ylabel('Amplitude')
plt.subplots_adjust(left=0.08, bottom=0.11, right=0.99,top=0.94, hspace=0.5, wspace=0.4)
plt.savefig(graph_path+"frequency_of_most_difficult_predict_subsignal.tif",format="TIFF",dpi=1200)
plt.savefig(graph_path+"frequency_of_most_difficult_predict_subsignal.eps",format="EPS",dpi=2000)
plt.show()

