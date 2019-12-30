#%%
import matplotlib.pyplot as plt
plt.rcParams['font.size']=6
from datetime import datetime,timedelta
from collections import OrderedDict
import pandas as pd
import numpy as np
from scipy.fftpack import fft
import os
root_path = os.path.dirname(os.path.abspath('__file__'))
# root_path = os.path.abspath(os.path.join(root_path,os.path.pardir)) # For run in CMD
graphs_path = root_path+'/graph/'

#%%
vmd_train = pd.read_csv(root_path+"/zjs_vmd/data/VMD_TRAIN.csv")
vmd_full = pd.read_csv(root_path+"/zjs_vmd/data/VMD_FULL.csv")
vmd_test = {}
vmd_subsignals=['IMF1','IMF2','IMF3','IMF4','IMF5','IMF6','IMF7','IMF8','IMF9','IMF10','IMF11',]
for imf in vmd_subsignals:
    test=[]
    for i in range(5261,6574+1):
        data=pd.read_csv(root_path+"/zjs_vmd/data/vmd-test/vmd_appended_test"+str(i)+".csv")
        test.append((data[imf].iloc[data.shape[0]-1:]).values.flatten()[0])
    vmd_test[imf]=test

#%%
eemd_train = pd.read_csv(root_path+"/zjs_eemd/data/EEMD_TRAIN.csv")
eemd_full = pd.read_csv(root_path+"/zjs_eemd/data/EEMD_FULL.csv")
eemd_test = {}
eemd_subsignals=['IMF1','IMF2','IMF3','IMF4','IMF5','IMF6','IMF7','IMF8','IMF9','IMF10','IMF11','IMF12',]
for imf in eemd_subsignals:
    test=[]
    for i in range(5261,6574+1):
        data=pd.read_csv(root_path+"/zjs_eemd/data/eemd-test/eemd_appended_test"+str(i)+".csv")
        test.append((data[imf].iloc[data.shape[0]-1:]).values.flatten()[0])
    eemd_test[imf]=test


#%%
dwt_train = pd.read_csv(root_path+"/zjs_wd/data/db45-3/WD_TRAIN.csv")
dwt_full = pd.read_csv(root_path+"/zjs_wd/data/db45-3/WD_FULL.csv")
dwt_test = {}
dwt_subsignals=['D1','D2','D3','A3',]
for imf in dwt_subsignals:
    test=[]
    for i in range(5261,6574+1):
        data=pd.read_csv(root_path+"/zjs_wd/data/db45-3/wd-test/wd_appended_test"+str(i)+".csv")
        test.append((data[imf].iloc[data.shape[0]-1:]).values.flatten()[0])
    dwt_test[imf]=test



#%%
t_full=list(range(1,6575))
t_val = list(range(5261,6575))
plt.figure(figsize=(7.48,7.48))
for i in range(len(vmd_subsignals)):
    plt.subplot(6,2,i+1)
    plt.title(r'$'+vmd_subsignals[i][0:3]+'_{'+vmd_subsignals[i][3:]+'}$')
    plt.plot(t_full,vmd_full[vmd_subsignals[i]],c='b',label='Hindcasting experiment')
    plt.plot(t_val,vmd_test[vmd_subsignals[i]],c='r',label='Forecasting experiment')
    plt.xlim(5261,6574)
    plt.ylabel(r"Flow ($m^3/s$)")
    if i==0:
        plt.legend(
            loc='upper left',
            # bbox_to_anchor=(0.08,1.01, 1,0.101),
            bbox_to_anchor=(0.7, 1.5),
            # bbox_transform=plt.gcf().transFigure,
            ncol=4,
            shadow=False,
            frameon=True,
        )
    if i==9 or i==10:
        plt.xlabel("Time (28/05/2011-31/12/2014)")

plt.subplots_adjust(left=0.08, bottom=0.06, right=0.98,top=0.94, hspace=0.6, wspace=0.25)
plt.savefig(graphs_path+'zjs_vmd_hind_fore_decom.tif',format='TIFF',dpi=1200)
plt.savefig(graphs_path+'zjs_vmd_hind_fore_decom.eps',format='EPS',dpi=2000)
plt.show()

#%%

plt.figure(figsize=(7.48,7.48))
for i in range(len(eemd_subsignals)):
    plt.subplot(6,2,i+1)
    if i==11:
        plt.title(r'$R$')
    else:
        plt.title(r'$'+eemd_subsignals[i][0:3]+'_{'+eemd_subsignals[i][3:]+'}$')
    plt.plot(t_full,eemd_full[eemd_subsignals[i]],c='b',label='Hindcasting experiment')
    plt.plot(t_val,eemd_test[eemd_subsignals[i]],c='r',label='Forecasting experiment')
    plt.xlim(5261,6574)
    plt.ylabel(r"Flow ($m^3/s$)")
    if i==0:
        plt.legend(
            loc='upper left',
            # bbox_to_anchor=(0.08,1.01, 1,0.101),
            bbox_to_anchor=(0.67, 1.5),
            # bbox_transform=plt.gcf().transFigure,
            ncol=4,
            shadow=False,
            frameon=True,
        )
    if i==10 or i==11:
        plt.xlabel("Time (28/05/2011-31/12/2014)")

plt.subplots_adjust(left=0.08, bottom=0.06, right=0.98,top=0.94, hspace=0.6, wspace=0.25)
plt.savefig(graphs_path+'zjs_eemd_hind_fore_decom.tif',format='TIFF',dpi=1200)
plt.savefig(graphs_path+'zjs_eemd_hind_fore_decom.eps',format='EPS',dpi=2000)
plt.show()



#%%

plt.figure(figsize=(7.48,2.48))
for i in range(len(dwt_subsignals)):
    plt.subplot(2,2,i+1)
    plt.title(r'$'+dwt_subsignals[i][0]+'_{'+dwt_subsignals[i][1]+'}$')
    plt.plot(t_full,dwt_full[dwt_subsignals[i]],c='b',label='Hindcasting experiment')
    plt.plot(t_val,dwt_test[dwt_subsignals[i]],c='r',label='Forecasting experiment')
    plt.xlim(5261,6574)
    plt.ylabel(r"Flow ($m^3/s$)")
    if i==0:
        plt.legend(
            loc='upper left',
            # bbox_to_anchor=(0.08,1.01, 1,0.101),
            bbox_to_anchor=(0.7, 1.5),
            # bbox_transform=plt.gcf().transFigure,
            ncol=4,
            shadow=False,
            frameon=True,
        )
    if i==2 or i==3:
        plt.xlabel("Time (28/05/2011-31/12/2014)")

plt.subplots_adjust(left=0.08, bottom=0.14, right=0.98,top=0.86, hspace=0.6, wspace=0.25)
plt.savefig(graphs_path+'zjs_dwt_hind_fore_decom.tif',format='TIFF',dpi=1200)
plt.savefig(graphs_path+'zjs_dwt_hind_fore_decom.eps',format='EPS',dpi=2000)
plt.show()

#%%
