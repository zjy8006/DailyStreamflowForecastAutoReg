#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# plt.rcParams['figure.figsize']=(10,8)
plt.rcParams['font.size']=6
# plt.rcParams["figure.figsize"] = [12, 8]
# plt.rcParams['image.cmap']='plasma'
# plt.rcParams['axes.linewidth']=0.8
from scipy.fftpack import fft


import os
root_path = os.path.dirname(os.path.abspath('__file__'))
# root_path = os.path.abspath(os.path.join(root_path, os.path.pardir))
# data_path = parent_path + '\\data\\'


#%%
yx_imfs = pd.read_csv(root_path+'/yx_vmd/data/VMD_TRAIN_K10.csv')
T=yx_imfs.shape[0]
t = np.arange(start=1,stop=T+1,step=1,dtype=np.float)/T
freqs = t-0.5-1/T
print(t)
yx_imf9=yx_imfs['IMF9']
yx_imf10=yx_imfs['IMF10']

zjs_imfs = pd.read_csv(root_path+'/zjs_vmd/data/VMD_TRAIN_K12.csv')
zjs_imf11=zjs_imfs['IMF11']
zjs_imf12=zjs_imfs['IMF12']

#%%
plt.figure(figsize=(3.54,3.54))
# plt.subplot(2,2,1)
# # plt.xlabel('Time(day)',)
# plt.title(r'$IMF_{9}$',loc='left',)
# plt.ylabel('Amplitude',)
# plt.plot(freqs,abs(fft(yx_imf9)),color='b',label='',linewidth=0.8)

plt.subplot(2,1,1)
plt.title(r'$IMF_{11}$',loc='left',)
# plt.xlabel('Frequency(1/day)',)
plt.ylabel('Amplitude',)
plt.plot(freqs,abs(fft(zjs_imf11)),color='b',label='',linewidth=0.8)


# plt.subplot(2,2,3)
# plt.title(r'$IMF_{10}$',loc='left',)
# plt.xlabel('Frequency(1/day)\n(a)',)
# plt.ylabel('Amplitude',)
# plt.plot(freqs,abs(fft(yx_imf10)),color='b',label='',linewidth=0.8,zorder=0)
# plt.vlines(x=-0.04,ymin=5,ymax=2500,lw=1.5,colors='r',zorder=1)
# plt.vlines(x=0.04,ymin=5,ymax=2500,lw=1.5,colors='r',zorder=1)
# plt.hlines(y=5,xmin=-0.04,xmax=0.04,lw=1.5,colors='r',zorder=1)
# plt.hlines(y=2500,xmin=-0.04,xmax=0.04,lw=1.5,colors='r',zorder=1)

plt.subplot(2,1,2)
plt.title(r'$IMF_{12}$',loc='left',)
plt.xlabel('Frequency (1/day)',)
plt.ylabel('Amplitude',)
plt.plot(freqs,abs(fft(zjs_imf12)),color='b',label='',linewidth=0.8,zorder=0)
plt.vlines(x=-0.04,ymin=5,ymax=1500,lw=1.5,colors='r',zorder=1)
plt.vlines(x=0.04,ymin=5,ymax=1500,lw=1.5,colors='r',zorder=1)
plt.hlines(y=5,xmin=-0.04,xmax=0.04,lw=1.5,colors='r',zorder=1)
plt.hlines(y=1500,xmin=-0.04,xmax=0.04,lw=1.5,colors='r',zorder=1)

plt.subplots_adjust(left=0.14, bottom=0.12, right=0.98, top=0.92, hspace=0.6, wspace=0.4)
plt.savefig(root_path+'/graph/zjs_aliasing.tif', format='TIFF',transparent=False, dpi=1200)
plt.savefig(root_path+'/graph/zjs_aliasing.eps', format='EPS',transparent=False, dpi=2000)
plt.show()

# %%
