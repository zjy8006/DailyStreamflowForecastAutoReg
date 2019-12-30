import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# plt.rcParams['figure.figsize']=(10,8)
plt.rcParams['font.size']=6
# plt.rcParams["figure.figsize"] = [12, 8]
# plt.rcParams['image.cmap']='plasma'
# plt.rcParams['axes.linewidth']=0.8
import os
root_path = os.path.dirname(os.path.abspath('__file__')) 
# root_path = os.path.abspath(os.path.join(root_path, os.path.pardir)) # for runing code in CMD
graph_path=root_path+'/graph/'


pacfs_vmd=pd.read_csv(root_path+'/zjs_vmd/data/PACF.csv')
print(pacfs_vmd)


lags=list(range(0,21))
t=list(range(-1,21))
z_line=np.zeros(len(t))
plt.figure(figsize=(3.54,3))
plt.xlim(-0.5,20)
plt.ylim(-1,1)
plt.xlabel('Lag (day)', )
plt.ylabel('PACF', )
plt.bar(lags,pacfs_vmd['IMF1'],color='b',width=0.8)
plt.plot([-1,21],[pacfs_vmd['UP'][0],pacfs_vmd['UP'][0]], '--', color='r', label='',lw=0.8)
plt.plot([-1,21],[pacfs_vmd['LOW'][0],pacfs_vmd['LOW'][0]], '--', color='r', label='',lw=0.8)
plt.plot(t,z_line, '-', color='blue', label='',)
plt.subplots_adjust(left=0.15, bottom=0.14, right=0.96,top=0.97, hspace=0.4, wspace=0.3)
plt.savefig(graph_path+'\\pacf_zjs_vmd_imf1.tif', transparent=False, format='TIFF', dpi=1200)
plt.savefig(graph_path+'\\pacf_zjs_vmd_imf1.eps', transparent=False, format='EPS', dpi=2000)
plt.show()

