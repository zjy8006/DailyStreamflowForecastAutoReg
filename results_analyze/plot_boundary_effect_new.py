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
subsignal="IMF1"
TRAIN = pd.read_csv(root_path+"/zjs_vmd/data/VMD_TRAIN.csv")[subsignal]
FULL = pd.read_csv(root_path+"/zjs_vmd/data/VMD_FULL.csv")[subsignal]
# TRAIN = pd.read_csv(root_path+"/zjs_eemd/data/EEMD_TRAIN.csv")[subsignal]
# FULL = pd.read_csv(root_path+"/zjs_eemd/data/EEMD_FULL.csv")[subsignal]



dates=["1953-01-01","2019-01-01"]
start,end = [datetime.strptime(_, "%Y-%m-%d") for _ in dates]
m=list(OrderedDict(((start + timedelta(_)).strftime(r"%b-%y"), None) for _ in range((end - start).days)).keys())
# print((m[5249:5270]))
test_imf = []
for i in range(5261,6574+1):
    data=pd.read_csv(root_path+"/zjs_vmd/data/vmd-test/vmd_appended_test"+str(i)+".csv")
    # data=pd.read_csv(root_path+"/zjs_eemd/data/eemd-test/eemd_appended_test"+str(i)+".csv")
    test_imf.append((data[subsignal].iloc[data.shape[0]-1:]).values.flatten()[0])
t_e=list(range(1,6575))
t_t=list(range(1,5261))

#%%
# from scipy import signal
# x1 = np.arange(5235,5261,0.1)
# x2 = np.arange(5260,5280,0.1)
# data1=100*(np.sin(x1)*np.cos(0.2*x1-10))+200
# data2=100*(np.sin(x2)*np.cos(0.2*x2-10))+200

# from numpy import *
# # that's the line, you need:
# a = diff(sign(diff(data1))).nonzero()[0] + 1 # local min+max
# b = (diff(sign(diff(data1))) > 0).nonzero()[0] + 1 # local min
# c = (diff(sign(diff(data1))) < 0).nonzero()[0] + 1 # local max

# # graphical output...
# from pylab import *
# plot(x1,data1,c='b',label='observations')
# plot(x2,data2,'--',c='r',label='future values')
# plot(x1[b], data1[b], "--", label="lower envelope")
# plot(x1[c], data1[c], "--", label="upper envelope")
# legend()
# show()


#%%

x1 = np.arange(5230,5260,0.1)
x2 = np.arange(5260,6574,0.1)
data1=100*(np.sin(x1)*np.cos(0.2*x1-10))+150
data2=100*(np.sin(x2)*np.cos(0.2*x2-10))+150



from numpy import *
# that's the line, you need:
a1 = diff(sign(diff(data1))).nonzero()[0] + 1 # local min+max
b1 = (diff(sign(diff(data1))) > 0).nonzero()[0] + 1 # local min
c1 = (diff(sign(diff(data1))) < 0).nonzero()[0] + 1 # local max
a2 = diff(sign(diff(data2))).nonzero()[0] + 1 # local min+max
b2 = (diff(sign(diff(data2))) > 0).nonzero()[0] + 1 # local min
c2 = (diff(sign(diff(data2))) < 0).nonzero()[0] + 1 # local max
# upper_x=[
#     5230,5231,5232,5233,5234,5235,5235.40000000002,
#     5236,5237,5238,5239,5240,5240.800000000039,
#     5241,5242,5243,5244,5245.100000000055,
#     5246,5247,5248,5249,5250,5251.100000000077,
#     5252,5253,5254,5255,5256,5256.500000000096,
# ]
# upper=[
#     230,235,240,245,250,252,243.59169408234865,
#     240,220,200,180,160,157.9373243131965,
#     165,185,195,205,214.7414626522648,
#     220,225,230,235,240,243.59585608323323,
#     230,215,200,185,170,157.94027014588613,

# ]

lower={}
upper={}
for b in b1:
    lower[x1[b]]=data1[b]
    print('Local min index='+str(x1[b])+'; value='+str(data1[b]))
for c in c1:
    upper[x1[c]]=data1[c]
    print('Local max index='+str(x1[c])+'; value='+str(data1[c]))

upper[5230]=230
upper[5231]=235
upper[5232]=245
upper[5233]=248
upper[5234]=250
upper[5235]=252
upper[5236]=240
upper[5237]=220
upper[5238]=200
upper[5239]=180
upper[5240]=160
upper[5241]=165
upper[5242]=185
upper[5243]=195
upper[5244]=205
upper[5245]=214
upper[5246]=220
upper[5247]=225
upper[5248]=230
upper[5249]=235
upper[5250]=240
upper[5251]=243
upper[5252]=230
upper[5253]=215
upper[5254]=200
upper[5255]=185
upper[5256]=170


upper_extra={
    5256.500000000096:157.94027014588613,
    5257:180,
    5258:200,
    5259:220,
    5260:250,
}
lower_extra={
    5258.200000000103:135.24529620507448,
    5259:125,
    5260:113.344,
}
lower[5230]=88.0
lower[5231]=78.0
lower[5232]=58.0

upper_x=[]
upper_y=[]
for key in sorted(upper.keys()):
    upper_x.append(key)
    upper_y.append(upper[key])
upper_extra_x=[]
upper_extra_y=[]
for key in sorted(upper_extra.keys()):
    upper_extra_x.append(key)
    upper_extra_y.append(upper_extra[key])
lower_x=[]
lower_y=[]
for key in sorted(lower.keys()):
    lower_x.append(key)
    lower_y.append(lower[key])
lower_extra_x=[]
lower_extra_y=[]
for key in sorted(lower_extra.keys()):
    lower_extra_x.append(key)
    lower_extra_y.append(lower_extra[key])


plt.figure(figsize=(7.48,5.5))
ax1=plt.subplot2grid((2,2),(0,0),colspan=2)
ax2=plt.subplot2grid((2,2),(1,0))
ax3=plt.subplot2grid((2,2),(1,1))
ax1.text(5230.2,380,'(a)',fontweight='normal')
ax1.set_xlabel('Time (03/05/2011-07/06/2011)')
ax1.set_ylabel(r"Flow ($m^3/s$)")
# ax1.axis('off')
# ax1.set_xticks([0,5,10,15,20,25,30])
# ax1.set_yticks([0,20,40,60,80,100])
ax1.plot(x1,data1,c='b',label='Observations')
ax1.plot(x2,data2,'--',c='r',label='Future values')
ax1.vlines(x=5260,ymin=0,ymax=400,label='The right boundary of observations',linestyle='--',color='black',linewidth=0.5)
ax1.plot(x1[b1],data1[b1],'o',color='purple',label='Local minima of observations')
ax1.plot(x1[c1],data1[c1],'o',color='tab:orange',label='Local maxima of observations')
ax1.plot(x2[b2],data2[b2],'o',color='tab:blue',label='Local minima of future values')
ax1.plot(x2[c2],data2[c2],'o',color='tab:green',label='Local maxima of future values')

ax1.plot([5215,5220],[280,280],'--',color='tab:orange',label='Upper envelope')
ax1.plot([5215,5220],[280,280],'--',color='purple',label='Lower envelope')
ax1.plot([5215,5220],[280,280],'--',color='tab:green',label='Extrapolated upper envelope')
ax1.plot([5215,5220],[280,280],'--',color='tab:blue',label='Extrapolated lower envelope')
ax1.plot([5215,5220],[280,280],'--',color='tab:orange',label='The actual upper envelope close to the right boundary',linewidth=0.5)
ax1.plot([5215,5220],[280,280],'--',color='purple',label='The actual lower envelope close to the right boundary',linewidth=0.5)
# ax1.grid(b=True)
# ax1.annotate('The right end of observations',
#             xy=(5260, 190), xycoords='data',
#             xytext=(0.7, 0.75), textcoords='axes fraction',
#             arrowprops=dict(facecolor='black', shrink=0.1),
#             horizontalalignment='right', verticalalignment='top')
# ax1.annotate('Upper envelope',
#             xy=(5245.15, 215), xycoords='data',
#             xytext=(0.4, 0.75), textcoords='axes fraction',
#             arrowprops=dict(facecolor='black', shrink=0.1),
#             horizontalalignment='right', verticalalignment='top')

# ax1.annotate('Lower envelope',
#             xy=(5238.35, 90), xycoords='data',
#             xytext=(0.3, 0.05), textcoords='axes fraction',
#             arrowprops=dict(facecolor='black', shrink=0.1),
#             horizontalalignment='right', verticalalignment='top')

# ax1.annotate('Extrapolated envelope',
#             xy=(5259.2, 115), xycoords='data',
#             xytext=(0.66, 0.05), textcoords='axes fraction',
#             arrowprops=dict(facecolor='black', shrink=0.1),
#             horizontalalignment='right', verticalalignment='top')

ax1.set_xlim(5230,5272)
ax1.set_ylim(0,400)
ax1.legend(ncol=3)
#===========================================================================================
print(t_e)
print(FULL)
ax2.plot(t_t,TRAIN,c='r',label="Concurrent decomposition of training set")
ax2.plot(t_e,FULL,c='b',label="Concurrent decomposition of entire streamflow")
ax2.set_xlabel("Time (03/05/2011-02/06/2011)")
ax2.text(5230.2,9.5,'(b)',fontweight='normal')
ax2.set_ylabel(r"Flow ($m^3/s$)")
ax2.set_xlim([5230,5265])
ax2.set_ylim([-0,10])
ax2.legend(loc='lower right')


t=list(range(5261,6575))
# print(t)
# print(test_imf)
ax3.plot(t,test_imf,c='r',label="Sequential decomposition of validation set")
ax3.plot(t,FULL.iloc[FULL.shape[0]-1314:],c='b',label="Concurrent decomposition of entire streamflow")
ax3.set_xlabel("Time (28/05/2011-31/12/2014)")
ax3.text(5202,285,'(c)',fontweight='normal')
ax3.set_ylabel(r"Flow ($m^3/s$)")
# plt.xlim([550,560])
ax3.set_ylim([0,300])
ax3.legend()
# plt.tight_layout()
plt.subplots_adjust(left=0.07, bottom=0.08, right=0.99,top=0.98, hspace=0.3, wspace=0.2)
plt.savefig(graphs_path+'/boundary_effect.eps',format='EPS',dpi=2000)
plt.savefig(graphs_path+'/boundary_effect.tif',format='TIFF',dpi=1200)
plt.show()


# %%
