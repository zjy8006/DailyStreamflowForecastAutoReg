#%%
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.ticker as ticker
from matplotlib import cm
import matplotlib.pylab
# plt.rcParams['figure.figsize']=(10,8)
plt.rcParams['font.size'] = 6
# plt.rcParams["figure.figsize"] = [12, 8]
plt.rcParams['image.cmap'] = 'plasma'
# plt.rcParams['axes.linewidth']=0.8
root_path = os.path.dirname(os.path.abspath('__file__'))
# root_path = os.path.abspath(os.path.join(root_path, os.path.pardir))

import sys
sys.path.append(root_path+'/tools/')
from metrics_ import normalized_mean_square_error,normalized_root_mean_square_error


graph_path = root_path+'/graph/'
model_path = root_path + \
    '/zjs_vmd/projects/lstm-models-history/multi_models_1_ahead_forecast_pacf/history/s1/'

# HU1=[8,16,24,32]
# HU2=[0,8,16,24,32]
# METRICS=[]
# for h1 in HU1:
#     for h2 in HU2:
#         metrics=[]
#         if h2==0:
#             HU='['+str(h1)+']'
#         else:
#             HU='['+str(h1)+','+str(h2)+']'
#         for dr1 in [0.0,0.1,0.2,0.3,0.4,0.5]:
#             for dr2 in [0.0,0.1,0.2,0.3,0.4,0.5]:
#                 data = model_path+'LSTM-S1-LR[0.01]-HU'+HU+'EPS[1000]-BS[512]-DR'

#%%
criterion = 'NRMSE'

HU1 = [8, 16, 24, 32]
DR1 = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6,0.7,0.8,0.9]
METRICS = []
HU1.reverse()
DR1.reverse()
print(HU1)
print(DR1)
met={}
for h1 in HU1:
    metrics = []
    for dr in DR1:
        data = pd.read_csv(model_path+'LSTM-S1-LR[0.01]-HU['+str(
            h1)+']-EPS[1000]-BS[512]-DR['+str(dr)+']-DC[0.0]-SEED[1].csv')
        dev_pred = data['dev_pred'][0:657]
        dev_y = data['dev_y'][0:657]
        nmse = normalized_mean_square_error(y_true=dev_y,y_pred=dev_pred)
        nrmse = normalized_root_mean_square_error(y_true=dev_y,y_pred=dev_pred)
        print(nmse)
        if criterion=='MSE':
            metrics.append(data['rmse_dev'][0]**2)
        elif criterion=='NMSE':
            metrics.append(nmse)
        elif criterion=='NRMSE':
            metrics.append(nrmse)
    METRICS.append(metrics)
    met[str(h1)]=metrics
print(pd.DataFrame(met))
metrics_df = pd.DataFrame(METRICS,index=['32','24','16','8'],columns=['0.9','0.8','0.7','0.6','0.5','0.4','0.3','0.2','0.1','0.0'])
print(metrics_df)
if criterion=='MSE':
    metrics_df.to_csv(root_path+'/results_analyze/results/zjs_vmd_imf1_mse_metrics.csv')
elif criterion=='NMSE':
    metrics_df.to_csv(root_path+'/results_analyze/results/zjs_vmd_imf1_nmse_metrics.csv')
elif criterion =='NRMSE':
    metrics_df.to_csv(root_path+'/results_analyze/results/zjs_vmd_imf1_nrmse_metrics.csv')
print('minimum in index = {}'.format(metrics_df.idxmin(axis=0)))
print('minimum in columns = {}'.format(metrics_df.idxmin(axis=1)))

def convert_grid_to_array(data):
    """ This converts the explicit square grid into the x,y,dz positional arrays required for plot_matrix core """
    xpos = []
    currentx = 0
    ypos = []
    currenty = 0
    dz = []
    adata = np.array(data)
    # print adata
    # print adata.transpose()

    arr_data_inverse = adata.transpose()
    for i in arr_data_inverse:
        for j in i:
            zdata = arr_data_inverse[currentx][currenty]
            if zdata != 0:
                xpos.append(currentx)
                ypos.append(currenty)
                dz.append(zdata)
            currenty = currenty+1
            # print xpos, ypos, dz
        currenty=0
        currentx=currentx+1
    return xpos, ypos, dz
 
def plot_matrix(data, color, filename):
    fig = plt.figure(figsize=(3.54,3.54))
    ax2 = Axes3D(fig)
 
    xpos, ypos, dz = convert_grid_to_array(data)
 
    zpos = np.zeros_like(xpos)
    dx = 0.5 * np.ones_like(zpos)
    dy = 0.5*np.ones_like(zpos)
    print('xpos={}'.format(xpos))
    print('ypos={}'.format(ypos))
    print('zpos={}'.format(list(zpos)))
    print('dx={}'.format(list(dx)))
    print('dy={}'.format(list(dy)))
    print('dz={}'.format(dz))
    # ax2.bar3d(xpos, ypos, zpos, dx, dy, dz, color=color)
    ax2.bar3d(xpos, ypos, zpos, dx, dy, dz,)
    ax2.dist = 11
    ax2.set_xlim3d(0,9)
    ax2.set_ylim3d(0,4)
    # ax2.set_zlim3d(0,15)
 
    ax2.set_xlabel(r'$Dropout$')
    ax2.set_ylabel(r'$Hidden\ units$')
    if criterion=='MSE':
        ax2.set_zlabel(r'$MSE$')
    elif criterion=='NMSE':
        ax2.set_zlabel(r'$NMSE$')
    elif criterion=='NRMSE':
        ax2.set_zlabel(r'$NRMSE$')
 
    xformatter = (r'$0.9$',r'$0.8$', r'$0.7$', r'$0.6$',r'$0.5$', r'$0.4$', r'$0.3$', r'$0.2$', r'$0.1$', r'$0.0$','')
    ax2.w_xaxis.set_major_formatter(ticker.FixedFormatter(xformatter))
 
    yformatter = ('',r'$32$', '', r'$24$','', r'$16$','', r'$8$',)
    ax2.w_yaxis.set_major_formatter(ticker.FixedFormatter(yformatter))
    # fig.subplots_adjust(bottom=0.4, top=0.99, left=0.02, right=0.6,wspace=0.3, hspace=0.1)
    plt.savefig(graph_path+"".join([filename, ".tif"]),format='TIFF',dpi=1200)
    plt.savefig(graph_path+"".join([filename, ".eps"]),format='EPS',dpi=2000)

if criterion=='MSE':
    plot_matrix(data=METRICS,color='g',filename='zjs_vmd_multi_fore_imf1_mse')
elif criterion=='NMSE':
    plot_matrix(data=METRICS,color='g',filename='zjs_vmd_multi_fore_imf1_nmse')
elif criterion=='NRMSE':
    plot_matrix(data=METRICS,color='g',filename='zjs_vmd_multi_fore_imf1_nrmse')
plt.show()



# %%
