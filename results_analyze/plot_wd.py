import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# plt.rcParams['figure.figsize']=(10,8)
plt.rcParams['font.size']=6
# plt.rcParams["figure.figsize"] = [12, 8]
plt.rcParams['image.cmap']='plasma'
# plt.rcParams['axes.linewidth']=0.8
import os
root_path = os.path.dirname(os.path.abspath('__file__')) 
# root_path = os.path.abspath(os.path.join(root_path, os.path.pardir))
graph_path=root_path+'/graph/'
yx_path=root_path+'/yx_wd/projects/lstm-models-history/'
zjs_path=root_path+'/zjs_wd/projects/lstm-models-history/'

x=['YX-L1-T','YX-L1-D','YX-L3-T','YX-L3-D','YX-L5-T','YX-L5-D','YX-L7-T','YX-L7-D',
'ZJS-L1-T','ZJS-L1-D','ZJS-L3-T','ZJS-L3-D','ZJS-L5-T','ZJS-L5-D','ZJS-L7-T','ZJS-L7-D',]
y=['db2-1','db2-2','db2-3',
'db5-1','db5-2','db5-3',
'db10-1','db10-2','db10-3',
'db15-1','db15-2','db15-3',
'db20-1','db20-2','db20-3',
'db25-1','db25-2','db25-3',
'db30-1','db30-2','db30-3',
'db35-1','db35-2','db35-3',
'db40-1','db40-2','db40-3',
'db45-1','db45-2','db45-3',
'bior 3.3-1','bior 3.3-2','bior 3.3-3',
'coif3-1','coif3-2','coif3-3',
'haar-1','haar-2','haar-3']

x_ticks_dict={}
y_ticks_dict={}

for i in range(0,len(x)):
    x_ticks_dict[x[i]]=i
for i in range(0,len(y)):
    y_ticks_dict[y[::-1][i]]=i

MULTI_NSE_S={}
MULTI_NRMSE_S={}
MULTI_PPTS_S={}
ONE_NSE_S={}
ONE_NRMSE_S={}
ONE_PPTS_S={}
for wavelet_lev in y:
    MULTI_NSE=[]
    MULTI_NRMSE=[]
    MULTI_PPTS=[]
    ONE_NSE=[]
    ONE_NRMSE=[]
    ONE_PPTS=[]
    for station_lead in x:
        info_ = station_lead.split('-')
        station=info_[0]
        lead=(info_[1].split('L'))[1]
        stage=info_[2]
        if station=='YX' and stage=='T':
            multi_data = pd.read_csv(yx_path+wavelet_lev+'/multi_models_'+lead+'_ahead_forecast_pacf/model_metrics.csv')
            multi_nse_ = multi_data['train_nse'][0]
            multi_nrmse_ = multi_data['train_nrmse'][0]
            multi_ppts_ = multi_data['train_ppts'][0]
            MULTI_NSE.append(multi_nse_)
            MULTI_NRMSE.append(multi_nrmse_)
            MULTI_PPTS.append(multi_ppts_)

            one_data = pd.read_csv(yx_path+wavelet_lev+'/one_model_'+lead+'_ahead_forecast_pacf/model_metrics.csv')
            one_nse_ = one_data['train_nse'][0]
            one_nrmse_ = one_data['train_nrmse'][0]
            one_ppts_ = one_data['train_ppts'][0]
            ONE_NSE.append(one_nse_)
            ONE_NRMSE.append(one_nrmse_)
            ONE_PPTS.append(one_ppts_)
        elif station=='YX' and stage=='D':
            multi_data = pd.read_csv(yx_path+wavelet_lev+'/multi_models_'+lead+'_ahead_forecast_pacf/model_metrics.csv')
            multi_nse_ = multi_data['dev_nse'][0]
            multi_nrmse_ = multi_data['dev_nrmse'][0]
            multi_ppts_ = multi_data['dev_ppts'][0]
            MULTI_NSE.append(multi_nse_)
            MULTI_NRMSE.append(multi_nrmse_)
            MULTI_PPTS.append(multi_ppts_)

            one_data = pd.read_csv(yx_path+wavelet_lev+'/one_model_'+lead+'_ahead_forecast_pacf/model_metrics.csv')
            one_nse_ = one_data['dev_nse'][0]
            one_nrmse_ = one_data['dev_nrmse'][0]
            one_ppts_ = one_data['dev_ppts'][0]
            ONE_NSE.append(one_nse_)
            ONE_NRMSE.append(one_nrmse_)
            ONE_PPTS.append(one_ppts_)
        elif station=='ZJS' and stage=='T':
            multi_data = pd.read_csv(zjs_path+wavelet_lev+'/multi_models_'+lead+'_ahead_forecast_pacf/model_metrics.csv')
            multi_nse_ = multi_data['train_nse'][0]
            multi_nrmse_ = multi_data['train_nrmse'][0]
            multi_ppts_ = multi_data['train_ppts'][0]
            MULTI_NSE.append(multi_nse_)
            MULTI_NRMSE.append(multi_nrmse_)
            MULTI_PPTS.append(multi_ppts_)

            one_data = pd.read_csv(zjs_path+wavelet_lev+'/one_model_'+lead+'_ahead_forecast_pacf/model_metrics.csv')
            one_nse_ = one_data['train_nse'][0]
            one_nrmse_ = one_data['train_nrmse'][0]
            one_ppts_ = one_data['train_ppts'][0]
            ONE_NSE.append(one_nse_)
            ONE_NRMSE.append(one_nrmse_)
            ONE_PPTS.append(one_ppts_)
        elif station=='ZJS' and stage=='D':
            multi_data = pd.read_csv(zjs_path+wavelet_lev+'/multi_models_'+lead+'_ahead_forecast_pacf/model_metrics.csv')
            multi_nse_ = multi_data['dev_nse'][0]
            multi_nrmse_ = multi_data['dev_nrmse'][0]
            multi_ppts_ = multi_data['dev_ppts'][0]
            MULTI_NSE.append(multi_nse_)
            MULTI_NRMSE.append(multi_nrmse_)
            MULTI_PPTS.append(multi_ppts_)

            one_data = pd.read_csv(zjs_path+wavelet_lev+'/one_model_'+lead+'_ahead_forecast_pacf/model_metrics.csv')
            one_nse_ = one_data['dev_nse'][0]
            one_nrmse_ = one_data['dev_nrmse'][0]
            one_ppts_ = one_data['dev_ppts'][0]
            ONE_NSE.append(one_nse_)
            ONE_NRMSE.append(one_nrmse_)
            ONE_PPTS.append(one_ppts_)
        print(wavelet_lev+':'+station_lead+':MULTI NSE={}'.format(multi_nse_))
        print(wavelet_lev+':'+station_lead+':MULTI NRMSE={}'.format(multi_nrmse_))
        print(wavelet_lev+':'+station_lead+':MULTI PPTS={}'.format(multi_ppts_))
        print(wavelet_lev+':'+station_lead+':ONE NSE={}'.format(one_nse_))
        print(wavelet_lev+':'+station_lead+':ONE NRMSE={}'.format(one_nrmse_))
        print(wavelet_lev+':'+station_lead+':ONE PPTS={}'.format(one_ppts_))
    MULTI_NSE_S[wavelet_lev]=MULTI_NSE
    MULTI_NRMSE_S[wavelet_lev]=MULTI_NRMSE
    MULTI_PPTS_S[wavelet_lev]=MULTI_PPTS
    ONE_NSE_S[wavelet_lev]=ONE_NSE
    ONE_NRMSE_S[wavelet_lev]=ONE_NRMSE
    ONE_PPTS_S[wavelet_lev]=ONE_PPTS
multi_nse_df = pd.DataFrame(MULTI_NSE_S,index=x)
multi_nrmse_df = pd.DataFrame(MULTI_NRMSE_S,index=x)
multi_ppts_df = pd.DataFrame(MULTI_PPTS_S,index=x)
one_nse_df = pd.DataFrame(ONE_NSE_S,index=x)
one_nrmse_df = pd.DataFrame(ONE_NRMSE_S,index=x)
one_ppts_df = pd.DataFrame(ONE_PPTS_S,index=x)
print('multi nse:\n{}'.format(multi_nse_df))
print('multi nrmse:\n{}'.format(multi_nrmse_df))
print('multi ppts:\n{}'.format(multi_ppts_df))
print('one nse:\n{}'.format(one_nse_df))
print('one nrmse:\n{}'.format(one_nrmse_df))
print('one ppts:\n{}'.format(one_ppts_df))

multi_max_nse=multi_nse_df.idxmax(axis=1)
multi_min_nrmse=multi_nrmse_df.idxmin(axis=1)
multi_min_ppts=multi_ppts_df.idxmin(axis=1)
one_max_nse=one_nse_df.idxmax(axis=1)
one_min_nrmse=one_nrmse_df.idxmin(axis=1)
one_min_ppts=one_ppts_df.idxmin(axis=1)
print('multi max nse:\n{}'.format(multi_max_nse))
print('multi min nrmse:\n{}'.format(multi_min_nrmse))
print('multi min ppts:\n{}'.format(multi_min_ppts))
print('one max nse:\n{}'.format(one_max_nse))
print('one min nrmse:\n{}'.format(one_min_nrmse))
print('one min ppts:\n{}'.format(one_min_ppts))
data = [multi_nse_df.T,multi_nrmse_df.T,multi_ppts_df.T,
        one_nse_df.T,one_nrmse_df.T,one_ppts_df.T,
]
optimal=[multi_max_nse,multi_min_nrmse,multi_min_ppts,
    one_max_nse,one_min_nrmse,one_min_ppts,]
name=['multi_nse_wd','multi_nrmse_wd','multi_ppts_wd',
'one_nse_wd','one_nrmse_wd','one_ppts_wd',
]

for i in range(0,len(data)):
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(3.5,6))
    max_v = round(max(data[i].max(axis=1)),1)
    min_v = round(min(data[i].min(axis=1)),1)
    interval=round((max_v-min_v)/4,1)
    if i==0 or i==3:
        im = ax.imshow(data[i], extent=[0,len(x),0,len(y)],cmap='viridis',vmin=0,vmax=1,interpolation='none',aspect='equal')  
    else:
        im = ax.imshow(data[i], extent=[0,len(x),0,len(y)],cmap='viridis_r',vmin=min_v,vmax=max_v,interpolation='none',aspect='equal')  
    ax.set_xticks(np.arange(0.5,len(x)+0.5,1))
    ax.set_yticks(np.arange(0.5,len(y)+0.5,1))
    ax.set_xticklabels(x,rotation=90)
    ax.set_yticklabels(y[::-1])
    print(x_ticks_dict)
    print(y_ticks_dict)
    for x_ in x:
        xx=x_ticks_dict[x_]
        yy=y_ticks_dict[optimal[i][x_]]
        plt.vlines(x=xx,ymin=yy,ymax=yy+1,colors='r',zorder=1)
        plt.vlines(x=xx+1,ymin=yy,ymax=yy+1,colors='r',zorder=1)
        plt.hlines(y=yy,xmin=xx,xmax=xx+1,colors='r',zorder=1)
        plt.hlines(y=yy+1,xmin=xx,xmax=xx+1,colors='r',zorder=1)
    cb_ax = fig.add_axes([0.85, 0.09, 0.06, 0.9])#[left,bottom,width,height]
    cbar = fig.colorbar(im, cax=cb_ax)
    if i==0 or i==3:
        cbar.set_ticks(np.arange(0, 1.1, 0.2))
    else:
        cbar.set_ticks(np.arange(min_v,max_v+0.1, interval))

    fig.subplots_adjust(bottom=0.09, top=0.99, left=0.02, right=0.96,wspace=0.3, hspace=0.1)
    # plt.savefig(graph_path+"nse_wd.eps",format="EPS",dpi=2000)
    plt.savefig(graph_path+name[i]+".tif",format="TIFF",dpi=1200)

    
plt.show()







