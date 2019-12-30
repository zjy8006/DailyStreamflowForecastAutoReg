#%% 1
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

x=[
    'MEF-ZJS-L1-T','MEF-ZJS-L1-D','MEF-ZJS-L3-T','MEF-ZJS-L3-D','MEF-ZJS-L5-T','MEF-ZJS-L5-D','MEF-ZJS-L7-T','MEF-ZJS-L7-D',
    'MEF-YX-L1-T','MEF-YX-L1-D','MEF-YX-L3-T','MEF-YX-L3-D','MEF-YX-L5-T','MEF-YX-L5-D','MEF-YX-L7-T','MEF-YX-L7-D',
    'SF-ZJS-L1-T','SF-ZJS-L1-D','SF-ZJS-L3-T','SF-ZJS-L3-D','SF-ZJS-L5-T','SF-ZJS-L5-D','SF-ZJS-L7-T','SF-ZJS-L7-D',
    'SF-YX-L1-T','SF-YX-L1-D','SF-YX-L3-T','SF-YX-L3-D','SF-YX-L5-T','SF-YX-L5-D','SF-YX-L7-T','SF-YX-L7-D',
    'SFMIS-ZJS-L1-T','SFMIS-ZJS-L1-D','SFMIS-ZJS-L3-T','SFMIS-ZJS-L3-D','SFMIS-ZJS-L5-T','SFMIS-ZJS-L5-D','SFMIS-ZJS-L7-T','SFMIS-ZJS-L7-D',
    'SFMIS-YX-L1-T','SFMIS-YX-L1-D','SFMIS-YX-L3-T','SFMIS-YX-L3-D','SFMIS-YX-L5-T','SFMIS-YX-L5-D','SFMIS-YX-L7-T','SFMIS-YX-L7-D',
    ]
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
'haar-1','haar-2','haar-3','EEMD','VMD','monoscale']



NSE_S={}
NMSE_S={}
NRMSE_S={}
PPTS_S={}

pattern_dict={
    'MEF':['multi_models_','forecast_pacf'],
    'SF':['one_model_','forecast_pacf'],
    'SFMIS':['one_model_','forecast_pacf_mis'],
    'MEH':['multi_models_','hindcast_pacf'],
    'SH':['one_model_','hindcast_pacf'],
    'SHMIS':['one_model_','hindcast_pacf_mis'],
}
for decomposer in y:
    NSE=[]
    NMSE=[]
    NRMSE=[]
    PPTS=[]
    
    for station_lead in x:
        info_ = station_lead.split('-')
        pattern=info_[0]
        station=info_[1]
        lead=(info_[2].split('L'))[1]
        stage=info_[3]
        print('Pattern:{}'.format(pattern))
        print('Station:{}'.format(station))
        print('leading time:{}'.format(lead))
        print('Stage:{}'.format(stage))

        if decomposer=='monoscale':
            data = pd.read_csv(root_path+'/'+station.lower()+'_orig/projects/lstm-models-history/'+lead+'_ahead/model_metrics.csv')
        elif decomposer=='EEMD' or decomposer=='VMD':
            data = pd.read_csv(root_path+'/'+station.lower()+'_'+decomposer.lower()+'/projects/lstm-models-history/'+pattern_dict[pattern][0]+lead+'_ahead_'+pattern_dict[pattern][1]+'/model_metrics.csv')
        else:
            data = pd.read_csv(root_path+'/'+station.lower()+'_wd/projects/lstm-models-history/'+decomposer.lower()+'/'+pattern_dict[pattern][0]+lead+'_ahead_'+pattern_dict[pattern][1]+'/model_metrics.csv')
            
        if stage=='T':
            nse_ = data['train_nse'][0]
            nmse_ = data['train_nmse'][0]
            nrmse_ = data['train_nrmse'][0]
            ppts_ = data['train_ppts'][0]
            NSE.append(nse_)
            NMSE.append(nmse_)
            NRMSE.append(nrmse_)
            PPTS.append(ppts_)
        elif  stage=='D':
            nse_ = data['dev_nse'][0]
            nmse_ = data['dev_nmse'][0]
            nrmse_ = data['dev_nrmse'][0]
            ppts_ = data['dev_ppts'][0]
            NSE.append(nse_)
            NMSE.append(nmse_)
            NRMSE.append(nrmse_)
            PPTS.append(ppts_)
        print('ppts:\n{}'.format(ppts_))

    NSE_S[decomposer]=NSE
    NMSE_S[decomposer]=NMSE
    NRMSE_S[decomposer]=NRMSE
    PPTS_S[decomposer]=PPTS


#%% 2
print(PPTS_S)
nse_df = pd.DataFrame(NSE_S,index=x)
nmse_df = pd.DataFrame(NMSE_S,index=x)
nrmse_df = pd.DataFrame(NRMSE_S,index=x)
ppts_df = pd.DataFrame(PPTS_S,index=x)

nse_df.to_csv(root_path+'/results_analyze/results/nse.csv')
nmse_df.to_csv(root_path+'/results_analyze/results/nmse.csv')
nrmse_df.to_csv(root_path+'/results_analyze/results/nrmse.csv')
ppts_df.to_csv(root_path+'/results_analyze/results/ppts.csv')

# print('nse:\n{}'.format(nse_df))
# print('nrmse:\n{}'.format(nrmse_df))
print("="*100)
print('ppts:\n{}'.format(ppts_df))

max_nse1=nse_df.idxmax(axis=1)
min_nmse1=nmse_df.idxmin(axis=1)
min_nrmse1=nrmse_df.idxmin(axis=1)
min_ppts1=ppts_df.idxmin(axis=1)

max_nse0=nse_df.idxmax(axis=0)
min_nmse0=nmse_df.idxmin(axis=0)
min_nrmse0=nrmse_df.idxmin(axis=0)
min_ppts0=ppts_df.idxmin(axis=0)

print('multi max nse(column):\n{}'.format(max_nse1))
print('multi min nmse(column):\n{}'.format(min_nmse1))
print('multi min nrmse(column):\n{}'.format(min_nrmse1))
print('multi min ppts(column):\n{}'.format(min_ppts1))
print('multi max nse(row):\n{}'.format(max_nse0))
print('multi min nmse(row):\n{}'.format(min_nmse0))
print('multi min nrmse(row):\n{}'.format(min_nrmse0))
print('multi min ppts(row):\n{}'.format(min_ppts0))

data = [nse_df.T,nmse_df.T,nrmse_df.T,ppts_df.T,]
optimal1=[max_nse1,min_nmse1,min_nrmse1,min_ppts1,]
optimal0=[max_nse0,min_nmse0,min_nrmse0,min_ppts0,]
name=['NSE','NMSE','NRMSE','PPTS',]
print('optimal0=\n{}'.format(optimal0))
print('optimal1=\n{}'.format(optimal1))


#%% 3
x_ticks_dict={}
y_ticks_dict={}
for i in range(0,len(x)):
    x_ticks_dict[x[i]]=i
for i in range(0,len(y)):
    y_ticks_dict[y[::-1][i]]=i
x_ticks_dict=pd.DataFrame(x_ticks_dict,index=[0])
y_ticks_dict=pd.DataFrame(y_ticks_dict,index=[0])
print('x_ticks_dict1={}\n'.format(x_ticks_dict))
print('y_ticks_dict1={}\n'.format(y_ticks_dict))
x_ticks_dict.to_csv('x_ticks_dict.csv')
y_ticks_dict.to_csv('y_ticks_dict.csv')



#%% 4
colorbar_labels=[r'$NSE$',r'$NMSE$',r'$NRMSE$',r'$PPTS(5)(\%)$']
for i in range(0,len(data)):
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(7.48,6.38))
    max_v = round(max(data[i].max(axis=1)),1)
    min_v = round(min(data[i].min(axis=1)),1)
    interval=round((max_v-min_v)/4,1)
    if i==0:
        im = ax.imshow(data[i], extent=[0,len(x),0,len(y)],cmap='viridis',vmin=0,vmax=1,interpolation='none',aspect='equal')  
    else:
        im = ax.imshow(data[i], extent=[0,len(x),0,len(y)],cmap='viridis_r',vmin=min_v,vmax=max_v,interpolation='none',aspect='equal')  
    ax.set_xticks(np.arange(0.5,len(x)+0.5,1))
    ax.set_yticks(np.arange(0.5,len(y)+0.5,1))
    ax.set_xticklabels(x,rotation=90)
    ax.set_yticklabels(y[::-1])
    for x_ in x:
        xx1=x_ticks_dict[x_]
        yy1=y_ticks_dict[optimal1[i][x_]]
        print('xx1={}'.format(xx1))
        print('yy1={}'.format(yy1))
        plt.vlines(x=xx1,ymin=yy1,ymax=yy1+1,colors='r',zorder=1)
        plt.vlines(x=xx1+1,ymin=yy1,ymax=yy1+1,colors='r',zorder=1)
        plt.hlines(y=yy1,xmin=xx1,xmax=xx1+1,colors='r',zorder=1)
        plt.hlines(y=yy1+1,xmin=xx1,xmax=xx1+1,colors='r',zorder=1)
    # for y_ in y:
    #     xx0=y_ticks_dict[y_]
    #     yy0=x_ticks_dict[optimal0[i][y_]]
    #     print('xx0={}'.format(xx0))
    #     print('yy0={}'.format(yy0))
    #     # plt.hlines(y=xx0,xmin=yy0,xmax=yy0+0.5,colors='r',zorder=1)
    #     plt.hlines(y=xx0+0.5,xmin=yy0,xmax=yy0+1,colors='r',zorder=1)
    #     # plt.vlines(x=yy0,ymin=xx0,ymax=xx0+0.5,colors='r',zorder=1)
    #     plt.vlines(x=yy0+0.5,ymin=xx0,ymax=xx0+1,colors='r',zorder=1)

    cb_ax = fig.add_axes([0.91, 0.13, 0.03, 0.84])#[left,bottom,width,height]
    cbar = fig.colorbar(im, cax=cb_ax)
    cbar.set_label(colorbar_labels[i])
    if i==0 or i==3:
        cbar.set_ticks(np.arange(0, 1.1, 0.2))
    else:
        cbar.set_ticks(np.arange(min_v,max_v+0.1, interval))

    fig.subplots_adjust(bottom=0.11, top=0.99, left=0.08, right=0.9,wspace=0.3, hspace=0.1)
    # plt.savefig(graph_path+"nse_wd.eps",format="EPS",dpi=2000)
    plt.savefig(graph_path+name[i]+".tif",format="TIFF",dpi=1200)
    # plt.savefig(graph_path+name[i]+".eps",format="EPS",dpi=2000)

plt.show()
# %%
