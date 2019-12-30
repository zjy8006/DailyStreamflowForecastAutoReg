import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
root_path = os.path.dirname(os.path.abspath('__file__'))
# root_path = os.path.abspath(os.path.join(root_path,os.path.pardir))
graph_path = root_path+'/graph/'
print("root path:{}".format(root_path))
# plt.rcParams['figure.figsize']=(10,8)
plt.rcParams['font.size']=6
# plt.rcParams["figure.figsize"] = [7.48, 5.61]
# plt.rcParams['image.cmap']='plasma'
plt.rcParams['image.cmap']='viridis'
# plt.rcParams['axes.linewidth']=0.8


yx_vmd_1 = pd.read_csv(root_path+"/yx_vmd/data/one_model_1_ahead_forecast_pacf/train_samples.csv")
yx_vmd_3 = pd.read_csv(root_path+"/yx_vmd/data/one_model_1_ahead_forecast_pacf/train_samples.csv")
yx_vmd_5 = pd.read_csv(root_path+"/yx_vmd/data/one_model_1_ahead_forecast_pacf/train_samples.csv")
yx_vmd_7 = pd.read_csv(root_path+"/yx_vmd/data/one_model_1_ahead_forecast_pacf/train_samples.csv")

yx_eemd_1 = pd.read_csv(root_path+"/yx_eemd/data/one_model_1_ahead_forecast_pacf/train_samples.csv")
yx_eemd_3 = pd.read_csv(root_path+"/yx_eemd/data/one_model_1_ahead_forecast_pacf/train_samples.csv")
yx_eemd_5 = pd.read_csv(root_path+"/yx_eemd/data/one_model_1_ahead_forecast_pacf/train_samples.csv")
yx_eemd_7 = pd.read_csv(root_path+"/yx_eemd/data/one_model_1_ahead_forecast_pacf/train_samples.csv")

yx_dwt_1 = pd.read_csv(root_path+"/yx_wd/data/db45-3/one_model_1_ahead_forecast_pacf/train_samples.csv")
yx_dwt_3 = pd.read_csv(root_path+"/yx_wd/data/db45-3/one_model_1_ahead_forecast_pacf/train_samples.csv")
yx_dwt_5 = pd.read_csv(root_path+"/yx_wd/data/db45-3/one_model_1_ahead_forecast_pacf/train_samples.csv")
yx_dwt_7 = pd.read_csv(root_path+"/yx_wd/data/db45-3/one_model_1_ahead_forecast_pacf/train_samples.csv")

zjs_vmd_1 = pd.read_csv(root_path+"/zjs_vmd/data/one_model_1_ahead_forecast_pacf/train_samples.csv")
zjs_vmd_3 = pd.read_csv(root_path+"/zjs_vmd/data/one_model_1_ahead_forecast_pacf/train_samples.csv")
zjs_vmd_5 = pd.read_csv(root_path+"/zjs_vmd/data/one_model_1_ahead_forecast_pacf/train_samples.csv")
zjs_vmd_7 = pd.read_csv(root_path+"/zjs_vmd/data/one_model_1_ahead_forecast_pacf/train_samples.csv")
zjs_eemd_1 = pd.read_csv(root_path+"/zjs_eemd/data/one_model_1_ahead_forecast_pacf/train_samples.csv")
zjs_eemd_3 = pd.read_csv(root_path+"/zjs_eemd/data/one_model_1_ahead_forecast_pacf/train_samples.csv")
zjs_eemd_5 = pd.read_csv(root_path+"/zjs_eemd/data/one_model_1_ahead_forecast_pacf/train_samples.csv")
zjs_eemd_7 = pd.read_csv(root_path+"/zjs_eemd/data/one_model_1_ahead_forecast_pacf/train_samples.csv")
zjs_dwt_1 = pd.read_csv(root_path+"/zjs_wd/data/db45-3/one_model_1_ahead_forecast_pacf/train_samples.csv")
zjs_dwt_3 = pd.read_csv(root_path+"/zjs_wd/data/db45-3/one_model_1_ahead_forecast_pacf/train_samples.csv")
zjs_dwt_5 = pd.read_csv(root_path+"/zjs_wd/data/db45-3/one_model_1_ahead_forecast_pacf/train_samples.csv")
zjs_dwt_7 = pd.read_csv(root_path+"/zjs_wd/data/db45-3/one_model_1_ahead_forecast_pacf/train_samples.csv")

data=[
    [zjs_vmd_1, zjs_eemd_1,zjs_dwt_1, yx_vmd_1, yx_eemd_1, yx_dwt_1],
    [zjs_vmd_3, zjs_eemd_3,zjs_dwt_3, yx_vmd_3, yx_eemd_3, yx_dwt_3],
    [zjs_vmd_5, zjs_eemd_5,zjs_dwt_5, yx_vmd_5, yx_eemd_5, yx_dwt_5],
    [zjs_vmd_7, zjs_eemd_7,zjs_dwt_7, yx_vmd_7, yx_eemd_7, yx_dwt_7],
   
    ]
labels=[
    [
        'SF-VMD-LSTM at ZJS',
        'SF-EEMD-LSTM at ZJS',
        'SF-DWT-LSTM(db45-3) at ZJS',
        'SF-VMD-LSTM at YX',
        'SF-EEMD-LSTM at YX',
        'SF-DWT-LSTM(db45-3) at YX',
        ],
    
    [
        '3-day ahead of SF-VMD-LSTM at ZJS',
        '3-day ahead of SF-EEMD-LSTM at ZJS',
        '3-day ahead of SF-DWT-LSTM(db45-3) at ZJS',
        '3-day ahead of SF-VMD-LSTM at YX',
        '3-day ahead of SF-EEMD-LSTM at YX',
        '3-day ahead of SF-DWT-LSTM(db45-3) at YX',
        ],

    [
        '5-day ahead of SF-VMD-LSTM at ZJS',
        '5-day ahead of SF-EEMD-LSTM at ZJS',
        '5-day ahead of SF-DWT-LSTM(db45-3) at ZJS',
        '5-day ahead of SF-VMD-LSTM at YX',
        '5-day ahead of SF-EEMD-LSTM at YX',
        '5-day ahead of SF-DWT-LSTM(db45-3) at YX',
        ],

    [
        '7-day ahead of SF-VMD-LSTM at ZJS',
        '7-day ahead of SF-EEMD-LSTM at ZJS',
        '7-day ahead of SF-DWT-LSTM(db45-3) at ZJS',
        '7-day ahead of SF-VMD-LSTM at YX',
        '7-day ahead of SF-EEMD-LSTM at YX',
        '7-day ahead of SF-DWT-LSTM(db45-3) at YX',
        ],
    
  
]

x = [i*0.1 for i in range(1,10)]
fig=plt.figure(figsize=(5.51,5.))
for i in range(len(data)):
    plt.subplot(2,2,i+1)
    if i==3 or i==2:
        plt.xlabel('Count')
    plt.ylabel('Pearson Coefficient')
    makers=['o','s','*','v','D','+']
    for j in range(len(data[i])):
        corr=abs(data[i][j].corr(method="pearson")['Y'][0:data[i][j].shape[1]-1]).sort_values(ascending=False)
        corr=corr[0:32]
        print(corr)
        corr0_1=corr[corr<=0.1].count()
        corr0_2=corr[corr<=0.2].count()-corr0_1
        corr0_3=corr[corr<=0.3].count()-(corr0_2+corr0_1)
        corr0_4=corr[corr<=0.4].count()-(corr0_3+corr0_2+corr0_1)
        corr0_5=corr[corr<=0.5].count()-(corr0_4+corr0_3+corr0_2+corr0_1)
        corr0_6=corr[corr<=0.6].count()-(corr0_5+corr0_4+corr0_3+corr0_2+corr0_1)
        corr0_7=corr[corr<=0.7].count()-(corr0_6+corr0_5+corr0_4+corr0_3+corr0_2+corr0_1)
        corr0_8=corr[corr<=0.8].count()-(corr0_7+corr0_6+corr0_5+corr0_4+corr0_3+corr0_2+corr0_1)
        corr0_9=corr[corr<=0.9].count()-(corr0_8+corr0_7+corr0_6+corr0_5+corr0_4+corr0_3+corr0_2+corr0_1)
        y=[corr0_1,corr0_2,corr0_3,corr0_4,corr0_5,corr0_6,corr0_7,corr0_8,corr0_9]
        print(sum(y))
        plt.plot(x,y,marker=makers[j],label=labels[i][j])
    if i==0:
        plt.legend(loc='upper left',
                bbox_to_anchor=(0.1,1.18),
                ncol=3,
                shadow=False,
                frameon=True,)
plt.subplots_adjust(bottom=0.08, top=0.94, left=0.1, right=0.99,wspace=0.3, hspace=0.1)
# # plt.savefig(graph_path+"nse_wd.eps",format="EPS",dpi=2000)
# plt.savefig(graph_path+"Pearson_corr_subsignals_imshow.tif",format="TIFF",dpi=1200)
plt.show()
