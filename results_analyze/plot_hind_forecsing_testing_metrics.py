#%%
import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['font.size'] = 6
root_path = os.path.dirname(os.path.abspath('__file__'))
# root_path = os.path.abspath(os.path.join(root_path,os.path.pardir))
graphs_path = root_path+'/graph/'
print("root path:{}".format(root_path))
sys.path.append(root_path+'/tools/')

def autolabel(rects, ax):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        height = round(height, 2)
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height/2),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')


def autolabels(rects, ax):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        height = round(height, 2)
        ax.text(
            x=rect.get_x() + rect.get_width() / 2,
            y=height,
            s='{}'.format(height),
            rotation=90,
            ha='center', va='bottom',
        )



#%%
x=[
    'MEF-YX-L1','MEF-YX-L1','MEF-YX-L3','MEF-YX-L3','MEF-YX-L5','MEF-YX-L5','MEF-YX-L7','MEF-YX-L7',
    'MEF-ZJS-L1','MEF-ZJS-L1','MEF-ZJS-L3','MEF-ZJS-L3','MEF-ZJS-L5','MEF-ZJS-L5','MEF-ZJS-L7','MEF-ZJS-L7',
    'SF-YX-L1','SF-YX-L1','SF-YX-L3','SF-YX-L3','SF-YX-L5','SF-YX-L5','SF-YX-L7','SF-YX-L7',
    'SF-ZJS-L1','SF-ZJS-L1','SF-ZJS-L3','SF-ZJS-L3','SF-ZJS-L5','SF-ZJS-L5','SF-ZJS-L7','SF-ZJS-L7',
    'SFMIS-YX-L1','SFMIS-YX-L1','SFMIS-YX-L3','SFMIS-YX-L3','SFMIS-YX-L5','SFMIS-YX-L5','SFMIS-YX-L7','SFMIS-YX-L7',
    'SFMIS-ZJS-L1','SFMIS-ZJS-L1','SFMIS-ZJS-L3','SFMIS-ZJS-L3','SFMIS-ZJS-L5','SFMIS-ZJS-L5','SFMIS-ZJS-L7','SFMIS-ZJS-L7',
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
    for station_lead in x:
        info_ = station_lead.split('-')
        pattern=info_[0]
        station=info_[1]
        lead=(info_[2].split('L'))[1]
        # print('Pattern:{}'.format(pattern))
        # print('Station:{}'.format(station))
        # print('leading time:{}'.format(lead))

        if decomposer=='monoscale':
            data = pd.read_csv(root_path+'/'+station.lower()+'_orig/projects/lstm-models-history/'+lead+'_ahead/model_metrics.csv')
        elif decomposer=='EEMD' or decomposer=='VMD':
            data = pd.read_csv(root_path+'/'+station.lower()+'_'+decomposer.lower()+'/projects/lstm-models-history/'+pattern_dict[pattern][0]+lead+'_ahead_'+pattern_dict[pattern][1]+'/model_metrics.csv')
        else:
            data = pd.read_csv(root_path+'/'+station.lower()+'_wd/projects/lstm-models-history/'+decomposer.lower()+'/'+pattern_dict[pattern][0]+lead+'_ahead_'+pattern_dict[pattern][1]+'/model_metrics.csv')

        NSE_S[station+'-'+decomposer+'-'+pattern+'-'+lead]=data['test_nse'][0]
        NRMSE_S[station+'-'+decomposer+'-'+pattern+'-'+lead]=data['test_nrmse'][0]
        PPTS_S[station+'-'+decomposer+'-'+pattern+'-'+lead]=data['test_ppts'][0]
        

nse_df = pd.DataFrame(NSE_S,index=[0])
print(nse_df)

#%%
ZJS_NSE_1=[NSE_S['ZJS-monoscale-SF-1'],NSE_S['ZJS-EEMD-SF-1'],NSE_S['ZJS-db45-3-SF-1'],NSE_S['ZJS-VMD-SF-1'],]
ZJS_NSE_3=[NSE_S['ZJS-monoscale-SF-3'],NSE_S['ZJS-EEMD-SF-3'],NSE_S['ZJS-db45-3-SF-3'],NSE_S['ZJS-VMD-SF-3'],]
ZJS_NSE_5=[NSE_S['ZJS-monoscale-SF-5'],NSE_S['ZJS-EEMD-SF-5'],NSE_S['ZJS-db45-3-SF-5'],NSE_S['ZJS-VMD-SF-5'],]
ZJS_NSE_7=[NSE_S['ZJS-monoscale-SF-7'],NSE_S['ZJS-EEMD-SF-7'],NSE_S['ZJS-db45-3-SF-7'],NSE_S['ZJS-VMD-SF-7'],]

ZJS_NRMSE_1=[NRMSE_S['ZJS-monoscale-SF-1'],NRMSE_S['ZJS-EEMD-SF-1'],NRMSE_S['ZJS-db45-3-SF-1'],NRMSE_S['ZJS-VMD-SF-1'],]
ZJS_NRMSE_3=[NRMSE_S['ZJS-monoscale-SF-3'],NRMSE_S['ZJS-EEMD-SF-3'],NRMSE_S['ZJS-db45-3-SF-3'],NRMSE_S['ZJS-VMD-SF-3'],]
ZJS_NRMSE_5=[NRMSE_S['ZJS-monoscale-SF-5'],NRMSE_S['ZJS-EEMD-SF-5'],NRMSE_S['ZJS-db45-3-SF-5'],NRMSE_S['ZJS-VMD-SF-5'],]
ZJS_NRMSE_7=[NRMSE_S['ZJS-monoscale-SF-7'],NRMSE_S['ZJS-EEMD-SF-7'],NRMSE_S['ZJS-db45-3-SF-7'],NRMSE_S['ZJS-VMD-SF-7'],]

ZJS_PPTS_1=[PPTS_S['ZJS-monoscale-SF-1'],PPTS_S['ZJS-EEMD-SF-1'],PPTS_S['ZJS-db45-3-SF-1'],PPTS_S['ZJS-VMD-SF-1'],]
ZJS_PPTS_3=[PPTS_S['ZJS-monoscale-SF-3'],PPTS_S['ZJS-EEMD-SF-3'],PPTS_S['ZJS-db45-3-SF-3'],PPTS_S['ZJS-VMD-SF-3'],]
ZJS_PPTS_5=[PPTS_S['ZJS-monoscale-SF-5'],PPTS_S['ZJS-EEMD-SF-5'],PPTS_S['ZJS-db45-3-SF-5'],PPTS_S['ZJS-VMD-SF-5'],]
ZJS_PPTS_7=[PPTS_S['ZJS-monoscale-SF-7'],PPTS_S['ZJS-EEMD-SF-7'],PPTS_S['ZJS-db45-3-SF-7'],PPTS_S['ZJS-VMD-SF-7'],]

YX_NSE_1=[NSE_S['YX-monoscale-SF-1'],NSE_S['YX-EEMD-SF-1'],NSE_S['YX-db45-3-SF-1'],NSE_S['YX-VMD-SF-1'],]
YX_NSE_3=[NSE_S['YX-monoscale-SF-3'],NSE_S['YX-EEMD-SF-3'],NSE_S['YX-db45-3-SF-3'],NSE_S['YX-VMD-SF-3'],]
YX_NSE_5=[NSE_S['YX-monoscale-SF-5'],NSE_S['YX-EEMD-SF-5'],NSE_S['YX-db45-3-SF-5'],NSE_S['YX-VMD-SF-5'],]
YX_NSE_7=[NSE_S['YX-monoscale-SF-7'],NSE_S['YX-EEMD-SF-7'],NSE_S['YX-db45-3-SF-7'],NSE_S['YX-VMD-SF-7'],]

YX_NRMSE_1=[NRMSE_S['YX-monoscale-SF-1'],NRMSE_S['YX-EEMD-SF-1'],NRMSE_S['YX-db45-3-SF-1'],NRMSE_S['YX-VMD-SF-1'],]
YX_NRMSE_3=[NRMSE_S['YX-monoscale-SF-3'],NRMSE_S['YX-EEMD-SF-3'],NRMSE_S['YX-db45-3-SF-3'],NRMSE_S['YX-VMD-SF-3'],]
YX_NRMSE_5=[NRMSE_S['YX-monoscale-SF-5'],NRMSE_S['YX-EEMD-SF-5'],NRMSE_S['YX-db45-3-SF-5'],NRMSE_S['YX-VMD-SF-5'],]
YX_NRMSE_7=[NRMSE_S['YX-monoscale-SF-7'],NRMSE_S['YX-EEMD-SF-7'],NRMSE_S['YX-db45-3-SF-7'],NRMSE_S['YX-VMD-SF-7'],]

YX_PPTS_1=[PPTS_S['YX-monoscale-SF-1'],PPTS_S['YX-EEMD-SF-1'],PPTS_S['YX-db45-3-SF-1'],PPTS_S['YX-VMD-SF-1'],]
YX_PPTS_3=[PPTS_S['YX-monoscale-SF-3'],PPTS_S['YX-EEMD-SF-3'],PPTS_S['YX-db45-3-SF-3'],PPTS_S['YX-VMD-SF-3'],]
YX_PPTS_5=[PPTS_S['YX-monoscale-SF-5'],PPTS_S['YX-EEMD-SF-5'],PPTS_S['YX-db45-3-SF-5'],PPTS_S['YX-VMD-SF-5'],]
YX_PPTS_7=[PPTS_S['YX-monoscale-SF-7'],PPTS_S['YX-EEMD-SF-7'],PPTS_S['YX-db45-3-SF-7'],PPTS_S['YX-VMD-SF-7'],]

metrics_lists = [
    [ZJS_NSE_1, ZJS_NSE_3, ZJS_NSE_5,ZJS_NSE_7],
    [YX_NSE_1, YX_NSE_3, YX_NSE_5,YX_NSE_7],
    [ZJS_NRMSE_1, ZJS_NRMSE_3, ZJS_NRMSE_5,ZJS_NRMSE_7],
    [YX_NRMSE_1, YX_NRMSE_3, YX_NRMSE_5,YX_NRMSE_7],
    [ZJS_PPTS_1, ZJS_PPTS_3, ZJS_PPTS_5,ZJS_PPTS_7],
    [YX_PPTS_1, YX_PPTS_3, YX_PPTS_5,YX_PPTS_7],
]

#%%
stations = ['1-day ahead', '3-day ahead', '5-day ahead','7-day ahead']
pos = [2, 5, 8, 11,]
print(pos)
width = 0.5
action = [-2, -1,0, 1,]
ylims = [
    [0, 1.2],
    [0, 1.7],
    [0, 3.3],
    [0, 570],
    [0, 90],
    [0, 360],
]
labels = ['LSTM', 'SF-EEMD-LSTM', 'SF-DWT-LSTM(db45-3)', 'SF-VMD-LSTM',]
y_labels = [r"$NSE$",r"$NSE$", r"$NRMSE(10^8m^3)$",r"$NRMSE(10^8m^3)$", r"$PPTS(5)(\%)$",r"$PPTS(5)(\%)$", ]
density = 5
hatch_str = ['/'*density, 'x'*density, '|'*density,'+'*density]

xx=[0.35,0.35,0.35,0.35,0.35,0.35]
yy=[0.95,0.95,2.25,1.68,70,67]
text=['(a)','(d)','(b)','(e)','(c)','(f)']
fig = plt.figure(figsize=(7.48, 7.48))
for i in range(len(metrics_lists)):
    ax = fig.add_subplot(3, 2, i+1)
    print(len(metrics_lists[i]))
    ax.text(xx[i],yy[i],text[i])
    for j in range(len(metrics_lists[i])):
        bars = ax.bar([p+action[j]*width for p in pos],
                      metrics_lists[i][j], width, alpha=0.5, label=stations[j])
        for bar in bars:
            bar.set_hatch(hatch_str[j])
            bar.set_edgecolor('k')
        # autolabels(bars,ax)
    # ax.set_ylim(ylims[i])
    ax.set_ylabel(y_labels[i])
    ax.set_xticks([p-width/2 for p in pos])
    ax.set_xticklabels(labels, rotation=45)
    if i == 0:
        ax.legend(
            loc='upper left',
            # bbox_to_anchor=(0.08,1.01, 1,0.101),
            bbox_to_anchor=(0.6, 1.15),
            # bbox_transform=plt.gcf().transFigure,
            ncol=4,
            shadow=False,
            frameon=True,
        )
plt.subplots_adjust(left=0.08, bottom=0.12, right=0.98,
                    top=0.97, hspace=0.6, wspace=0.25)
plt.savefig(graphs_path+'sf_testing_metrics.eps', format='EPS', dpi=2000)
plt.savefig(graphs_path+'sf_testing_metrics.tif', format='TIFF', dpi=1200)
plt.show()

# %%
