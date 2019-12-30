#%%
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


import sys
sys.path.append(root_path+'/tools/')
from fit_line import compute_linear_fit,compute_list_linear_fit

# x=[
#     'MEF-YX-L1-T','MEF-YX-L1-D','MEF-YX-L3-T','MEF-YX-L3-D','MEF-YX-L5-T','MEF-YX-L5-D','MEF-YX-L7-T','MEF-YX-L7-D',
#     'MEF-ZJS-L1-T','MEF-ZJS-L1-D','MEF-ZJS-L3-T','MEF-ZJS-L3-D','MEF-ZJS-L5-T','MEF-ZJS-L5-D','MEF-ZJS-L7-T','MEF-ZJS-L7-D',
#     'SF-YX-L1-T','SF-YX-L1-D','SF-YX-L3-T','SF-YX-L3-D','SF-YX-L5-T','SF-YX-L5-D','SF-YX-L7-T','SF-YX-L7-D',
#     'SF-ZJS-L1-T','SF-ZJS-L1-D','SF-ZJS-L3-T','SF-ZJS-L3-D','SF-ZJS-L5-T','SF-ZJS-L5-D','SF-ZJS-L7-T','SF-ZJS-L7-D',
#     'SFMIS-YX-L1-T','SFMIS-YX-L1-D','SFMIS-YX-L3-T','SFMIS-YX-L3-D','SFMIS-YX-L5-T','SFMIS-YX-L5-D','SFMIS-YX-L7-T','SFMIS-YX-L7-D',
#     'SFMIS-ZJS-L1-T','SFMIS-ZJS-L1-D','SFMIS-ZJS-L3-T','SFMIS-ZJS-L3-D','SFMIS-ZJS-L5-T','SFMIS-ZJS-L5-D','SFMIS-ZJS-L7-T','SFMIS-ZJS-L7-D',
#     ]

# y=['db2-1','db2-2','db2-3',
# 'db5-1','db5-2','db5-3',
# 'db10-1','db10-2','db10-3',
# 'db15-1','db15-2','db15-3',
# 'db20-1','db20-2','db20-3',
# 'db25-1','db25-2','db25-3',
# 'db30-1','db30-2','db30-3',
# 'db35-1','db35-2','db35-3',
# 'db40-1','db40-2','db40-3',
# 'db45-1','db45-2','db45-3',
# 'bior 3.3-1','bior 3.3-2','bior 3.3-3',
# 'coif3-1','coif3-2','coif3-3',
# 'haar-1','haar-2','haar-3','EEMD','VMD','monoscale']

x=[
    'MEF-YX-L1','MEF-YX-L3','MEF-YX-L5','MEF-YX-L7',
    'MEF-ZJS-L1','MEF-ZJS-L3','MEF-ZJS-L5','MEF-ZJS-L7',
    'SF-YX-L1','SF-YX-L3','SF-YX-L5','SF-YX-L7',
    'SF-ZJS-L1','SF-ZJS-L3','SF-ZJS-L5','SF-ZJS-L7',
    'SFMIS-YX-L1','SFMIS-YX-L3','SFMIS-YX-L5','SFMIS-YX-L7',
    'SFMIS-ZJS-L1','SFMIS-ZJS-L3','SFMIS-ZJS-L5','SFMIS-ZJS-L7',
    ]

y=['db45-3','EEMD','VMD','monoscale']

pattern_dict={
    'MEF':['multi_models_','forecast_pacf'],
    'SF':['one_model_','forecast_pacf'],
    'SFMIS':['one_model_','forecast_pacf_mis'],
    'MEH':['multi_models_','hindcast_pacf'],
    'SH':['one_model_','hindcast_pacf'],
    'SHMIS':['one_model_','hindcast_pacf_mis'],
}

preds_dict={}
records_dict={}

for station_lead in x:
    info_ = station_lead.split('-')
    pattern=info_[0]
    station=info_[1]
    lead=(info_[2].split('L'))[1]
    print('Pattern:{}'.format(pattern))
    print('Station:{}'.format(station))
    print('leading time:{}'.format(lead))
    
    for decomposer in y:
        if decomposer=='monoscale':
            data = pd.read_csv(root_path+'/'+station.lower()+'_orig/projects/lstm-models-history/'+lead+'_ahead/model_test_results.csv')
        elif decomposer=='EEMD' or decomposer=='VMD':
            data = pd.read_csv(root_path+'/'+station.lower()+'_'+decomposer.lower()+'/projects/lstm-models-history/'+pattern_dict[pattern][0]+lead+'_ahead_'+pattern_dict[pattern][1]+'/model_test_results.csv')
        else:
            data = pd.read_csv(root_path+'/'+station.lower()+'_wd/projects/lstm-models-history/'+decomposer.lower()+'/'+pattern_dict[pattern][0]+lead+'_ahead_'+pattern_dict[pattern][1]+'/model_test_results.csv')

        records=data['test_y']
        preds=data['test_pred']
        preds_dict[station+'-'+lead+'-'+pattern+'-'+decomposer]=preds.values
        records_dict[station+'-'+lead+'-'+pattern+'-'+decomposer]=records.values

    

#%%
pred_df = pd.DataFrame(preds_dict)
record_df = pd.DataFrame(records_dict)


#%%

t = list(range(1,658))
zorder_dict={
    'ZJS-1-SF-VMD':1,
    'ZJS-1-SF-db45-3':2,
    'ZJS-1-SF-EEMD':3,
    'ZJS-1-SF-monoscale':4,
    'ZJS-1-MEF-db45-3':5,
    'ZJS-1-MEF-VMD':6,
    'ZJS-1-MEF-EEMD':7,
    'ZJS-1-SFMIS-db45-3':8,
    'ZJS-1-SFMIS-EEMD':9,
    'ZJS-1-SFMIS-VMD':10,
    'ZJS-3-SF-VMD':1,
    'ZJS-3-SF-db45-3':2,
    'ZJS-3-SF-EEMD':3,
    'ZJS-3-SF-monoscale':4,
    'ZJS-3-MEF-db45-3':5,
    'ZJS-3-MEF-VMD':6,
    'ZJS-3-MEF-EEMD':7,
    'ZJS-3-SFMIS-db45-3':8,
    'ZJS-3-SFMIS-EEMD':9,
    'ZJS-3-SFMIS-VMD':10,
    'ZJS-5-SF-VMD':1,
    'ZJS-5-SF-db45-3':2,
    'ZJS-5-SF-EEMD':3,
    'ZJS-5-SF-monoscale':4,
    'ZJS-5-MEF-db45-3':5,
    'ZJS-5-MEF-VMD':6,
    'ZJS-5-MEF-EEMD':7,
    'ZJS-5-SFMIS-db45-3':8,
    'ZJS-5-SFMIS-EEMD':9,
    'ZJS-5-SFMIS-VMD':10,
    'ZJS-7-SF-VMD':1,
    'ZJS-7-SF-db45-3':2,
    'ZJS-7-SF-EEMD':3,
    'ZJS-7-SF-monoscale':4,
    'ZJS-7-MEF-db45-3':5,
    'ZJS-7-MEF-VMD':6,
    'ZJS-7-MEF-EEMD':7,
    'ZJS-7-SFMIS-db45-3':8,
    'ZJS-7-SFMIS-EEMD':9,
    'ZJS-7-SFMIS-VMD':10,
    'YX-1-SF-VMD':1,
    'YX-1-SF-db45-3':2,
    'YX-1-SF-EEMD':3,
    'YX-1-SF-monoscale':4,
    'YX-1-MEF-db45-3':5,
    'YX-1-MEF-VMD':6,
    'YX-1-MEF-EEMD':7,
    'YX-1-SFMIS-db45-3':8,
    'YX-1-SFMIS-EEMD':9,
    'YX-1-SFMIS-VMD':10,
    'YX-3-SF-VMD':1,
    'YX-3-SF-db45-3':2,
    'YX-3-SF-EEMD':3,
    'YX-3-SF-monoscale':4,
    'YX-3-MEF-db45-3':5,
    'YX-3-MEF-VMD':6,
    'YX-3-MEF-EEMD':7,
    'YX-3-SFMIS-db45-3':8,
    'YX-3-SFMIS-EEMD':9,
    'YX-3-SFMIS-VMD':10,
    'YX-5-SF-VMD':1,
    'YX-5-SF-db45-3':2,
    'YX-5-SF-EEMD':3,
    'YX-5-SF-monoscale':4,
    'YX-5-MEF-db45-3':5,
    'YX-5-MEF-VMD':6,
    'YX-5-MEF-EEMD':7,
    'YX-5-SFMIS-db45-3':8,
    'YX-5-SFMIS-EEMD':9,
    'YX-5-SFMIS-VMD':10,
    'YX-7-SF-VMD':1,
    'YX-7-SF-db45-3':2,
    'YX-7-SF-EEMD':3,
    'YX-7-SF-monoscale':4,
    'YX-7-MEF-db45-3':5,
    'YX-7-MEF-VMD':6,
    'YX-7-MEF-EEMD':7,
    'YX-7-SFMIS-db45-3':8,
    'YX-7-SFMIS-EEMD':9,
    'YX-7-SFMIS-VMD':10,

}
lw_dict={
    'SF':1.5,
    'MEF':1.0,
    'SFMIS':0.5,
}
lw_dict1={
    'SF-VMD':2.5,
    'SF-db45-3':1.5,
    'SF-monoscale':1,
    'SF-EEMD':0.5,
}
labels_dict={
    'MEF-monoscale':'LSTM',
    'SF-monoscale':'LSTM',
    'SFMIS-monoscale':'LSTM',
    'MEF-VMD':'MEF-VMD-LSTM',
    'SF-VMD':'SF-VMD-LSTM',
    'SFMIS-VMD':'SFMIS-VMD-LSTM',
    'MEF-EEMD':'MEF-EEMD-LSTM',
    'SF-EEMD':'SF-EEMD-LSTM',
    'SFMIS-EEMD':'SFMIS-EEMD-LSTM',
    'MEF-db45-3':'MEF-DWT-LSTM(db45-3)',
    'SF-db45-3':'SF-DWT-LSTM(db45-3)',
    'SFMIS-db45-3':'SFMIS-DWT-LSTM(db45-3)',
}
zorders_dict={
    'MEF-monoscale':4,
    'SF-monoscale':4,
    'SFMIS-monoscale':4,
    'MEF-VMD':7,
    'SF-VMD':12,
    'SFMIS-VMD':6,
    'MEF-EEMD':7,
    'SF-EEMD':5,
    'SFMIS-EEMD':8,
    'MEF-db45-3':11,
    'SF-db45-3':10,
    'SFMIS-db45-3':9,
}

colors_dict={
    'SF-monoscale':'tab:red',
    # 'SFMIS-monoscale':4,
    # 'MEF-VMD':7,
    'SF-VMD':'tab:blue',
    # 'SFMIS-VMD':6,
    # 'MEF-EEMD':7,
    'SF-EEMD':'tab:orange',
    # 'SFMIS-EEMD':8,
    # 'MEF-db45-3':11,
    'SF-db45-3':'tab:green',
    # 'SFMIS-db45-3':9,
}
xxx={
    'ZJS':{1:685,3:660,5:635,7:680},
    'YX':{1:2750,3:3050,5:2750,7:2750},
}
yyy={
    'ZJS':{1:25,3:25,5:25,7:25},
    'YX':{1:100,3:100,5:100,7:100},
}
for station in ['ZJS','YX']:
    fig= plt.figure(figsize=(7.48,7.48))
    ax1 = plt.subplot2grid((2,2), (0,0),aspect='equal')
    ax2 = plt.subplot2grid((2,2), (0,1),aspect='equal')
    ax3 = plt.subplot2grid((2,2), (1,0),aspect='equal')
    ax4 = plt.subplot2grid((2,2), (1,1),aspect='equal')
    axes={1:ax1,3:ax2,5:ax3,7:ax4}
    fig_id={1:'(a)',3:'(b)',5:'(c)',7:'(d)'}
    
    
    for lead in [1,3,5,7]:
        records_list=[]
        predictions_list=[]
        labels=[]
        zorders=[]
        colors=[]
        for pattern in ['SF']:
            for decomposer in y:
                records=record_df[station+'-'+str(lead)+'-'+pattern+'-'+decomposer]
                predictions=pred_df[station+'-'+str(lead)+'-'+pattern+'-'+decomposer]
                if pattern=='SF'and decomposer=='monoscale':
                    records_list.append(records)
                    predictions_list.append(predictions)
                    labels.append(labels_dict[pattern+'-'+decomposer])
                    zorders.append(zorders_dict[pattern+'-'+decomposer])
                    colors.append(colors_dict[pattern+'-'+decomposer])
                elif decomposer=='EEMD' or decomposer=='VMD':
                    records_list.append(records)
                    predictions_list.append(predictions)
                    labels.append(labels_dict[pattern+'-'+decomposer])
                    zorders.append(zorders_dict[pattern+'-'+decomposer])
                    colors.append(colors_dict[pattern+'-'+decomposer])
                elif decomposer=='db45-3':
                    records_list.append(records)
                    predictions_list.append(predictions)
                    labels.append(labels_dict[pattern+'-'+decomposer])
                    zorders.append(zorders_dict[pattern+'-'+decomposer])
                    colors.append(colors_dict[pattern+'-'+decomposer])
        
        print(records_list)
        xx,linear_list,xymin,xymax=compute_list_linear_fit(
            records_list=records_list,
            predictions_list=predictions_list,
        )
        axes[lead].text(xxx[station][lead],yyy[station][lead],fig_id[lead]+str(lead)+'-day ahead')
        axes[lead].set_ylabel(r'records($m^3/s$)')
        axes[lead].set_xlabel(r'predictands($m^3/s$)')
        axes[lead].plot([xymin,xymax], [xymin,xymax], '-', color='black', label='Ideal fit',linewidth=1.0)
        axes[lead].set_xlim([xymin,xymax])
        axes[lead].set_ylim([xymin,xymax])
        markers=['s','v','o','*']
        for i in range(len(records_list)):
            axes[lead].plot(xx, linear_list[i], '--',color=colors[i], label=labels[i],linewidth=1.0,zorder=zorders[i])
            axes[lead].plot(
                predictions_list[i],
                records_list[i],
                markers[i],
                markerfacecolor=colors[i],
                markeredgecolor='slategray',
                markersize=5.0,
                label=labels[i],
                zorder=zorders[i])
        
        axes[1].legend(loc='upper left',
            bbox_to_anchor=(0.25,1.12),
            ncol=5,
            shadow=False,
            frameon=True,)


    fig.subplots_adjust(bottom=0.06, top=0.95, left=0.07, right=0.99,wspace=0.15, hspace=0.15)
    # plt.savefig(graph_path+"nse_wd.eps",format="EPS",dpi=2000)
    plt.savefig(graph_path+station+"_scatters.tif",format="TIFF",dpi=1200)
    

plt.show()









# %%
