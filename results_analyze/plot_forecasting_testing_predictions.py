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
yx_path=root_path+'/yx_wd/projects/lstm-models-history/'
zjs_path=root_path+'/zjs_wd/projects/lstm-models-history/'

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
    # 'MEF-YX-L1','MEF-YX-L3','MEF-YX-L5','MEF-YX-L7',
    # 'MEF-ZJS-L1','MEF-ZJS-L3','MEF-ZJS-L5','MEF-ZJS-L7',
    'SF-YX-L1','SF-YX-L3','SF-YX-L5','SF-YX-L7',
    'SF-ZJS-L1','SF-ZJS-L3','SF-ZJS-L5','SF-ZJS-L7',
    # 'SFMIS-YX-L1','SFMIS-YX-L3','SFMIS-YX-L5','SFMIS-YX-L7',
    # 'SFMIS-ZJS-L1','SFMIS-ZJS-L3','SFMIS-ZJS-L5','SFMIS-ZJS-L7',
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
yy={
    'ZJS':{1:800,3:800,5:780,7:800},
    'YX':{1:3400,3:3800,5:3400,7:3400},
}
t = list(range(1,658))
zorders_dict={
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
zorders_dict={
    # 'MEF-monoscale':4,
    'SF-monoscale':4,
    # 'SFMIS-monoscale':4,
    # 'MEF-VMD':7,
    'SF-VMD':1,
    # 'SFMIS-VMD':6,
    # 'MEF-EEMD':7,
    'SF-EEMD':3,
    # 'SFMIS-EEMD':8,
    # 'MEF-db45-3':11,
    'SF-db45-3':2,
    # 'SFMIS-db45-3':9,
}

lw_dict={
    'SF':1.5,
    'MEF':1.0,
    'SFMIS':0.5,
}
lw_dict1={
    'SF-VMD':2.0,
    'SF-db45-3':1,
    'SF-monoscale':1,
    'SF-EEMD':1,
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
for station in ['ZJS','YX']:
    fig, ax = plt.subplots(nrows=4, ncols=1, figsize=(7.48,7.48))
    axes={1:ax[0],3:ax[1],5:ax[2],7:ax[3]}
    fig_id={1:'(a)',3:'(b)',5:'(c)',7:'(d)'}
    for lead in [1,3,5,7]:
        axes[lead].plot(t,record_df[station+'-'+str(lead)+'-SF-VMD'],c='slategray',label='records',lw=6,zorder=0,alpha=0.5)
        axes[lead].set_ylabel(r'flow($m^3/s$)')
        axes[lead].text(0,yy[station][lead],fig_id[lead]+str(lead)+'-day ahead')
        if lead==7:
            axes[lead].set_xlabel('Time(2013/03/15-2014/12/31)')
        else:
            axes[lead].set_xticks([])
        for pattern in [
            # 'MEF',
            'SF',
            # 'SFMIS',
            ]:
            for decomposer in y:
                if pattern=='SF'and decomposer=='monoscale':
                    axes[lead].plot(
                        t,
                        pred_df[station+'-'+str(lead)+'-'+pattern+'-'+decomposer],
                        '--',
                        label='LSTM',
                        color=colors_dict[pattern+'-'+decomposer],
                        lw=lw_dict1[pattern+'-'+decomposer],
                        zorder=zorders_dict[pattern+'-'+decomposer])
                elif decomposer=='EEMD':
                    axes[lead].plot(
                        t,
                        pred_df[station+'-'+str(lead)+'-'+pattern+'-'+decomposer],
                        '--',
                        label=pattern+'-'+decomposer.upper()+'-LSTM',
                        color=colors_dict[pattern+'-'+decomposer],
                        lw=lw_dict1[pattern+'-'+decomposer],
                        zorder=zorders_dict[pattern+'-'+decomposer])
                elif decomposer=='VMD':
                    axes[lead].plot(
                        t,
                        pred_df[station+'-'+str(lead)+'-'+pattern+'-'+decomposer],
                        # '--',
                        label=pattern+'-'+decomposer.upper()+'-LSTM',
                        color=colors_dict[pattern+'-'+decomposer],
                        lw=lw_dict1[pattern+'-'+decomposer],
                        zorder=zorders_dict[pattern+'-'+decomposer])
                elif decomposer=='db45-3':
                    axes[lead].plot(
                        t,
                        pred_df[station+'-'+str(lead)+'-'+pattern+'-'+decomposer],
                        '--',
                        label=pattern+'-DWT-LSTM('+decomposer+')',
                        color=colors_dict[pattern+'-'+decomposer],
                        lw=lw_dict1[pattern+'-'+decomposer],
                        zorder=zorders_dict[pattern+'-'+decomposer],
                        )
        axes[1].legend(loc='upper center',
            bbox_to_anchor=(0.5,1.15),
            ncol=5,
            shadow=False,
            frameon=True,)
    

    fig.subplots_adjust(bottom=0.06, top=0.97, left=0.08, right=0.98,wspace=0.05, hspace=0.05)
    # plt.savefig(graph_path+"nse_wd.eps",format="EPS",dpi=2000)
    plt.savefig(graph_path+station+"_prediction.tif",format="TIFF",dpi=1200)
    

plt.show()









# %%
