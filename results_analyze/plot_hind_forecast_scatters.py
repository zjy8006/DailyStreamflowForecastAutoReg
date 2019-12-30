#%%
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
plt.rcParams['font.size']=6
import pandas as pd
import numpy as np
import os
root_path = os.path.dirname(os.path.abspath('__file__'))
# root_path = os.path.abspath(os.path.join(root_path,os.path.pardir))
graphs_path = root_path+'/graph/'
print(root_path)
import sys
sys.path.append(root_path+'/tools/')
from fit_line import compute_linear_fit,compute_list_linear_fit


#%%
ZJS_MEF_VMD_HIND1 = pd.read_csv(root_path+'/zjs_vmd/projects/lstm-models-history/multi_models_1_ahead_hindcast_pacf/model_test_results.csv')
ZJS_MEF_VMD_FORE1 = pd.read_csv(root_path+'/zjs_vmd/projects/lstm-models-history/multi_models_1_ahead_forecast_pacf/model_test_results.csv')
ZJS_MEF_VMD_HIND3 = pd.read_csv(root_path+'/zjs_vmd/projects/lstm-models-history/multi_models_3_ahead_hindcast_pacf/model_test_results.csv')
ZJS_MEF_VMD_FORE3 = pd.read_csv(root_path+'/zjs_vmd/projects/lstm-models-history/multi_models_3_ahead_forecast_pacf/model_test_results.csv')
ZJS_MEF_VMD_HIND5 = pd.read_csv(root_path+'/zjs_vmd/projects/lstm-models-history/multi_models_5_ahead_hindcast_pacf/model_test_results.csv')
ZJS_MEF_VMD_FORE5 = pd.read_csv(root_path+'/zjs_vmd/projects/lstm-models-history/multi_models_5_ahead_forecast_pacf/model_test_results.csv')
ZJS_MEF_VMD_HIND7 = pd.read_csv(root_path+'/zjs_vmd/projects/lstm-models-history/multi_models_7_ahead_hindcast_pacf/model_test_results.csv')
ZJS_MEF_VMD_FORE7 = pd.read_csv(root_path+'/zjs_vmd/projects/lstm-models-history/multi_models_7_ahead_forecast_pacf/model_test_results.csv')

ZJS_MEF_EEMD_HIND1 = pd.read_csv(root_path+'/zjs_eemd/projects/lstm-models-history/multi_models_1_ahead_hindcast_pacf/model_test_results.csv')
ZJS_MEF_EEMD_FORE1 = pd.read_csv(root_path+'/zjs_eemd/projects/lstm-models-history/multi_models_1_ahead_forecast_pacf/model_test_results.csv')
ZJS_MEF_EEMD_HIND3 = pd.read_csv(root_path+'/zjs_eemd/projects/lstm-models-history/multi_models_3_ahead_hindcast_pacf/model_test_results.csv')
ZJS_MEF_EEMD_FORE3 = pd.read_csv(root_path+'/zjs_eemd/projects/lstm-models-history/multi_models_3_ahead_forecast_pacf/model_test_results.csv')
ZJS_MEF_EEMD_HIND5 = pd.read_csv(root_path+'/zjs_eemd/projects/lstm-models-history/multi_models_5_ahead_hindcast_pacf/model_test_results.csv')
ZJS_MEF_EEMD_FORE5 = pd.read_csv(root_path+'/zjs_eemd/projects/lstm-models-history/multi_models_5_ahead_forecast_pacf/model_test_results.csv')
ZJS_MEF_EEMD_HIND7 = pd.read_csv(root_path+'/zjs_eemd/projects/lstm-models-history/multi_models_7_ahead_hindcast_pacf/model_test_results.csv')
ZJS_MEF_EEMD_FORE7 = pd.read_csv(root_path+'/zjs_eemd/projects/lstm-models-history/multi_models_7_ahead_forecast_pacf/model_test_results.csv')

ZJS_MEF_DWT_HIND1 = pd.read_csv(root_path+'/zjs_wd/projects/lstm-models-history/db45-3/multi_models_1_ahead_hindcast_pacf/model_test_results.csv')
ZJS_MEF_DWT_FORE1 = pd.read_csv(root_path+'/zjs_wd/projects/lstm-models-history/db45-3/multi_models_1_ahead_forecast_pacf/model_test_results.csv')
ZJS_MEF_DWT_HIND3 = pd.read_csv(root_path+'/zjs_wd/projects/lstm-models-history/db45-3/multi_models_3_ahead_hindcast_pacf/model_test_results.csv')
ZJS_MEF_DWT_FORE3 = pd.read_csv(root_path+'/zjs_wd/projects/lstm-models-history/db45-3/multi_models_3_ahead_forecast_pacf/model_test_results.csv')
ZJS_MEF_DWT_HIND5 = pd.read_csv(root_path+'/zjs_wd/projects/lstm-models-history/db45-3/multi_models_5_ahead_hindcast_pacf/model_test_results.csv')
ZJS_MEF_DWT_FORE5 = pd.read_csv(root_path+'/zjs_wd/projects/lstm-models-history/db45-3/multi_models_5_ahead_forecast_pacf/model_test_results.csv')
ZJS_MEF_DWT_HIND7 = pd.read_csv(root_path+'/zjs_wd/projects/lstm-models-history/db45-3/multi_models_7_ahead_hindcast_pacf/model_test_results.csv')
ZJS_MEF_DWT_FORE7 = pd.read_csv(root_path+'/zjs_wd/projects/lstm-models-history/db45-3/multi_models_7_ahead_forecast_pacf/model_test_results.csv')


YX_MEF_VMD_HIND1 = pd.read_csv(root_path+'/yx_vmd/projects/lstm-models-history/multi_models_1_ahead_hindcast_pacf/model_test_results.csv')
YX_MEF_VMD_FORE1 = pd.read_csv(root_path+'/yx_vmd/projects/lstm-models-history/multi_models_1_ahead_forecast_pacf/model_test_results.csv')
YX_MEF_VMD_HIND3 = pd.read_csv(root_path+'/yx_vmd/projects/lstm-models-history/multi_models_3_ahead_hindcast_pacf/model_test_results.csv')
YX_MEF_VMD_FORE3 = pd.read_csv(root_path+'/yx_vmd/projects/lstm-models-history/multi_models_3_ahead_forecast_pacf/model_test_results.csv')
YX_MEF_VMD_HIND5 = pd.read_csv(root_path+'/yx_vmd/projects/lstm-models-history/multi_models_5_ahead_hindcast_pacf/model_test_results.csv')
YX_MEF_VMD_FORE5 = pd.read_csv(root_path+'/yx_vmd/projects/lstm-models-history/multi_models_5_ahead_forecast_pacf/model_test_results.csv')
YX_MEF_VMD_HIND7 = pd.read_csv(root_path+'/yx_vmd/projects/lstm-models-history/multi_models_7_ahead_hindcast_pacf/model_test_results.csv')
YX_MEF_VMD_FORE7 = pd.read_csv(root_path+'/yx_vmd/projects/lstm-models-history/multi_models_7_ahead_forecast_pacf/model_test_results.csv')
YX_MEF_EEMD_HIND1 = pd.read_csv(root_path+'/yx_eemd/projects/lstm-models-history/multi_models_1_ahead_hindcast_pacf/model_test_results.csv')
YX_MEF_EEMD_FORE1 = pd.read_csv(root_path+'/yx_eemd/projects/lstm-models-history/multi_models_1_ahead_forecast_pacf/model_test_results.csv')
YX_MEF_EEMD_HIND3 = pd.read_csv(root_path+'/yx_eemd/projects/lstm-models-history/multi_models_3_ahead_hindcast_pacf/model_test_results.csv')
YX_MEF_EEMD_FORE3 = pd.read_csv(root_path+'/yx_eemd/projects/lstm-models-history/multi_models_3_ahead_forecast_pacf/model_test_results.csv')
YX_MEF_EEMD_HIND5 = pd.read_csv(root_path+'/yx_eemd/projects/lstm-models-history/multi_models_5_ahead_hindcast_pacf/model_test_results.csv')
YX_MEF_EEMD_FORE5 = pd.read_csv(root_path+'/yx_eemd/projects/lstm-models-history/multi_models_5_ahead_forecast_pacf/model_test_results.csv')
YX_MEF_EEMD_HIND7 = pd.read_csv(root_path+'/yx_eemd/projects/lstm-models-history/multi_models_7_ahead_hindcast_pacf/model_test_results.csv')
YX_MEF_EEMD_FORE7 = pd.read_csv(root_path+'/yx_eemd/projects/lstm-models-history/multi_models_7_ahead_forecast_pacf/model_test_results.csv')
YX_MEF_DWT_HIND1 = pd.read_csv(root_path+'/yx_wd/projects/lstm-models-history/db45-3/multi_models_1_ahead_hindcast_pacf/model_test_results.csv')
YX_MEF_DWT_FORE1 = pd.read_csv(root_path+'/yx_wd/projects/lstm-models-history/db45-3/multi_models_1_ahead_forecast_pacf/model_test_results.csv')
YX_MEF_DWT_HIND3 = pd.read_csv(root_path+'/yx_wd/projects/lstm-models-history/db45-3/multi_models_3_ahead_hindcast_pacf/model_test_results.csv')
YX_MEF_DWT_FORE3 = pd.read_csv(root_path+'/yx_wd/projects/lstm-models-history/db45-3/multi_models_3_ahead_forecast_pacf/model_test_results.csv')
YX_MEF_DWT_HIND5 = pd.read_csv(root_path+'/yx_wd/projects/lstm-models-history/db45-3/multi_models_5_ahead_hindcast_pacf/model_test_results.csv')
YX_MEF_DWT_FORE5 = pd.read_csv(root_path+'/yx_wd/projects/lstm-models-history/db45-3/multi_models_5_ahead_forecast_pacf/model_test_results.csv')
YX_MEF_DWT_HIND7 = pd.read_csv(root_path+'/yx_wd/projects/lstm-models-history/db45-3/multi_models_7_ahead_hindcast_pacf/model_test_results.csv')
YX_MEF_DWT_FORE7 = pd.read_csv(root_path+'/yx_wd/projects/lstm-models-history/db45-3/multi_models_7_ahead_forecast_pacf/model_test_results.csv')

zjs_records_list=[
    [ZJS_MEF_VMD_FORE1['test_y'].values,ZJS_MEF_VMD_HIND1['test_y'].values],
    [ZJS_MEF_VMD_FORE3['test_y'].values,ZJS_MEF_VMD_HIND3['test_y'].values],
    [ZJS_MEF_VMD_FORE5['test_y'].values,ZJS_MEF_VMD_HIND5['test_y'].values],
    [ZJS_MEF_VMD_FORE7['test_y'].values,ZJS_MEF_VMD_HIND7['test_y'].values],
    [ZJS_MEF_EEMD_FORE1['test_y'].values,ZJS_MEF_EEMD_HIND1['test_y'].values],
    [ZJS_MEF_EEMD_FORE3['test_y'].values,ZJS_MEF_EEMD_HIND3['test_y'].values],
    [ZJS_MEF_EEMD_FORE5['test_y'].values,ZJS_MEF_EEMD_HIND5['test_y'].values],
    [ZJS_MEF_EEMD_FORE7['test_y'].values,ZJS_MEF_EEMD_HIND7['test_y'].values],
    [ZJS_MEF_DWT_FORE1['test_y'].values,ZJS_MEF_DWT_HIND1['test_y'].values],
    [ZJS_MEF_DWT_FORE3['test_y'].values,ZJS_MEF_DWT_HIND3['test_y'].values],
    [ZJS_MEF_DWT_FORE5['test_y'].values,ZJS_MEF_DWT_HIND5['test_y'].values],
    [ZJS_MEF_DWT_FORE7['test_y'].values,ZJS_MEF_DWT_HIND7['test_y'].values],    
]
zjs_predictions_list=[
    [ZJS_MEF_VMD_FORE1['test_pred'].values,ZJS_MEF_VMD_HIND1['test_pred'].values],
    [ZJS_MEF_VMD_FORE3['test_pred'].values,ZJS_MEF_VMD_HIND3['test_pred'].values],
    [ZJS_MEF_VMD_FORE5['test_pred'].values,ZJS_MEF_VMD_HIND5['test_pred'].values],
    [ZJS_MEF_VMD_FORE7['test_pred'].values,ZJS_MEF_VMD_HIND7['test_pred'].values],
    [ZJS_MEF_EEMD_FORE1['test_pred'].values,ZJS_MEF_EEMD_HIND1['test_pred'].values],
    [ZJS_MEF_EEMD_FORE3['test_pred'].values,ZJS_MEF_EEMD_HIND3['test_pred'].values],
    [ZJS_MEF_EEMD_FORE5['test_pred'].values,ZJS_MEF_EEMD_HIND5['test_pred'].values],
    [ZJS_MEF_EEMD_FORE7['test_pred'].values,ZJS_MEF_EEMD_HIND7['test_pred'].values],
    [ZJS_MEF_DWT_FORE1['test_pred'].values,ZJS_MEF_DWT_HIND1['test_pred'].values],
    [ZJS_MEF_DWT_FORE3['test_pred'].values,ZJS_MEF_DWT_HIND3['test_pred'].values],
    [ZJS_MEF_DWT_FORE5['test_pred'].values,ZJS_MEF_DWT_HIND5['test_pred'].values],
    [ZJS_MEF_DWT_FORE7['test_pred'].values,ZJS_MEF_DWT_HIND7['test_pred'].values],  
]

yx_records_list=[
    [YX_MEF_VMD_FORE1['test_y'].values,YX_MEF_VMD_HIND1['test_y'].values],
    [YX_MEF_VMD_FORE3['test_y'].values,YX_MEF_VMD_HIND3['test_y'].values],
    [YX_MEF_VMD_FORE5['test_y'].values,YX_MEF_VMD_HIND5['test_y'].values],
    [YX_MEF_VMD_FORE7['test_y'].values,YX_MEF_VMD_HIND7['test_y'].values],
    [YX_MEF_EEMD_FORE1['test_y'].values,YX_MEF_EEMD_HIND1['test_y'].values],
    [YX_MEF_EEMD_FORE3['test_y'].values,YX_MEF_EEMD_HIND3['test_y'].values],
    [YX_MEF_EEMD_FORE5['test_y'].values,YX_MEF_EEMD_HIND5['test_y'].values],
    [YX_MEF_EEMD_FORE7['test_y'].values,YX_MEF_EEMD_HIND7['test_y'].values],
    [YX_MEF_DWT_FORE1['test_y'].values,YX_MEF_DWT_HIND1['test_y'].values],
    [YX_MEF_DWT_FORE3['test_y'].values,YX_MEF_DWT_HIND3['test_y'].values],
    [YX_MEF_DWT_FORE5['test_y'].values,YX_MEF_DWT_HIND5['test_y'].values],
    [YX_MEF_DWT_FORE7['test_y'].values,YX_MEF_DWT_HIND7['test_y'].values],    
]
yx_predictions_list=[
    [YX_MEF_VMD_FORE1['test_pred'].values,YX_MEF_VMD_HIND1['test_pred'].values],
    [YX_MEF_VMD_FORE3['test_pred'].values,YX_MEF_VMD_HIND3['test_pred'].values],
    [YX_MEF_VMD_FORE5['test_pred'].values,YX_MEF_VMD_HIND5['test_pred'].values],
    [YX_MEF_VMD_FORE7['test_pred'].values,YX_MEF_VMD_HIND7['test_pred'].values],
    [YX_MEF_EEMD_FORE1['test_pred'].values,YX_MEF_EEMD_HIND1['test_pred'].values],
    [YX_MEF_EEMD_FORE3['test_pred'].values,YX_MEF_EEMD_HIND3['test_pred'].values],
    [YX_MEF_EEMD_FORE5['test_pred'].values,YX_MEF_EEMD_HIND5['test_pred'].values],
    [YX_MEF_EEMD_FORE7['test_pred'].values,YX_MEF_EEMD_HIND7['test_pred'].values],
    [YX_MEF_DWT_FORE1['test_pred'].values,YX_MEF_DWT_HIND1['test_pred'].values],
    [YX_MEF_DWT_FORE3['test_pred'].values,YX_MEF_DWT_HIND3['test_pred'].values],
    [YX_MEF_DWT_FORE5['test_pred'].values,YX_MEF_DWT_HIND5['test_pred'].values],
    [YX_MEF_DWT_FORE7['test_pred'].values,YX_MEF_DWT_HIND7['test_pred'].values],  
]


fig_id=[
    '(a1)','(a2)','(a3)','(a4)',
    '(b1)','(b2)','(b3)','(b4)',
    '(c1)','(c2)','(c3)','(c4)',
]
models_labels=[
    ['SF-VMD-LSTM(1-day ahead)','SH-VMD-LSTM(1-day ahead)',],
    ['SF-VMD-LSTM(3-day ahead)','SH-VMD-LSTM(3-day ahead)',],
    ['SF-VMD-LSTM(5-day ahead)','SH-VMD-LSTM(5-day ahead)',],
    ['SF-VMD-LSTM(7-day ahead)','SH-VMD-LSTM(7-day ahead)',],
    ['SF-EEMD-LSTM(1-day ahead)','SH-EEMD-LSTM(1-day ahead)',],
    ['SF-EEMD-LSTM(3-day ahead)','SH-EEMD-LSTM(3-day ahead)',],
    ['SF-EEMD-LSTM(5-day ahead)','SH-EEMD-LSTM(5-day ahead)',],
    ['SF-EEMD-LSTM(7-day ahead)','SH-EEMD-LSTM(7-day ahead)',],
    ['SF-DWT-LSTM(db45-3,1-day ahead)','SH-DWT-LSTM(db45-3,1-day ahead)',],
    ['SF-DWT-LSTM(db45-3,3-day ahead)','SH-DWT-LSTM(db45-3,3-day ahead)',],
    ['SF-DWT-LSTM(db45-3,5-day ahead)','SH-DWT-LSTM(db45-3,5-day ahead)',],
    ['SF-DWT-LSTM(db45-3,7-day ahead)','SH-DWT-LSTM(db45-3,7-day ahead)',], 
]
models_labels=[
    ['linear fit(MEF)','liner fit(MEH)',],
    ['linear fit(MEF)','liner fit(MEH)',],
    ['linear fit(MEF)','liner fit(MEH)',],
    ['linear fit(MEF)','liner fit(MEH)',],
    ['linear fit(MEF)','liner fit(MEH)',],
    ['linear fit(MEF)','liner fit(MEH)',],
    ['linear fit(MEF)','liner fit(MEH)',],
    ['linear fit(MEF)','liner fit(MEH)',],
    ['linear fit(MEF)','liner fit(MEH)',],
    ['linear fit(MEF)','liner fit(MEH)',],
    ['linear fit(MEF)','liner fit(MEH)',],
    ['linear fit(MEF)','liner fit(MEH)',],

]



x_ss=[
    [410,400,470,540,440,400,400,400,440,400,400,400,],#ZJS
    [1750,1770,1800,1770,1910,1770,1780,1770,1800,2050,2320,2330,],#YX
]
y_ss=[
    [20,20,20,20,20,20,20,20,20,20,20,20,],#ZJS
    [80,80,80,80,80,80,80,80,100,130,130,130,],#YX
]
records_lists=[zjs_records_list,yx_records_list]
predictions_lists=[zjs_predictions_list,yx_predictions_list]
station=['zjs','yx']
for s in range(len(records_lists)):
    records_list=records_lists[s]
    predictions_list=predictions_lists[s]
    x_s=x_ss[s]
    y_s=y_ss[s]
    print('x_s={}'.format(x_s))
    print('y_s={}'.format(y_s))
    plt.figure(figsize=(7.48,5.5))
    for j in range(len(records_list)):
        ax=plt.subplot(3,4,j+1, aspect='equal')
        xx,linear_list,xymin,xymax=compute_list_linear_fit(
            records_list=records_list[j],
            predictions_list=predictions_list[j],
        )
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
        if j in range(8,12):
            plt.xlabel('Predictands (' + r'$m^3/s$' +')', )
        if j in [0,4,8]:
            plt.ylabel('Records (' + r'$m^3/s$' + ')', )
        models=['linear fit (SF)','linear fit (SH)']
        scatters=['SF','SH']
        markers=['o','v',]
        markeredgecolor=['slategray','slategray']
        markerfacecolor=['tab:blue','tab:red']
        zorders=[1,0]
        plt.text(x_s[j],y_s[j],fig_id[j],fontweight='normal',zorder=2)
        for i in range(len(predictions_list[j])):
            # plt.plot(predictions_list[i], records_list[i],marker=markers[i], markerfacecolor='w',markeredgecolor='blue',markersize=4.5)

            plt.plot(
                predictions_list[j][i], 
                records_list[j][i],
                markers[i],
                # marker=markers[i],
                label=scatters[i],
                markersize=4.5,
                markerfacecolor=markerfacecolor[i],
                markeredgecolor=markeredgecolor[i],
                zorder=zorders[i])
            plt.plot(xx, linear_list[i], '--',color=markerfacecolor[i], label=models[i],linewidth=1.0,zorder=zorders[i])
        plt.plot([xymin,xymax], [xymin,xymax], '-', color='black', label='Ideal fit',linewidth=1.0)
        plt.xlim([xymin,xymax])
        plt.ylim([xymin,xymax])
        if j==1:
            plt.legend(
                        loc='upper left',
                        # bbox_to_anchor=(0.08,1.01, 1,0.101),
                        bbox_to_anchor=(-0.125,1.18),
                        ncol=5,
                        shadow=False,
                        frameon=True,
                        )

    plt.subplots_adjust(left=0.06, bottom=0.08, right=0.99,top=0.99, hspace=0.2, wspace=0.2)
    plt.savefig(graphs_path+station[s]+'_mef_meh_scatter.eps',format='EPS',dpi=2000)
    plt.savefig(graphs_path+station[s]+'_mef_meh_scatter.tif',format='TIFF',dpi=600)

#%%
#===============================================================================================================================================================================
ZJS_SF_VMD_HIND1 = pd.read_csv(root_path+'/zjs_vmd/projects/lstm-models-history/one_model_1_ahead_hindcast_pacf/model_test_results.csv')
ZJS_SF_VMD_FORE1 = pd.read_csv(root_path+'/zjs_vmd/projects/lstm-models-history/one_model_1_ahead_forecast_pacf/model_test_results.csv')
ZJS_SF_VMD_HIND3 = pd.read_csv(root_path+'/zjs_vmd/projects/lstm-models-history/one_model_3_ahead_hindcast_pacf/model_test_results.csv')
ZJS_SF_VMD_FORE3 = pd.read_csv(root_path+'/zjs_vmd/projects/lstm-models-history/one_model_3_ahead_forecast_pacf/model_test_results.csv')
ZJS_SF_VMD_HIND5 = pd.read_csv(root_path+'/zjs_vmd/projects/lstm-models-history/one_model_5_ahead_hindcast_pacf/model_test_results.csv')
ZJS_SF_VMD_FORE5 = pd.read_csv(root_path+'/zjs_vmd/projects/lstm-models-history/one_model_5_ahead_forecast_pacf/model_test_results.csv')
ZJS_SF_VMD_HIND7 = pd.read_csv(root_path+'/zjs_vmd/projects/lstm-models-history/one_model_7_ahead_hindcast_pacf/model_test_results.csv')
ZJS_SF_VMD_FORE7 = pd.read_csv(root_path+'/zjs_vmd/projects/lstm-models-history/one_model_7_ahead_forecast_pacf/model_test_results.csv')

ZJS_SF_EEMD_HIND1 = pd.read_csv(root_path+'/zjs_eemd/projects/lstm-models-history/one_model_1_ahead_hindcast_pacf/model_test_results.csv')
ZJS_SF_EEMD_FORE1 = pd.read_csv(root_path+'/zjs_eemd/projects/lstm-models-history/one_model_1_ahead_forecast_pacf/model_test_results.csv')
ZJS_SF_EEMD_HIND3 = pd.read_csv(root_path+'/zjs_eemd/projects/lstm-models-history/one_model_3_ahead_hindcast_pacf/model_test_results.csv')
ZJS_SF_EEMD_FORE3 = pd.read_csv(root_path+'/zjs_eemd/projects/lstm-models-history/one_model_3_ahead_forecast_pacf/model_test_results.csv')
ZJS_SF_EEMD_HIND5 = pd.read_csv(root_path+'/zjs_eemd/projects/lstm-models-history/one_model_5_ahead_hindcast_pacf/model_test_results.csv')
ZJS_SF_EEMD_FORE5 = pd.read_csv(root_path+'/zjs_eemd/projects/lstm-models-history/one_model_5_ahead_forecast_pacf/model_test_results.csv')
ZJS_SF_EEMD_HIND7 = pd.read_csv(root_path+'/zjs_eemd/projects/lstm-models-history/one_model_7_ahead_hindcast_pacf/model_test_results.csv')
ZJS_SF_EEMD_FORE7 = pd.read_csv(root_path+'/zjs_eemd/projects/lstm-models-history/one_model_7_ahead_forecast_pacf/model_test_results.csv')

ZJS_SF_DWT_HIND1 = pd.read_csv(root_path+'/zjs_wd/projects/lstm-models-history/db45-3/one_model_1_ahead_hindcast_pacf/model_test_results.csv')
ZJS_SF_DWT_FORE1 = pd.read_csv(root_path+'/zjs_wd/projects/lstm-models-history/db45-3/one_model_1_ahead_forecast_pacf/model_test_results.csv')
ZJS_SF_DWT_HIND3 = pd.read_csv(root_path+'/zjs_wd/projects/lstm-models-history/db45-3/one_model_3_ahead_hindcast_pacf/model_test_results.csv')
ZJS_SF_DWT_FORE3 = pd.read_csv(root_path+'/zjs_wd/projects/lstm-models-history/db45-3/one_model_3_ahead_forecast_pacf/model_test_results.csv')
ZJS_SF_DWT_HIND5 = pd.read_csv(root_path+'/zjs_wd/projects/lstm-models-history/db45-3/one_model_5_ahead_hindcast_pacf/model_test_results.csv')
ZJS_SF_DWT_FORE5 = pd.read_csv(root_path+'/zjs_wd/projects/lstm-models-history/db45-3/one_model_5_ahead_forecast_pacf/model_test_results.csv')
ZJS_SF_DWT_HIND7 = pd.read_csv(root_path+'/zjs_wd/projects/lstm-models-history/db45-3/one_model_7_ahead_hindcast_pacf/model_test_results.csv')
ZJS_SF_DWT_FORE7 = pd.read_csv(root_path+'/zjs_wd/projects/lstm-models-history/db45-3/one_model_7_ahead_forecast_pacf/model_test_results.csv')


YX_SF_VMD_HIND1 = pd.read_csv(root_path+'/yx_vmd/projects/lstm-models-history/one_model_1_ahead_hindcast_pacf/model_test_results.csv')
YX_SF_VMD_FORE1 = pd.read_csv(root_path+'/yx_vmd/projects/lstm-models-history/one_model_1_ahead_forecast_pacf/model_test_results.csv')
YX_SF_VMD_HIND3 = pd.read_csv(root_path+'/yx_vmd/projects/lstm-models-history/one_model_3_ahead_hindcast_pacf/model_test_results.csv')
YX_SF_VMD_FORE3 = pd.read_csv(root_path+'/yx_vmd/projects/lstm-models-history/one_model_3_ahead_forecast_pacf/model_test_results.csv')
YX_SF_VMD_HIND5 = pd.read_csv(root_path+'/yx_vmd/projects/lstm-models-history/one_model_5_ahead_hindcast_pacf/model_test_results.csv')
YX_SF_VMD_FORE5 = pd.read_csv(root_path+'/yx_vmd/projects/lstm-models-history/one_model_5_ahead_forecast_pacf/model_test_results.csv')
YX_SF_VMD_HIND7 = pd.read_csv(root_path+'/yx_vmd/projects/lstm-models-history/one_model_7_ahead_hindcast_pacf/model_test_results.csv')
YX_SF_VMD_FORE7 = pd.read_csv(root_path+'/yx_vmd/projects/lstm-models-history/one_model_7_ahead_forecast_pacf/model_test_results.csv')
YX_SF_EEMD_HIND1 = pd.read_csv(root_path+'/yx_eemd/projects/lstm-models-history/one_model_1_ahead_hindcast_pacf/model_test_results.csv')
YX_SF_EEMD_FORE1 = pd.read_csv(root_path+'/yx_eemd/projects/lstm-models-history/one_model_1_ahead_forecast_pacf/model_test_results.csv')
YX_SF_EEMD_HIND3 = pd.read_csv(root_path+'/yx_eemd/projects/lstm-models-history/one_model_3_ahead_hindcast_pacf/model_test_results.csv')
YX_SF_EEMD_FORE3 = pd.read_csv(root_path+'/yx_eemd/projects/lstm-models-history/one_model_3_ahead_forecast_pacf/model_test_results.csv')
YX_SF_EEMD_HIND5 = pd.read_csv(root_path+'/yx_eemd/projects/lstm-models-history/one_model_5_ahead_hindcast_pacf/model_test_results.csv')
YX_SF_EEMD_FORE5 = pd.read_csv(root_path+'/yx_eemd/projects/lstm-models-history/one_model_5_ahead_forecast_pacf/model_test_results.csv')
YX_SF_EEMD_HIND7 = pd.read_csv(root_path+'/yx_eemd/projects/lstm-models-history/one_model_7_ahead_hindcast_pacf/model_test_results.csv')
YX_SF_EEMD_FORE7 = pd.read_csv(root_path+'/yx_eemd/projects/lstm-models-history/one_model_7_ahead_forecast_pacf/model_test_results.csv')
YX_SF_DWT_HIND1 = pd.read_csv(root_path+'/yx_wd/projects/lstm-models-history/db45-3/one_model_1_ahead_hindcast_pacf/model_test_results.csv')
YX_SF_DWT_FORE1 = pd.read_csv(root_path+'/yx_wd/projects/lstm-models-history/db45-3/one_model_1_ahead_forecast_pacf/model_test_results.csv')
YX_SF_DWT_HIND3 = pd.read_csv(root_path+'/yx_wd/projects/lstm-models-history/db45-3/one_model_3_ahead_hindcast_pacf/model_test_results.csv')
YX_SF_DWT_FORE3 = pd.read_csv(root_path+'/yx_wd/projects/lstm-models-history/db45-3/one_model_3_ahead_forecast_pacf/model_test_results.csv')
YX_SF_DWT_HIND5 = pd.read_csv(root_path+'/yx_wd/projects/lstm-models-history/db45-3/one_model_5_ahead_hindcast_pacf/model_test_results.csv')
YX_SF_DWT_FORE5 = pd.read_csv(root_path+'/yx_wd/projects/lstm-models-history/db45-3/one_model_5_ahead_forecast_pacf/model_test_results.csv')
YX_SF_DWT_HIND7 = pd.read_csv(root_path+'/yx_wd/projects/lstm-models-history/db45-3/one_model_7_ahead_hindcast_pacf/model_test_results.csv')
YX_SF_DWT_FORE7 = pd.read_csv(root_path+'/yx_wd/projects/lstm-models-history/db45-3/one_model_7_ahead_forecast_pacf/model_test_results.csv')


zjs_records_list=[
    [ZJS_SF_VMD_FORE1['test_y'].values,ZJS_SF_VMD_HIND1['test_y'].values],
    [ZJS_SF_VMD_FORE3['test_y'].values,ZJS_SF_VMD_HIND3['test_y'].values],
    [ZJS_SF_VMD_FORE5['test_y'].values,ZJS_SF_VMD_HIND5['test_y'].values],
    [ZJS_SF_VMD_FORE7['test_y'].values,ZJS_SF_VMD_HIND7['test_y'].values],
    [ZJS_SF_EEMD_FORE1['test_y'].values,ZJS_SF_EEMD_HIND1['test_y'].values],
    [ZJS_SF_EEMD_FORE3['test_y'].values,ZJS_SF_EEMD_HIND3['test_y'].values],
    [ZJS_SF_EEMD_FORE5['test_y'].values,ZJS_SF_EEMD_HIND5['test_y'].values],
    [ZJS_SF_EEMD_FORE7['test_y'].values,ZJS_SF_EEMD_HIND7['test_y'].values],
    [ZJS_SF_DWT_FORE1['test_y'].values,ZJS_SF_DWT_HIND1['test_y'].values],
    [ZJS_SF_DWT_FORE3['test_y'].values,ZJS_SF_DWT_HIND3['test_y'].values],
    [ZJS_SF_DWT_FORE5['test_y'].values,ZJS_SF_DWT_HIND5['test_y'].values],
    [ZJS_SF_DWT_FORE7['test_y'].values,ZJS_SF_DWT_HIND7['test_y'].values],    
]
zjs_predictions_list=[
    [ZJS_SF_VMD_FORE1['test_pred'].values,ZJS_SF_VMD_HIND1['test_pred'].values],
    [ZJS_SF_VMD_FORE3['test_pred'].values,ZJS_SF_VMD_HIND3['test_pred'].values],
    [ZJS_SF_VMD_FORE5['test_pred'].values,ZJS_SF_VMD_HIND5['test_pred'].values],
    [ZJS_SF_VMD_FORE7['test_pred'].values,ZJS_SF_VMD_HIND7['test_pred'].values],
    [ZJS_SF_EEMD_FORE1['test_pred'].values,ZJS_SF_EEMD_HIND1['test_pred'].values],
    [ZJS_SF_EEMD_FORE3['test_pred'].values,ZJS_SF_EEMD_HIND3['test_pred'].values],
    [ZJS_SF_EEMD_FORE5['test_pred'].values,ZJS_SF_EEMD_HIND5['test_pred'].values],
    [ZJS_SF_EEMD_FORE7['test_pred'].values,ZJS_SF_EEMD_HIND7['test_pred'].values],
    [ZJS_SF_DWT_FORE1['test_pred'].values,ZJS_SF_DWT_HIND1['test_pred'].values],
    [ZJS_SF_DWT_FORE3['test_pred'].values,ZJS_SF_DWT_HIND3['test_pred'].values],
    [ZJS_SF_DWT_FORE5['test_pred'].values,ZJS_SF_DWT_HIND5['test_pred'].values],
    [ZJS_SF_DWT_FORE7['test_pred'].values,ZJS_SF_DWT_HIND7['test_pred'].values],  
]

yx_records_list=[
    [YX_SF_VMD_FORE1['test_y'].values,YX_SF_VMD_HIND1['test_y'].values],
    [YX_SF_VMD_FORE3['test_y'].values,YX_SF_VMD_HIND3['test_y'].values],
    [YX_SF_VMD_FORE5['test_y'].values,YX_SF_VMD_HIND5['test_y'].values],
    [YX_SF_VMD_FORE7['test_y'].values,YX_SF_VMD_HIND7['test_y'].values],
    [YX_SF_EEMD_FORE1['test_y'].values,YX_SF_EEMD_HIND1['test_y'].values],
    [YX_SF_EEMD_FORE3['test_y'].values,YX_SF_EEMD_HIND3['test_y'].values],
    [YX_SF_EEMD_FORE5['test_y'].values,YX_SF_EEMD_HIND5['test_y'].values],
    [YX_SF_EEMD_FORE7['test_y'].values,YX_SF_EEMD_HIND7['test_y'].values],
    [YX_SF_DWT_FORE1['test_y'].values,YX_SF_DWT_HIND1['test_y'].values],
    [YX_SF_DWT_FORE3['test_y'].values,YX_SF_DWT_HIND3['test_y'].values],
    [YX_SF_DWT_FORE5['test_y'].values,YX_SF_DWT_HIND5['test_y'].values],
    [YX_SF_DWT_FORE7['test_y'].values,YX_SF_DWT_HIND7['test_y'].values],    
]
yx_predictions_list=[
    [YX_SF_VMD_FORE1['test_pred'].values,YX_SF_VMD_HIND1['test_pred'].values],
    [YX_SF_VMD_FORE3['test_pred'].values,YX_SF_VMD_HIND3['test_pred'].values],
    [YX_SF_VMD_FORE5['test_pred'].values,YX_SF_VMD_HIND5['test_pred'].values],
    [YX_SF_VMD_FORE7['test_pred'].values,YX_SF_VMD_HIND7['test_pred'].values],
    [YX_SF_EEMD_FORE1['test_pred'].values,YX_SF_EEMD_HIND1['test_pred'].values],
    [YX_SF_EEMD_FORE3['test_pred'].values,YX_SF_EEMD_HIND3['test_pred'].values],
    [YX_SF_EEMD_FORE5['test_pred'].values,YX_SF_EEMD_HIND5['test_pred'].values],
    [YX_SF_EEMD_FORE7['test_pred'].values,YX_SF_EEMD_HIND7['test_pred'].values],
    [YX_SF_DWT_FORE1['test_pred'].values,YX_SF_DWT_HIND1['test_pred'].values],
    [YX_SF_DWT_FORE3['test_pred'].values,YX_SF_DWT_HIND3['test_pred'].values],
    [YX_SF_DWT_FORE5['test_pred'].values,YX_SF_DWT_HIND5['test_pred'].values],
    [YX_SF_DWT_FORE7['test_pred'].values,YX_SF_DWT_HIND7['test_pred'].values],  
]


fig_id=[
    '(a1)1-day ahead','(a2)3-day ahead','(a3)5-day ahead','(a4)7-day ahead',
    '(b1)1-day ahead','(b2)3-day ahead','(b3)5-day ahead','(b4)7-day ahead',
    '(c1)1-day ahead','(c2)3-day ahead','(c3)5-day ahead','(c4)7-day ahead',
]
models_labels=[
    ['SF-VMD-LSTM(1-day ahead)','SH-VMD-LSTM(1-day ahead)',],
    ['SF-VMD-LSTM(3-day ahead)','SH-VMD-LSTM(3-day ahead)',],
    ['SF-VMD-LSTM(5-day ahead)','SH-VMD-LSTM(5-day ahead)',],
    ['SF-VMD-LSTM(7-day ahead)','SH-VMD-LSTM(7-day ahead)',],
    ['SF-EEMD-LSTM(1-day ahead)','SH-EEMD-LSTM(1-day ahead)',],
    ['SF-EEMD-LSTM(3-day ahead)','SH-EEMD-LSTM(3-day ahead)',],
    ['SF-EEMD-LSTM(5-day ahead)','SH-EEMD-LSTM(5-day ahead)',],
    ['SF-EEMD-LSTM(7-day ahead)','SH-EEMD-LSTM(7-day ahead)',],
    ['SF-DWT-LSTM(db45-3,1-day ahead)','SH-DWT-LSTM(db45-3,1-day ahead)',],
    ['SF-DWT-LSTM(db45-3,3-day ahead)','SH-DWT-LSTM(db45-3,3-day ahead)',],
    ['SF-DWT-LSTM(db45-3,5-day ahead)','SH-DWT-LSTM(db45-3,5-day ahead)',],
    ['SF-DWT-LSTM(db45-3,7-day ahead)','SH-DWT-LSTM(db45-3,7-day ahead)',], 
]



#%%
x_ss=[
    [410,400,470,540,440,400,400,400,440,400,400,400,],#ZJS
    [1750,1770,1800,1770,1910,1770,1780,1770,1800,2050,2320,2330,],#YX
]
y_ss=[
    [20,20,20,20,20,20,20,20,20,20,20,20,],#ZJS
    [80,80,80,80,80,80,80,80,100,130,130,130,],#YX
]

records_lists=[zjs_records_list,yx_records_list]
predictions_lists=[zjs_predictions_list,yx_predictions_list]
station=['zjs','yx']
for s in range(len(records_lists)):
    records_list=records_lists[s]
    predictions_list=predictions_lists[s]
    x_s=x_ss[s]
    y_s=y_ss[s]
    print('x_s={}'.format(x_s))
    print('y_s={}'.format(y_s))
    plt.figure(figsize=(7.48,5.5))
    for j in range(len(records_list)):
        ax=plt.subplot(3,4,j+1, aspect='equal')
        xx,linear_list,xymin,xymax=compute_list_linear_fit(
            records_list=records_list[j],
            predictions_list=predictions_list[j],
        )
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
        if j in range(8,12):
            plt.xlabel('Predictands (' + r'$m^3/s$' +')', )
        if j in [0,4,8]:
            plt.ylabel('Records (' + r'$m^3/s$' + ')', )
        models=['linear fit(SF)','linear fit(SH)']
        scatters=['SF','SH']
        markers=['o','v',]
        markeredgecolor=['slategray','slategray']
        markerfacecolor=['tab:blue','tab:red']
        zorders=[1,0]
        plt.text(x_s[j],y_s[j],fig_id[j],fontweight='normal',zorder=2)
        for i in range(len(predictions_list[j])):
            # plt.plot(predictions_list[i], records_list[i],marker=markers[i], markerfacecolor='w',markeredgecolor='blue',markersize=4.5)

            plt.plot(
                predictions_list[j][i], 
                records_list[j][i],
                markers[i],
                # marker=markers[i],
                label=scatters[i],
                markersize=4.5,
                markerfacecolor=markerfacecolor[i],
                markeredgecolor=markeredgecolor[i],
                zorder=zorders[i])
            plt.plot(xx, linear_list[i], '--',color=markerfacecolor[i], label=models[i],linewidth=1.0,zorder=zorders[i])
        plt.plot([xymin,xymax], [xymin,xymax], '-', color='black', label='Ideal fit',linewidth=1.0)
        plt.xlim([xymin,xymax])
        plt.ylim([xymin,xymax])
        if j==1:
            plt.legend(
                        loc='upper left',
                        # bbox_to_anchor=(0.08,1.01, 1,0.101),
                        bbox_to_anchor=(-0.125,1.18),
                        ncol=5,
                        shadow=False,
                        frameon=True,
                        )

    plt.subplots_adjust(left=0.06, bottom=0.08, right=0.99,top=0.96, hspace=0.2, wspace=0.2)
    plt.savefig(graphs_path+station[s]+'_sf_sh_scatter.eps',format='EPS',dpi=2000)
    plt.savefig(graphs_path+station[s]+'_sf_sh_scatter.tif',format='TIFF',dpi=600)
# plt.show()
#%%
#===========================================================================================================================================================================

ZJS_SFMIS_VMD_HIND1 = pd.read_csv(root_path+'/zjs_vmd/projects/lstm-models-history/one_model_1_ahead_hindcast_pacf_mis/model_test_results.csv')
ZJS_SFMIS_VMD_FORE1 = pd.read_csv(root_path+'/zjs_vmd/projects/lstm-models-history/one_model_1_ahead_forecast_pacf_mis/model_test_results.csv')
ZJS_SFMIS_VMD_HIND3 = pd.read_csv(root_path+'/zjs_vmd/projects/lstm-models-history/one_model_3_ahead_hindcast_pacf_mis/model_test_results.csv')
ZJS_SFMIS_VMD_FORE3 = pd.read_csv(root_path+'/zjs_vmd/projects/lstm-models-history/one_model_3_ahead_forecast_pacf_mis/model_test_results.csv')
ZJS_SFMIS_VMD_HIND5 = pd.read_csv(root_path+'/zjs_vmd/projects/lstm-models-history/one_model_5_ahead_hindcast_pacf_mis/model_test_results.csv')
ZJS_SFMIS_VMD_FORE5 = pd.read_csv(root_path+'/zjs_vmd/projects/lstm-models-history/one_model_5_ahead_forecast_pacf_mis/model_test_results.csv')
ZJS_SFMIS_VMD_HIND7 = pd.read_csv(root_path+'/zjs_vmd/projects/lstm-models-history/one_model_7_ahead_hindcast_pacf_mis/model_test_results.csv')
ZJS_SFMIS_VMD_FORE7 = pd.read_csv(root_path+'/zjs_vmd/projects/lstm-models-history/one_model_7_ahead_forecast_pacf_mis/model_test_results.csv')

ZJS_SFMIS_EEMD_HIND1 = pd.read_csv(root_path+'/zjs_eemd/projects/lstm-models-history/one_model_1_ahead_hindcast_pacf_mis/model_test_results.csv')
ZJS_SFMIS_EEMD_FORE1 = pd.read_csv(root_path+'/zjs_eemd/projects/lstm-models-history/one_model_1_ahead_forecast_pacf_mis/model_test_results.csv')
ZJS_SFMIS_EEMD_HIND3 = pd.read_csv(root_path+'/zjs_eemd/projects/lstm-models-history/one_model_3_ahead_hindcast_pacf_mis/model_test_results.csv')
ZJS_SFMIS_EEMD_FORE3 = pd.read_csv(root_path+'/zjs_eemd/projects/lstm-models-history/one_model_3_ahead_forecast_pacf_mis/model_test_results.csv')
ZJS_SFMIS_EEMD_HIND5 = pd.read_csv(root_path+'/zjs_eemd/projects/lstm-models-history/one_model_5_ahead_hindcast_pacf_mis/model_test_results.csv')
ZJS_SFMIS_EEMD_FORE5 = pd.read_csv(root_path+'/zjs_eemd/projects/lstm-models-history/one_model_5_ahead_forecast_pacf_mis/model_test_results.csv')
ZJS_SFMIS_EEMD_HIND7 = pd.read_csv(root_path+'/zjs_eemd/projects/lstm-models-history/one_model_7_ahead_hindcast_pacf_mis/model_test_results.csv')
ZJS_SFMIS_EEMD_FORE7 = pd.read_csv(root_path+'/zjs_eemd/projects/lstm-models-history/one_model_7_ahead_forecast_pacf_mis/model_test_results.csv')

ZJS_SFMIS_DWT_HIND1 = pd.read_csv(root_path+'/zjs_wd/projects/lstm-models-history/db45-3/one_model_1_ahead_hindcast_pacf_mis/model_test_results.csv')
ZJS_SFMIS_DWT_FORE1 = pd.read_csv(root_path+'/zjs_wd/projects/lstm-models-history/db45-3/one_model_1_ahead_forecast_pacf_mis/model_test_results.csv')
ZJS_SFMIS_DWT_HIND3 = pd.read_csv(root_path+'/zjs_wd/projects/lstm-models-history/db45-3/one_model_3_ahead_hindcast_pacf_mis/model_test_results.csv')
ZJS_SFMIS_DWT_FORE3 = pd.read_csv(root_path+'/zjs_wd/projects/lstm-models-history/db45-3/one_model_3_ahead_forecast_pacf_mis/model_test_results.csv')
ZJS_SFMIS_DWT_HIND5 = pd.read_csv(root_path+'/zjs_wd/projects/lstm-models-history/db45-3/one_model_5_ahead_hindcast_pacf_mis/model_test_results.csv')
ZJS_SFMIS_DWT_FORE5 = pd.read_csv(root_path+'/zjs_wd/projects/lstm-models-history/db45-3/one_model_5_ahead_forecast_pacf_mis/model_test_results.csv')
ZJS_SFMIS_DWT_HIND7 = pd.read_csv(root_path+'/zjs_wd/projects/lstm-models-history/db45-3/one_model_7_ahead_hindcast_pacf_mis/model_test_results.csv')
ZJS_SFMIS_DWT_FORE7 = pd.read_csv(root_path+'/zjs_wd/projects/lstm-models-history/db45-3/one_model_7_ahead_forecast_pacf_mis/model_test_results.csv')


YX_SFMIS_VMD_HIND1 = pd.read_csv(root_path+'/yx_vmd/projects/lstm-models-history/one_model_1_ahead_hindcast_pacf_mis/model_test_results.csv')
YX_SFMIS_VMD_FORE1 = pd.read_csv(root_path+'/yx_vmd/projects/lstm-models-history/one_model_1_ahead_forecast_pacf_mis/model_test_results.csv')
YX_SFMIS_VMD_HIND3 = pd.read_csv(root_path+'/yx_vmd/projects/lstm-models-history/one_model_3_ahead_hindcast_pacf_mis/model_test_results.csv')
YX_SFMIS_VMD_FORE3 = pd.read_csv(root_path+'/yx_vmd/projects/lstm-models-history/one_model_3_ahead_forecast_pacf_mis/model_test_results.csv')
YX_SFMIS_VMD_HIND5 = pd.read_csv(root_path+'/yx_vmd/projects/lstm-models-history/one_model_5_ahead_hindcast_pacf_mis/model_test_results.csv')
YX_SFMIS_VMD_FORE5 = pd.read_csv(root_path+'/yx_vmd/projects/lstm-models-history/one_model_5_ahead_forecast_pacf_mis/model_test_results.csv')
YX_SFMIS_VMD_HIND7 = pd.read_csv(root_path+'/yx_vmd/projects/lstm-models-history/one_model_7_ahead_hindcast_pacf_mis/model_test_results.csv')
YX_SFMIS_VMD_FORE7 = pd.read_csv(root_path+'/yx_vmd/projects/lstm-models-history/one_model_7_ahead_forecast_pacf_mis/model_test_results.csv')
YX_SFMIS_EEMD_HIND1 = pd.read_csv(root_path+'/yx_eemd/projects/lstm-models-history/one_model_1_ahead_hindcast_pacf_mis/model_test_results.csv')
YX_SFMIS_EEMD_FORE1 = pd.read_csv(root_path+'/yx_eemd/projects/lstm-models-history/one_model_1_ahead_forecast_pacf_mis/model_test_results.csv')
YX_SFMIS_EEMD_HIND3 = pd.read_csv(root_path+'/yx_eemd/projects/lstm-models-history/one_model_3_ahead_hindcast_pacf_mis/model_test_results.csv')
YX_SFMIS_EEMD_FORE3 = pd.read_csv(root_path+'/yx_eemd/projects/lstm-models-history/one_model_3_ahead_forecast_pacf_mis/model_test_results.csv')
YX_SFMIS_EEMD_HIND5 = pd.read_csv(root_path+'/yx_eemd/projects/lstm-models-history/one_model_5_ahead_hindcast_pacf_mis/model_test_results.csv')
YX_SFMIS_EEMD_FORE5 = pd.read_csv(root_path+'/yx_eemd/projects/lstm-models-history/one_model_5_ahead_forecast_pacf_mis/model_test_results.csv')
YX_SFMIS_EEMD_HIND7 = pd.read_csv(root_path+'/yx_eemd/projects/lstm-models-history/one_model_7_ahead_hindcast_pacf_mis/model_test_results.csv')
YX_SFMIS_EEMD_FORE7 = pd.read_csv(root_path+'/yx_eemd/projects/lstm-models-history/one_model_7_ahead_forecast_pacf_mis/model_test_results.csv')
YX_SFMIS_DWT_HIND1 = pd.read_csv(root_path+'/yx_wd/projects/lstm-models-history/db45-3/one_model_1_ahead_hindcast_pacf_mis/model_test_results.csv')
YX_SFMIS_DWT_FORE1 = pd.read_csv(root_path+'/yx_wd/projects/lstm-models-history/db45-3/one_model_1_ahead_forecast_pacf_mis/model_test_results.csv')
YX_SFMIS_DWT_HIND3 = pd.read_csv(root_path+'/yx_wd/projects/lstm-models-history/db45-3/one_model_3_ahead_hindcast_pacf_mis/model_test_results.csv')
YX_SFMIS_DWT_FORE3 = pd.read_csv(root_path+'/yx_wd/projects/lstm-models-history/db45-3/one_model_3_ahead_forecast_pacf_mis/model_test_results.csv')
YX_SFMIS_DWT_HIND5 = pd.read_csv(root_path+'/yx_wd/projects/lstm-models-history/db45-3/one_model_5_ahead_hindcast_pacf_mis/model_test_results.csv')
YX_SFMIS_DWT_FORE5 = pd.read_csv(root_path+'/yx_wd/projects/lstm-models-history/db45-3/one_model_5_ahead_forecast_pacf_mis/model_test_results.csv')
YX_SFMIS_DWT_HIND7 = pd.read_csv(root_path+'/yx_wd/projects/lstm-models-history/db45-3/one_model_7_ahead_hindcast_pacf_mis/model_test_results.csv')
YX_SFMIS_DWT_FORE7 = pd.read_csv(root_path+'/yx_wd/projects/lstm-models-history/db45-3/one_model_7_ahead_forecast_pacf_mis/model_test_results.csv')


zjs_records_list=[
    [ZJS_SFMIS_VMD_FORE1['test_y'].values,ZJS_SFMIS_VMD_HIND1['test_y'].values],
    [ZJS_SFMIS_VMD_FORE3['test_y'].values,ZJS_SFMIS_VMD_HIND3['test_y'].values],
    [ZJS_SFMIS_VMD_FORE5['test_y'].values,ZJS_SFMIS_VMD_HIND5['test_y'].values],
    [ZJS_SFMIS_VMD_FORE7['test_y'].values,ZJS_SFMIS_VMD_HIND7['test_y'].values],
    [ZJS_SFMIS_EEMD_FORE1['test_y'].values,ZJS_SFMIS_EEMD_HIND1['test_y'].values],
    [ZJS_SFMIS_EEMD_FORE3['test_y'].values,ZJS_SFMIS_EEMD_HIND3['test_y'].values],
    [ZJS_SFMIS_EEMD_FORE5['test_y'].values,ZJS_SFMIS_EEMD_HIND5['test_y'].values],
    [ZJS_SFMIS_EEMD_FORE7['test_y'].values,ZJS_SFMIS_EEMD_HIND7['test_y'].values],
    [ZJS_SFMIS_DWT_FORE1['test_y'].values,ZJS_SFMIS_DWT_HIND1['test_y'].values],
    [ZJS_SFMIS_DWT_FORE3['test_y'].values,ZJS_SFMIS_DWT_HIND3['test_y'].values],
    [ZJS_SFMIS_DWT_FORE5['test_y'].values,ZJS_SFMIS_DWT_HIND5['test_y'].values],
    [ZJS_SFMIS_DWT_FORE7['test_y'].values,ZJS_SFMIS_DWT_HIND7['test_y'].values],    
]
zjs_predictions_list=[
    [ZJS_SFMIS_VMD_FORE1['test_pred'].values,ZJS_SFMIS_VMD_HIND1['test_pred'].values],
    [ZJS_SFMIS_VMD_FORE3['test_pred'].values,ZJS_SFMIS_VMD_HIND3['test_pred'].values],
    [ZJS_SFMIS_VMD_FORE5['test_pred'].values,ZJS_SFMIS_VMD_HIND5['test_pred'].values],
    [ZJS_SFMIS_VMD_FORE7['test_pred'].values,ZJS_SFMIS_VMD_HIND7['test_pred'].values],
    [ZJS_SFMIS_EEMD_FORE1['test_pred'].values,ZJS_SFMIS_EEMD_HIND1['test_pred'].values],
    [ZJS_SFMIS_EEMD_FORE3['test_pred'].values,ZJS_SFMIS_EEMD_HIND3['test_pred'].values],
    [ZJS_SFMIS_EEMD_FORE5['test_pred'].values,ZJS_SFMIS_EEMD_HIND5['test_pred'].values],
    [ZJS_SFMIS_EEMD_FORE7['test_pred'].values,ZJS_SFMIS_EEMD_HIND7['test_pred'].values],
    [ZJS_SFMIS_DWT_FORE1['test_pred'].values,ZJS_SFMIS_DWT_HIND1['test_pred'].values],
    [ZJS_SFMIS_DWT_FORE3['test_pred'].values,ZJS_SFMIS_DWT_HIND3['test_pred'].values],
    [ZJS_SFMIS_DWT_FORE5['test_pred'].values,ZJS_SFMIS_DWT_HIND5['test_pred'].values],
    [ZJS_SFMIS_DWT_FORE7['test_pred'].values,ZJS_SFMIS_DWT_HIND7['test_pred'].values],  
]

yx_records_list=[
    [YX_SFMIS_VMD_FORE1['test_y'].values,YX_SFMIS_VMD_HIND1['test_y'].values],
    [YX_SFMIS_VMD_FORE3['test_y'].values,YX_SFMIS_VMD_HIND3['test_y'].values],
    [YX_SFMIS_VMD_FORE5['test_y'].values,YX_SFMIS_VMD_HIND5['test_y'].values],
    [YX_SFMIS_VMD_FORE7['test_y'].values,YX_SFMIS_VMD_HIND7['test_y'].values],
    [YX_SFMIS_EEMD_FORE1['test_y'].values,YX_SFMIS_EEMD_HIND1['test_y'].values],
    [YX_SFMIS_EEMD_FORE3['test_y'].values,YX_SFMIS_EEMD_HIND3['test_y'].values],
    [YX_SFMIS_EEMD_FORE5['test_y'].values,YX_SFMIS_EEMD_HIND5['test_y'].values],
    [YX_SFMIS_EEMD_FORE7['test_y'].values,YX_SFMIS_EEMD_HIND7['test_y'].values],
    [YX_SFMIS_DWT_FORE1['test_y'].values,YX_SFMIS_DWT_HIND1['test_y'].values],
    [YX_SFMIS_DWT_FORE3['test_y'].values,YX_SFMIS_DWT_HIND3['test_y'].values],
    [YX_SFMIS_DWT_FORE5['test_y'].values,YX_SFMIS_DWT_HIND5['test_y'].values],
    [YX_SFMIS_DWT_FORE7['test_y'].values,YX_SFMIS_DWT_HIND7['test_y'].values],    
]
yx_predictions_list=[
    [YX_SFMIS_VMD_FORE1['test_pred'].values,YX_SFMIS_VMD_HIND1['test_pred'].values],
    [YX_SFMIS_VMD_FORE3['test_pred'].values,YX_SFMIS_VMD_HIND3['test_pred'].values],
    [YX_SFMIS_VMD_FORE5['test_pred'].values,YX_SFMIS_VMD_HIND5['test_pred'].values],
    [YX_SFMIS_VMD_FORE7['test_pred'].values,YX_SFMIS_VMD_HIND7['test_pred'].values],
    [YX_SFMIS_EEMD_FORE1['test_pred'].values,YX_SFMIS_EEMD_HIND1['test_pred'].values],
    [YX_SFMIS_EEMD_FORE3['test_pred'].values,YX_SFMIS_EEMD_HIND3['test_pred'].values],
    [YX_SFMIS_EEMD_FORE5['test_pred'].values,YX_SFMIS_EEMD_HIND5['test_pred'].values],
    [YX_SFMIS_EEMD_FORE7['test_pred'].values,YX_SFMIS_EEMD_HIND7['test_pred'].values],
    [YX_SFMIS_DWT_FORE1['test_pred'].values,YX_SFMIS_DWT_HIND1['test_pred'].values],
    [YX_SFMIS_DWT_FORE3['test_pred'].values,YX_SFMIS_DWT_HIND3['test_pred'].values],
    [YX_SFMIS_DWT_FORE5['test_pred'].values,YX_SFMIS_DWT_HIND5['test_pred'].values],
    [YX_SFMIS_DWT_FORE7['test_pred'].values,YX_SFMIS_DWT_HIND7['test_pred'].values],  
]


fig_id=[
    '(a1)','(a2)','(a3)','(a4)',
    '(b1)','(b2)','(b3)','(b4)',
    '(c1)','(c2)','(c3)','(c4)',
]
models_labels=[
    ['SF-VMD-LSTM(1-day ahead)','SH-VMD-LSTM(1-day ahead)',],
    ['SF-VMD-LSTM(3-day ahead)','SH-VMD-LSTM(3-day ahead)',],
    ['SF-VMD-LSTM(5-day ahead)','SH-VMD-LSTM(5-day ahead)',],
    ['SF-VMD-LSTM(7-day ahead)','SH-VMD-LSTM(7-day ahead)',],
    ['SF-EEMD-LSTM(1-day ahead)','SH-EEMD-LSTM(1-day ahead)',],
    ['SF-EEMD-LSTM(3-day ahead)','SH-EEMD-LSTM(3-day ahead)',],
    ['SF-EEMD-LSTM(5-day ahead)','SH-EEMD-LSTM(5-day ahead)',],
    ['SF-EEMD-LSTM(7-day ahead)','SH-EEMD-LSTM(7-day ahead)',],
    ['SF-DWT-LSTM(db45-3,1-day ahead)','SH-DWT-LSTM(db45-3,1-day ahead)',],
    ['SF-DWT-LSTM(db45-3,3-day ahead)','SH-DWT-LSTM(db45-3,3-day ahead)',],
    ['SF-DWT-LSTM(db45-3,5-day ahead)','SH-DWT-LSTM(db45-3,5-day ahead)',],
    ['SF-DWT-LSTM(db45-3,7-day ahead)','SH-DWT-LSTM(db45-3,7-day ahead)',], 
]
models_labels=[
    ['linear fit(SFMIS)','liner fit(SHMIS)',],
    ['linear fit(SFMIS)','liner fit(SHMIS)',],
    ['linear fit(SFMIS)','liner fit(SHMIS)',],
    ['linear fit(SFMIS)','liner fit(SHMIS)',],
    ['linear fit(SFMIS)','liner fit(SHMIS)',],
    ['linear fit(SFMIS)','liner fit(SHMIS)',],
    ['linear fit(SFMIS)','liner fit(SHMIS)',],
    ['linear fit(SFMIS)','liner fit(SHMIS)',],
    ['linear fit(SFMIS)','liner fit(SHMIS)',],
    ['linear fit(SFMIS)','liner fit(SHMIS)',],
    ['linear fit(SFMIS)','liner fit(SHMIS)',],
    ['linear fit(SFMIS)','liner fit(SHMIS)',],

]


#%%
x_ss=[
    [410,400,470,540,440,400,400,400,440,400,400,400,],#ZJS
    [1750,1770,1800,1770,1910,1770,1780,1770,1800,2050,2320,2330,],#YX
]
y_ss=[
    [20,20,20,20,20,20,20,20,20,20,20,20,],#ZJS
    [80,80,80,80,80,80,80,80,100,130,130,130,],#YX
]
records_lists=[zjs_records_list,yx_records_list]
predictions_lists=[zjs_predictions_list,yx_predictions_list]
station=['zjs','yx']
for s in range(len(records_lists)):
    records_list=records_lists[s]
    predictions_list=predictions_lists[s]
    x_s=x_ss[s]
    y_s=y_ss[s]
    print('x_s={}'.format(x_s))
    print('y_s={}'.format(y_s))
    plt.figure(figsize=(7.48,5.5))
    for j in range(len(records_list)):
        ax=plt.subplot(3,4,j+1, aspect='equal')
        xx,linear_list,xymin,xymax=compute_list_linear_fit(
            records_list=records_list[j],
            predictions_list=predictions_list[j],
        )
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
        if j in range(8,12):
            plt.xlabel('Predictands (' + r'$m^3/s$' +')', )
        if j in [0,4,8]:
            plt.ylabel('Records (' + r'$m^3/s$' + ')', )
        models=['linear fit(SF)','linear fit(SH)']
        scatters=['SF','SH']
        markers=['o','v',]
        markeredgecolor=['slategray','slategray']
        markerfacecolor=['tab:blue','tab:red']
        zorders=[1,0]
        plt.text(x_s[j],y_s[j],fig_id[j],fontweight='normal',zorder=2)
        for i in range(len(predictions_list[j])):
            # plt.plot(predictions_list[i], records_list[i],marker=markers[i], markerfacecolor='w',markeredgecolor='blue',markersize=4.5)

            plt.plot(
                predictions_list[j][i], 
                records_list[j][i],
                markers[i],
                # marker=markers[i],
                label=scatters[i],
                markersize=4.5,
                markerfacecolor=markerfacecolor[i],
                markeredgecolor=markeredgecolor[i],
                zorder=zorders[i])
            plt.plot(xx, linear_list[i], '--',color=markerfacecolor[i], label=models[i],linewidth=1.0,zorder=zorders[i])
        plt.plot([xymin,xymax], [xymin,xymax], '-', color='black', label='Ideal fit',linewidth=1.0)
        plt.xlim([xymin,xymax])
        plt.ylim([xymin,xymax])
        if j==1:
            plt.legend(
                        loc='upper left',
                        # bbox_to_anchor=(0.08,1.01, 1,0.101),
                        bbox_to_anchor=(-0.125,1.18),
                        ncol=5,
                        shadow=False,
                        frameon=True,
                        )

    plt.subplots_adjust(left=0.06, bottom=0.08, right=0.99,top=0.99, hspace=0.2, wspace=0.2)
    plt.savefig(graphs_path+station[s]+'_sfmis_shmis_scatter.eps',format='EPS',dpi=2000)
    plt.savefig(graphs_path+station[s]+'_sfmis_shmis_scatter.tif',format='TIFF',dpi=600)
plt.show()