import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, mean_squared_log_error
import os
root_path = os.path.os.path.dirname(os.path.abspath('__file__'))
# root_path = os.path.abspath(os.path.join(root_path, os.path.pardir))
# root_path = os.path.abspath(os.path.join(root_path, os.path.pardir))
print(10*'-'+' Root Path: {}'.format(root_path))

import sys
sys.path.append(root_path+'/tools/')
from plot_utils import plot_rela_pred,plot_history,plot_error_distribution,plot_subsignals_preds
from ensemble import ensemble_optimization
from metrics_ import PPTS
from variables import variables

orig = pd.read_excel(root_path+'/time_series/YangXianDailyFlow1997-2014.xlsx')['DailyFlow']
for lead in [1,3,5,7]:
    ensemble_optimization(#multi-models ensemble forecasting
        root_path=root_path,
        station='yx',
        decomposer='eemd',
        lev=12,
        variables=variables,
        orig_df=orig,
        pattern='multi_models_'+str(lead)+'_ahead_forecast_pacf',
    )
    ensemble_optimization(#single-model forecasting
        root_path=root_path,
        station='yx',
        decomposer='eemd',
        lev=12,
        variables=variables,
        orig_df=orig,
        pattern='one_model_'+str(lead)+'_ahead_forecast_pacf',
    )
    ensemble_optimization(#single-model forecasting with most influential subsignals
        root_path=root_path,
        station='yx',
        decomposer='eemd',
        lev=12,
        variables=variables,
        orig_df=orig,
        pattern='one_model_'+str(lead)+'_ahead_forecast_pacf_mis',
    )
    ensemble_optimization(#multi-models ensemble hindcasting
        root_path=root_path,
        station='yx',
        decomposer='eemd',
        lev=12,
        variables=variables,
        orig_df=orig,
        pattern='multi_models_'+str(lead)+'_ahead_hindcast_pacf',
    )
    ensemble_optimization(#single-model hindcasting
        root_path=root_path,
        station='yx',
        decomposer='eemd',
        lev=12,
        variables=variables,
        orig_df=orig,
        pattern='one_model_'+str(lead)+'_ahead_hindcast_pacf',
    )
    ensemble_optimization(#single-model hindcasting with most influential subsignals
        root_path=root_path,
        station='yx',
        decomposer='eemd',
        lev=12,
        variables=variables,
        orig_df=orig,
        pattern='one_model_'+str(lead)+'_ahead_hindcast_pacf_mis',
    )

 

    
    
