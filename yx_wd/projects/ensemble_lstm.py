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

for wavelet in [
    # 'bior 3.3','coif3','db2','db5','db10','db15','db20','db25','db30','db35','db40',
    'db45',
    # 'haar',
    ]:
    for lev in [
        # 1,2,
        3]:
        for lead in [1,3,5,7]:
            # ensemble_optimization(
            #     root_path=root_path,
            #     station='yx',
            #     decomposer='wd',
            #     lev=lev,
            #     variables=variables,
            #     orig_df=orig,
            #     pattern='multi_models_'+str(lead)+'_ahead_forecast_pacf',
            #     wavelet=wavelet,
            # )
            # ensemble_optimization(
            #     root_path=root_path,
            #     station='yx',
            #     decomposer='wd',
            #     lev=lev,
            #     variables=variables,
            #     orig_df=orig,
            #     pattern='one_model_'+str(lead)+'_ahead_forecast_pacf',
            #     wavelet=wavelet,
            # )
            # ensemble_optimization(
            #     root_path=root_path,
            #     station='yx',
            #     decomposer='wd',
            #     lev=lev,
            #     variables=variables,
            #     orig_df=orig,
            #     pattern='one_model_'+str(lead)+'_ahead_forecast_pacf_mis',
            #     wavelet=wavelet,
            # )

            ensemble_optimization(
                root_path=root_path,
                station='yx',
                decomposer='wd',
                lev=lev,
                variables=variables,
                orig_df=orig,
                pattern='one_model_'+str(lead)+'_ahead_hindcast_pacf_mis',
                wavelet=wavelet,
            )