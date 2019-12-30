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

for leading_time in [1,3,5,7]:
    ensemble_optimization(
        root_path=root_path,
        station='zjs',
        variables=variables,
        orig_df=orig,
        pattern=str(leading_time)+'_ahead',
    )
    