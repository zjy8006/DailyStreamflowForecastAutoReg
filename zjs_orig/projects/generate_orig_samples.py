import pandas as pd
import numpy as np

import os
root_path = os.path.dirname(os.path.abspath('__file__'))
root_path = os.path.abspath(os.path.join(root_path, os.path.pardir))
root_path = os.path.abspath(os.path.join(root_path, os.path.pardir))
print(10*'-'+' Current Path: {}'.format(root_path))
import sys
sys.path.append(root_path+'/tools/')
from samples_generator import gen_samples
from variables import variables

for leading_time in [1,3,5,7]:
    gen_samples(
        data_path=root_path+'/time_series/ZhangJiaShanDailyFlow1997-2014.xlsx',
        save_path=root_path+'/zjs_orig/data/',
        column='DailyFlow',
        lag=variables['lags_dict']['orig'],
        test_len=657,
        leading_time=leading_time,
    )

