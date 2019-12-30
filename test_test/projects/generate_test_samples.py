import pandas as pd
import numpy as np

import os
root_path = os.path.dirname(os.path.abspath('__file__'))
# root_path = os.path.abspath(os.path.join(root_path, os.path.pardir))
# root_path = os.path.abspath(os.path.join(root_path, os.path.pardir))
print(10*'-'+' Current Path: {}'.format(root_path))
import sys
sys.path.append(root_path+'/tools/')
from samples_generator import gen_multi_models_hindcast_samples
from samples_generator import gen_multi_models_forecast_samples
from samples_generator import gen_one_model_hindcast_samples
from samples_generator import gen_one_model_forecast_samples
from samples_generator import gen_one_model_long_leading_forecast_pcr
from samples_generator import gen_one_model_long_leading_forecast_pacf
from variables import variables

gen_multi_models_hindcast_samples(
    path=root_path+'/test_test/data/',
    decomposer='test',
    lev=3,
    test_len=5,
    lags_dict=variables['lags_dict'],
)

gen_multi_models_forecast_samples(
    path=root_path+'/test_test/data/',
    decomposer='test',
    lev=3,
    test_len=5,
    start_id=17,
    stop_id=26,
    lags_dict=variables['lags_dict'],
)

gen_one_model_hindcast_samples(
    path=root_path+'/test_test/data/',
    decomposer='test',
    lev=3,
    test_len=5,
    lags_dict=variables['lags_dict'],
)
gen_one_model_forecast_samples(
    path=root_path+'/test_test/data/',
    decomposer='test',
    lev=3,
    test_len=5,
    start_id=17,
    stop_id=26,
    lags_dict=variables['lags_dict'],
)
gen_one_model_long_leading_forecast_pcr(
    path=root_path+'/test_test/data/',
    decomposer='test',
    lev=3,
    test_len=5,
    start_id=17,
    stop_id=26,
    lags_dict=variables['lags_dict'],
    leading_time=3,
    pre_times=5,

)
gen_one_model_long_leading_forecast_pacf(
    path=root_path+'/test_test/data/',
    decomposer='test',
    lev=3,
    test_len=5,
    start_id=17,
    stop_id=26,
    lags_dict=variables['lags_dict'],
    leading_time=3,
)

