import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, mean_squared_log_error
import os
root_path = os.path.os.path.dirname(os.path.abspath('__file__'))
# root_path = os.path.abspath(os.path.join(root_path, os.path.pardir))
print(10*'-'+' Root Path: {}'.format(root_path))


data1 = pd.read_csv(root_path+'/zjs_wd/data/db10-3/multi_models_1_ahead_forecast/minmax_unsample_dev_s1.csv')
data2 = pd.read_csv(root_path+'/zjs_wd/data/db10-3/multi_models_1_ahead_forecast_pacf/minmax_unsample_dev_s1.csv')

print(sum((data1-data1).sum()))
