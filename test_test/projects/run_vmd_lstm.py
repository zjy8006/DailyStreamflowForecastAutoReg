import os
root_path = os.path.dirname(os.path.abspath('__file__'))
# root_path = os.path.abspath(os.path.join(root_path, os.path.pardir))
# root_path = os.path.abspath(os.path.join(root_path, os.path.pardir))
print(10*'-'+' Root Path: {}'.format(root_path))

import sys
sys.path.append(root_path+'/tools/')
from build_lstm import my_lstm

lev=3
# for k in range(1,lev+1):
for i in [8,16,24,32]:
    for j in [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,]:
        my_lstm(
            path=root_path+'\\test_test',
            lev=lev,
            pattern='one_model_3_ahead_forecast_pacf',
            LR=0.07,
            HU1=i,
            DR1=j,
            MODEL_ID=None,
        )



