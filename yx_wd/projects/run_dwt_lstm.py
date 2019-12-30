import os
root_path = os.path.dirname(os.path.abspath('__file__'))

import sys
sys.path.append(root_path+'/tools/')
from build_lstm import my_lstm

# wavelet='db2'
# lev=2 #lev=1,2,3
# for k in range(1,lev+2):
#     for i in [8,16,24,32]:
#         for j in [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,]:
#             my_lstm(
#                 path=root_path+'/yx_wd/',
#                 lev=lev,
#                 pattern='multi_models_7_ahead_forecast_pacf',
#                 HU1=i,
#                 DR1=j,
#                 MODEL_ID=k,
#                 wavelet=wavelet,
#             )
# for wavelet in ['bior 3.3','coif3','db2','db5','db10','db15','db20','db25','db30','db35','db40','db45','haar']:
#     for lev in [1,2,3]:
#         for lead in [1,3,5,7]:
#             for i in [8,16,24,32]:
#                 for j in [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,]:
#                     my_lstm(
#                         path=root_path+'/yx_wd/',
#                         lev=lev,
#                         pattern='one_model_'+str(lead)+'_ahead_forecast_pacf',
#                         HU1=i,
#                         DR1=j,
#                         MODEL_ID=None,
#                         wavelet=wavelet,
#                     )
for lead in [1,3,5,7]:
    for i in [8,16,24,32]:
        for j in [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,]:
            my_lstm(
                path=root_path+'/yx_wd/',
                lev=3,
                pattern='one_model_'+str(lead)+'_ahead_hindcast_pacf_mis',
                HU1=i,
                DR1=j,
                MODEL_ID=None,
                wavelet='db45',
            )


