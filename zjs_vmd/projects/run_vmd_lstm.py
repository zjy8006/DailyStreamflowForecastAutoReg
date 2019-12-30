import os
root_path = os.path.dirname(os.path.abspath('__file__'))
# root_path = os.path.abspath(os.path.join(root_path, os.path.pardir))
# root_path = os.path.abspath(os.path.join(root_path, os.path.pardir))
print(10*'-'+' Root Path: {}'.format(root_path))

import sys
sys.path.append(root_path+'/tools/')
from build_lstm import my_lstm
for lr in [0.0001,0.0003,0.0007,0.001,0.003,0.007,0.01,0.03,0.07,0.1]:
    my_lstm(
        path=root_path+'\\zjs_vmd\\',
        lev=11,
        pattern='learning_rate_tuning_nmse',
        EPS=500,
        # HL=1,
        HU1=8,
        DR1=0.0,
        # HU2=hu2,
        # DR2=dr2,
        LR=lr,#0.0001,0.0003,0.0007,0.001,0.003,0.007,0.01,0.03,0.07,0.1
        MODEL_ID=1,
        EARLY_STOPING = False,
        loss='custom_loss',
    )

# for hu1 in [8,16,24,32]:
#     for hu2 in [8,16,24,32]:
# for dr1 in [0.0,0.1,0.2,0.3,0.4,0.5,]:
#     for dr2 in [0.0,0.1,0.2,0.3,0.4,0.5,]:
#         my_lstm(
#             path=root_path+'\\zjs_vmd\\',
#             lev=11,
#             pattern='multi_models_1_ahead_forecast_pacf',
#             HL=2,
#             HU1=32,
#             HU2=32,
#             DR1=dr1,
#             DR2=dr2,
#             LR=0.01,#0.0001,0.0003,0.0007,0.001,0.003,0.007,0.01,0.03,0.07,0.1
#             MODEL_ID=1,
#         )


# lev=11
# for k in range(1,lev+1):
#     for i in [8,16,24,32]:
#         for j in [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,]:
#             my_lstm(
#                 path=root_path+'\\zjs_vmd\\',
#                 lev=lev,
#                 pattern='multi_models_5_ahead_hindcast_pacf',
#                 HU1=i,
#                 DR1=j,
#                 MODEL_ID=k,
#             )

# for lead in [1,3,5,7]:
#         for i in [8,16,24,32]:
#             for j in [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,]:
#                 my_lstm(
#                     path=root_path+'/zjs_vmd/',
#                     lev=lev,
#                     pattern='one_model_'+str(lead)+'_ahead_hindcast_pacf_loss_nmse',
#                     HU1=i,
#                     DR1=j,
#                     MODEL_ID=None,
#                 )

# test loss of nmse:
# for lead in [1,3,5,7]:
#         for i in [8,16,24,32]:
#             for j in [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,]:
#                 my_lstm(
#                     path=root_path+'/zjs_vmd/',
#                     lev=lev,
#                     pattern='one_model_'+str(lead)+'_ahead_forecast_pacf',
#                     HU1=i,
#                     DR1=j,
#                     LR=0.01,
#                     MODEL_ID=None,
#                 )



