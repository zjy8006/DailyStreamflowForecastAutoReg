import pandas as pd
import os
root_path = os.path.dirname(os.path.abspath('__file__'))

pacf_data = pd.read_csv(root_path+'/zjs_vmd/data/PACF.csv')
up_bounds=pacf_data['UP']
lo_bounds=pacf_data['LOW']
subsignals_pacf = pacf_data.drop(['ORIG','UP','LOW'],axis=1)
lags_dict={}
for signal in subsignals_pacf.columns.tolist():
    # print(subsignals_pacf[signal])
    lag=0
    for i in range(subsignals_pacf[signal].shape[0]):
        if abs(subsignals_pacf[signal][i])>0.5 and abs(subsignals_pacf[signal][i])>up_bounds[0]:
            lag=i
    lags_dict[signal]=lag    


variables={
    'lags_dict':{'vmd':lags_dict},
    'full_len' :6574,
    'train_len' :5260,
    'dev_len' : 657,
    'test_len' : 657,
}
print('variables:{}'.format(variables))

