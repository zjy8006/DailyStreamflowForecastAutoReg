import numpy as np
import pandas as pd
import os
current_path = os.path.dirname(os.path.abspath(__file__))
parent_path = os.path.abspath(os.path.join(current_path, os.path.pardir))
grandpa_path = os.path.abspath(os.path.join(parent_path, os.path.pardir))
data_path = parent_path + '\\data\\'

def MAE(records,preds,gamma_1,gamma_2):
    """ 
    Compute peak percentage threshold statistic
    args:
        records:the observed records
        preds:the predictions
        gamma:lower value percentage
    """
    # records
    r = pd.DataFrame(records,columns=['r'])
    print('original time series:\n{}'.format(r))
    # predictions
    p = pd.DataFrame(preds,columns=['p'])
    print('predicted time series:\n{}'.format(p))
    # The number of samples
    N = r['r'].size
    print('series size={}'.format(N))
    # The number of top data
    G_1 = round((gamma_1/100)*N)
    G_2 = round((gamma_2/100)*N)
    rp = pd.concat([r,p],axis=1)
    rps=rp.sort_values(by=['r'],ascending=False)
    rps_g_1=rps.iloc[:G_1]
    rps_g_2=rps.iloc[:G_2]
    records_1 = (rps_g_1['r']).values
    records_2 = (rps_g_2['r']).values
    preds_1 = (rps_g_1['p']).values
    preds_2 = (rps_g_2['p']).values
    abss_1=np.abs((records_1-preds_1))
    abss_2=np.abs((records_2-preds_2))
    print('abs error={}'.format(abss_1))
    print('abs error={}'.format(abss_2))
    sums_1 = np.sum(abss_1)
    sums_2 = np.sum(abss_2)
    print('sum of abs1 error={}'.format(abss_1))
    print('sum of abs2 error={}'.format(abss_2))
    mae = (sums_2-sums_1)*(1/(((gamma_2/100)*N)-((gamma_1/100)*N)))
    print('mae('+str(gamma_1)+'%)={}'.format(mae))
    return mae

if __name__ == '__main__':
   """ # data = pd.read_excel(current_path+'\\gbr-models\\gbr_pred_lstm.xlsx')
    data = pd.read_excel(current_path+'\\gbr-models'+ MODEL_NAME+'.xlsx')
    # test_data_size=4325
    y_test = data['y_train'][1:657+1]
    test_predictions=data['train_pred'][1:test_data_size+1]
    # print(y_test)
    ppts5 = PPTS(y_test.values,test_predictions.values,5)
    # print('ppts5={}'.format(ppts5))
    ppts15 = PPTS(y_test.values,test_predictions.values,15)
    # print('ppts15={}'.format(ppts15))
    ppts20 = PPTS(y_test.values,test_predictions.values,20)
    # print('ppts20={}'.format(ppts20))
    ppts25 = PPTS(y_test.values,test_predictions.values,25)
    # print('ppts25={}'.format(ppts25))"""


    # data = pd.read_excel(current_path+'\\e-svr-models\\gbr_pred_e_svr.xlsx')
    # # test_data_size=4325
    # y_test = data['y_train'][1:test_data_size+1]
    # test_predictions=data['train_pred'][1:test_data_size+1]
    # # print(y_test)
    # ppts = PPTS(y_test.values,test_predictions.values,5)
    # # print(ppts)

    # data = pd.read_excel(current_path+'\\bpnn-models\\gbr_pred_bpnn.xlsx')
    # # test_data_size=4325
    # y_test = data['y_train'][1:test_data_size+1]
    # test_predictions=data['train_pred'][1:test_data_size+1]
    # # print(y_test)
    # ppts = PPTS(y_test.values,test_predictions.values,5)
    # # print(ppts)
    # print('='*100)
    # data = pd.read_csv(current_path+'\\lstm-models\\lstm_ensemble_test_result.csv')
    # # test_data_size=4325
    # y_test = data['orig']
    # test_predictions=data['pred']
    # # print(y_test)
    # ppts5 = PPTS(y_test.values,test_predictions.values,5)
    # # print('ppts5={}'.format(ppts5))
    # ppts15 = PPTS(y_test.values,test_predictions.values,15)
    # # print('ppts15={}'.format(ppts15))
    # ppts20 = PPTS(y_test.values,test_predictions.values,20)
    # # print('ppts20={}'.format(ppts20))
    # ppts25 = PPTS(y_test.values,test_predictions.values,25)
    # # print('ppts25={}'.format(ppts25))
