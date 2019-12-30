import numpy as np
import pandas as pd
import tensorflow as tf
import math
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, mean_squared_log_error
import os
current_path = os.path.dirname(os.path.abspath(__file__))
parent_path = os.path.abspath(os.path.join(current_path, os.path.pardir))
grandpa_path = os.path.abspath(os.path.join(parent_path, os.path.pardir))
data_path = parent_path + '\\data\\'

def PPTS(records,preds,gamma):
    """ 
    Compute peak percentage threshold statistic
    args:
        records:the observed records
        preds:the predictions
        gamma:lower value percentage
    """
    # records
    r = pd.DataFrame(records,columns=['r'])
    # print('original time series:\n{}'.format(r))
    # predictions
    p = pd.DataFrame(preds,columns=['p'])
    # print('predicted time series:\n{}'.format(p))
    # The number of samples
    N = r['r'].size
    # print('series size={}'.format(N))
    # The number of top data
    G = round((gamma/100)*N)
    rp = pd.concat([r,p],axis=1)
    rps=rp.sort_values(by=['r'],ascending=False)
    rps_g=rps.iloc[:G]
    records = (rps_g['r']).values
    preds = (rps_g['p']).values
    abss=np.abs((records-preds)/records*100)
    # print('abs error={}'.format(abss))
    sums = np.sum(abss)
    # print('sum of abs error={}'.format(abss))
    ppts = sums*(1/((gamma/100)*N))
    # print('ppts('+str(gamma)+'%)={}'.format(ppts))
    return ppts

def normalized_mean_square_error(y_true,y_pred):
    # print(type(y_true))
    # print(type(y_pred))
    if type(y_true)==list:
        y_true=pd.Series(y_true)
    if type(y_pred)==list:
        y_pred=pd.Series(y_pred)
    if type(y_true)==np.array:
        y_true=pd.Series(y_true)
    if type(y_pred)==np.array:
        y_pred=pd.Series(y_pred)
    # print(type(y_true))
    # print(type(y_pred))
    
    assert y_true.shape[0]==y_pred.shape[0]

    avg_y_true = sum(y_true)/y_true.shape[0]
    avg_y_pred = sum(y_pred)/y_pred.shape[0]
    nmse1=sum(((y_true-y_pred)**2)/(avg_y_true*avg_y_pred))/y_true.shape[0]
    nmse2=y_true.shape[0]*sum((y_true-y_pred)**2)/(sum(y_true)*sum(y_pred))
    # print('NMSE1={}'.format(nmse1))
    # print('NMSE2={}'.format(nmse2))
    # assert nmse1==nmse2
    return nmse1

def normalized_root_mean_square_error(y_true,y_pred):
    # print(type(y_true))
    # print(type(y_pred))
    if type(y_true)==list:
        y_true=pd.Series(y_true)
    if type(y_pred)==list:
        y_pred=pd.Series(y_pred)
    if type(y_true)==np.array:
        y_true=pd.Series(y_true)
    if type(y_pred)==np.array:
        y_pred=pd.Series(y_pred)
    # print(type(y_true))
    # print(type(y_pred))
    
    assert y_true.shape[0]==y_pred.shape[0]
    nrmse = math.sqrt(mean_squared_error(y_true, y_pred))/(sum(y_true)/len(y_true))
    return nrmse
    

if __name__ == '__main__':
    # data = pd.read_excel(current_path+'\\gbr-models\\gbr_pred_lstm.xlsx')
    # data = pd.read_excel(current_path+'\\gbr-models'+ MODEL_NAME+'.xlsx')
    # # test_data_size=4325
    # y_test = data['y_train'][1:657+1]
    # test_predictions=data['train_pred'][1:test_data_size+1]
    # # print(y_test)
    # ppts5 = PPTS(y_test.values,test_predictions.values,5)
    # # print('ppts5={}'.format(ppts5))
    # ppts15 = PPTS(y_test.values,test_predictions.values,15)
    # # print('ppts15={}'.format(ppts15))
    # ppts20 = PPTS(y_test.values,test_predictions.values,20)
    # # print('ppts20={}'.format(ppts20))
    # ppts25 = PPTS(y_test.values,test_predictions.values,25)
    # print('ppts25={}'.format(ppts25))


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
    a=[1,2,3,4,5]
    b=[1.1,2.1,3.1,4.1,5.1]
    nmse = normalized_mean_square_error(y_true=a,y_pred=b)
    print(nmse)